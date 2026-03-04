"""
cf_logger.py
============
Sends pipeline run and step data to the GCP exec-logger Cloud Function.

Firestore structure created:
  runs/{runId}                  — created by create_run()
  steps/{runId}/{stepNum}       — one doc per step, created by log_step()
  summary/{runId}               — final summary, created by log_summary()

GCS path for file attachments:
  {date}/{timestamp}/{runId}/{stepNum}/{filename}

Disable CF calls: export CF_ENABLED=false
Custom URL:       export CF_FUNC_URL=https://...
"""

import os
import uuid
import datetime
import requests

FUNC_URL = os.environ.get(
    "CF_FUNC_URL",
    "https://exec-logger-7jstnd322q-uc.a.run.app"
)
CF_TIMEOUT = int(os.environ.get("CF_TIMEOUT", "30"))


class CFLogger:
    """
    Sends exactly one Firestore document per pipeline event:
      create_run()   → /run   (1 doc)
      log_step() ×7 → /step  (7 docs, one per step)
      log_summary()  → /step  (1 summary doc, stepNum=99)
    """

    def __init__(self, run_id: str = None, enabled: bool = True):
        env_flag = os.environ.get("CF_ENABLED", "true").lower()
        self.enabled = enabled and (env_flag not in ("false", "0", "no"))

        now = datetime.datetime.utcnow()
        self.run_id  = run_id or f"run-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.date    = now.strftime("%Y-%m-%d")
        self.ts      = now.strftime("%H%M%S")

        if self.enabled:
            print(f"  [CF] enabled  run_id={self.run_id}  url={FUNC_URL}")
        else:
            print("  [CF] disabled (set CF_ENABLED=true to enable)")

    # ── Public API ─────────────────────────────────────────────────────────────

    def create_run(self, program: str, cmd: str, options: dict = None):
        """POST /run — register this execution. Creates one Firestore doc."""
        if not self.enabled:
            return
        self._post_json("/run", {
            "runId":   self.run_id,
            "date":    self.date,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "program": program,
            "cmd":     cmd,
            "options": options or {},
            "status":  "started",
        })

    def log_step(self, step_num: int, input_text: str, output_text: str,
                 status: str = "ok", file_path: str = None, file_mime: str = None):
        """
        POST /step — one Firestore doc per step.
        If file_path exists, uploads it to GCS at:
          {date}/{timestamp}/{runId}/{stepNum}/{filename}
        and stores the gcsUri in the Firestore doc.
        """
        if not self.enabled:
            return

        data = {
            "runId":   self.run_id,
            "date":    self.date,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stepNum": step_num,
            "input":   str(input_text)[:2000],
            "output":  str(output_text)[:2000],
            "status":  status,
        }

        if file_path and os.path.isfile(file_path):
            basename = os.path.basename(file_path)
            # GCS path: date/timestamp/runId/stepNum/filename
            gcs_name = f"{self.date}/{self.ts}/{self.run_id}/{step_num}/{basename}"
            mime = file_mime or _guess_mime(file_path)
            try:
                with open(file_path, "rb") as fh:
                    self._post_multipart("/step", data, gcs_name, fh, mime)
            except Exception as e:
                print(f"  [CF] WARNING: could not attach {file_path}: {e}")
                self._post_json("/step", data)
        else:
            self._post_json("/step", data)

    def log_summary(self, measurements: dict, height_cm: float,
                    csv_path: str = None, json_path: str = None):
        """
        POST /step with stepNum=99 — compact summary doc.
        Attaches summary.csv to GCS.
        """
        if not self.enabled:
            return

        flat = {k: round(v["cm"], 1) for k, v in measurements.items()
                if isinstance(v, dict) and "cm" in v}

        data = {
            "runId":     self.run_id,
            "date":      self.date,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stepNum":   99,
            "input":     "pipeline complete — final summary",
            "output":    f"height={height_cm}cm | " +
                         " | ".join(f"{k}={v}cm" for k, v in list(flat.items())[:8]),
            "status":    "summary",
            "summary":   flat,
        }

        attach = csv_path if (csv_path and os.path.isfile(csv_path)) else json_path
        if attach and os.path.isfile(attach):
            basename = os.path.basename(attach)
            gcs_name = f"{self.date}/{self.ts}/{self.run_id}/summary/{basename}"
            mime = "text/csv" if attach.endswith(".csv") else "application/json"
            try:
                with open(attach, "rb") as fh:
                    self._post_multipart("/step", data, gcs_name, fh, mime)
            except Exception as e:
                print(f"  [CF] WARNING: could not attach summary file: {e}")
                self._post_json("/step", data)
        else:
            self._post_json("/step", data)

    def log_summary_doc(self, csv_path: str):
        """
        POST /step stepNum=100 — full structured Firestore document.

        Sends as application/json (NOT multipart) so Firestore stores
        the nested arrays and maps correctly, not as Python repr strings.

        Firestore document fields:
          runId, date, timestamp, stepNum=100, status="summary_doc"
          joint_positions : list of {joint, raw_x,y,z, fit_x,y,z}
          joint_lengths   : list of {label, left_cm, right_cm, diff_cm}
          joint_angles    : list of {label, left_deg, right_deg, diff_deg}
          metadata        : dict of {label: value}
        """
        if not self.enabled:
            return
        if not csv_path or not os.path.isfile(csv_path):
            print(f"  [CF] WARNING: log_summary_doc — csv not found: {csv_path}")
            return

        parsed = _parse_summary_csv(csv_path)

        # ── Send as pure JSON so Firestore gets proper arrays/maps ────────────
        # (multipart data= sends nested Python objects as repr strings — broken)
        payload = {
            "runId":           self.run_id,
            "date":            self.date,
            "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
            "stepNum":         100,
            "input":           "full structured summary parsed from summary.csv",
            "output":          (
                f"{len(parsed['joint_positions'])} joints | "
                f"{len(parsed['joint_lengths'])} length measurements | "
                f"{len(parsed['joint_angles'])} angles"
            ),
            "status":          "summary_doc",
            # ── structured table data ──────────────────────────────────────────
            "joint_positions": parsed["joint_positions"],   # list of dicts
            "joint_lengths":   parsed["joint_lengths"],     # list of dicts
            "joint_angles":    parsed["joint_angles"],      # list of dicts
            "metadata":        parsed["metadata"],          # flat dict
        }
        self._post_json("/step", payload)


    def log_physio_doc(self, physio_data: dict):
        """
        POST /step stepNum=101 — physio-measurements Firestore document.

        Sends as application/json so Firestore stores nested arrays/maps
        correctly (groups 1-5, flat_table rows, asymmetry_summary).

        physio_data: output of step_physio.compute_physio_measurements()
        """
        if not self.enabled:
            return

        flagged = physio_data.get("asymmetry_summary", {}).get("total_flagged", 0)
        flat    = physio_data.get("flat_table", [])

        payload = {
            "runId":             self.run_id,
            "date":              self.date,
            "timestamp":         datetime.datetime.utcnow().isoformat() + "Z",
            "stepNum":           101,
            "input":             "physio measurements from SMPL joint data",
            "output":            (
                f"{len(flat)} measurements across 5 groups | "
                f"{flagged} asymmetry flag(s) detected"
            ),
            "status":            "physio_measurements",
            # ── 5-group structured table ──────────────────────────────────────
            "groups":            physio_data.get("groups", {}),
            "flat_table":        flat,           # single list, one row per code
            "asymmetry_summary": physio_data.get("asymmetry_summary", {}),
        }
        self._post_json("/step", payload)

    def complete_run(self, status: str = "complete"):
        """POST /run — update status to complete."""
        if not self.enabled:
            return
        self._post_json("/run", {
            "runId":       self.run_id,
            "program":     "run_pipeline.py",
            "cmd":         "run_pipeline.py",   # required by CF /run endpoint
            "status":      status,
            "completedAt": datetime.datetime.utcnow().isoformat() + "Z",
        })


    # ── Internal helpers ────────────────────────────────────────────────────────

    def _post_json(self, path: str, payload: dict):
        try:
            r = requests.post(f"{FUNC_URL}{path}", json=payload, timeout=CF_TIMEOUT)
            if r.status_code >= 400:
                print(f"  [CF] WARNING: {path} → {r.status_code}: {r.text[:200]}")
            else:
                print(f"  [CF] ✓ {path} stepNum={payload.get('stepNum', '-')} → {r.status_code}")
        except Exception as e:
            print(f"  [CF] WARNING: {path} failed: {e}")

    def _post_multipart(self, path: str, data: dict,
                        gcs_fname: str, fh, mime: str):
        try:
            r = requests.post(
                f"{FUNC_URL}{path}",
                data=data,
                files={"file": (gcs_fname, fh, mime)},
                timeout=CF_TIMEOUT,
            )
            if r.status_code >= 400:
                print(f"  [CF] WARNING: {path} multipart → {r.status_code}: {r.text[:200]}")
            else:
                print(f"  [CF] ✓ {path} stepNum={data.get('stepNum', '-')} +{os.path.basename(gcs_fname)} → {r.status_code}")
        except Exception as e:
            print(f"  [CF] WARNING: {path} multipart failed: {e}")


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv":  "text/csv",
        ".tsv":  "text/tab-separated-values",
        ".json": "application/json",
        ".npz":  "application/octet-stream",
        ".npy":  "application/octet-stream",
        ".pkl":  "application/octet-stream",
        ".txt":  "text/plain",
    }.get(ext, "application/octet-stream")


def _safe_float(val: str):
    """Convert string to float, return None if empty or invalid."""
    try:
        return float(val.strip()) if val.strip() else None
    except ValueError:
        return None


def _parse_summary_csv(csv_path: str) -> dict:
    """
    Parse summary.csv into 4 structured sections:

      joint_positions: list of dicts
        { joint, raw_x, raw_y, raw_z, fit_x, fit_y, fit_z }

      joint_lengths: list of dicts
        { label, left_cm, right_cm, diff_cm }

      joint_angles: list of dicts
        { label, left_deg, right_deg, diff_deg }

      metadata: dict
        { label: value }
    """
    import csv as _csv

    joint_positions = []
    joint_lengths   = []
    joint_angles    = []
    metadata        = {}

    # Which section are we in?
    SECTION_JOINTS   = "joints"
    SECTION_LENGTHS  = "lengths"
    SECTION_ANGLES   = "angles"
    SECTION_META     = "meta"
    section = None

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = _csv.reader(f)
        for row in reader:
            # Skip completely blank rows (they separate sections)
            if not any(cell.strip() for cell in row):
                continue

            header = row[0].strip()

            # ── Section headers ───────────────────────────────────
            if header == "Joint" and len(row) >= 7:
                section = SECTION_JOINTS
                continue
            if header == "Joint Length":
                section = SECTION_LENGTHS
                continue
            if header == "Joint Angles":
                section = SECTION_ANGLES
                continue
            if header in ("RMS-Difference", "Length between Cameras",
                          "Height of Human-calculated", "Height from Mesh",
                          "Height of Human-user-provided"):
                section = SECTION_META

            # ── Data rows ─────────────────────────────────────────
            if section == SECTION_JOINTS:
                while len(row) < 7:
                    row.append("")
                joint_positions.append({
                    "joint":  header,
                    "raw_x":  _safe_float(row[1]),
                    "raw_y":  _safe_float(row[2]),
                    "raw_z":  _safe_float(row[3]),
                    "fit_x":  _safe_float(row[4]),
                    "fit_y":  _safe_float(row[5]),
                    "fit_z":  _safe_float(row[6]),
                })

            elif section == SECTION_LENGTHS:
                while len(row) < 4:
                    row.append("")
                joint_lengths.append({
                    "label":    header,
                    "left_cm":  _safe_float(row[1]),
                    "right_cm": _safe_float(row[2]),
                    "diff_cm":  _safe_float(row[3]),
                })

            elif section == SECTION_ANGLES:
                while len(row) < 4:
                    row.append("")
                joint_angles.append({
                    "label":     header,
                    "left_deg":  _safe_float(row[1]),
                    "right_deg": _safe_float(row[2]),
                    "diff_deg":  _safe_float(row[3]),
                })

            elif section == SECTION_META:
                val = _safe_float(row[1]) if len(row) > 1 else None
                metadata[header] = val

    return {
        "joint_positions": joint_positions,
        "joint_lengths":   joint_lengths,
        "joint_angles":    joint_angles,
        "metadata":        metadata,
    }

