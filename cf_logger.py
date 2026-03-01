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
        POST /step with stepNum=99 — summary doc in Firestore.
        Attaches summary.csv to GCS if it exists.
        measurements: dict of {label: {'cm': float, 'in': float}}
        """
        if not self.enabled:
            return

        # Build flat measurement dict for Firestore (label → cm value)
        flat = {k: round(v["cm"], 1) for k, v in measurements.items()
                if isinstance(v, dict) and "cm" in v}

        data = {
            "runId":   self.run_id,
            "date":    self.date,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stepNum": 99,
            "input":   "pipeline complete — final summary",
            "output":  f"height={height_cm}cm | " +
                       " | ".join(f"{k}={v}cm" for k, v in list(flat.items())[:8]),
            "status":  "summary",
            "summary": flat,
        }

        # Attach summary.csv if present
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

    def complete_run(self, status: str = "complete"):
        """POST /run — update status to complete."""
        if not self.enabled:
            return
        self._post_json("/run", {
            "runId":       self.run_id,
            "program":     "run_pipeline.py",   # required by CF /run endpoint
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
