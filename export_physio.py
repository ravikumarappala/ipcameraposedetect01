"""
export_physio.py
================
Download physio_measurements from Firestore and export to CSV
(open directly in Excel or Google Sheets).

Usage:
  # Export most recent run (auto-discovers latest runId)
  conda run -n posedetect python export_physio.py

  # Export a specific run
  conda run -n posedetect python export_physio.py --run-id run-20260304-032349-32ca06

  # Export all runs
  conda run -n posedetect python export_physio.py --all

Output: physio_export_{runId}.csv  (in current directory)
"""

import os
import sys
import json
import csv
import argparse
import subprocess
import requests
import datetime

GCP_PROJECT = os.environ.get("FIRESTORE_PROJECT", "ml-ai-001-414605")
COLLECTION  = "physio_measurements"
FS_BASE     = (
    f"https://firestore.googleapis.com/v1/"
    f"projects/{GCP_PROJECT}/databases/(default)/documents"
)

# ── Auth ───────────────────────────────────────────────────────────────────────
def _token():
    t = os.environ.get("GOOGLE_ACCESS_TOKEN", "").strip()
    if t:
        return t
    try:
        return subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sys.exit("ERROR: no GCP token. Run: gcloud auth login")

# ── Firestore value decoder ────────────────────────────────────────────────────
def _decode(v):
    """Recursively decode a Firestore REST value object to Python."""
    if "nullValue"    in v: return None
    if "booleanValue" in v: return v["booleanValue"]
    if "integerValue" in v: return int(v["integerValue"])
    if "doubleValue"  in v: return float(v["doubleValue"])
    if "stringValue"  in v: return v["stringValue"]
    if "arrayValue"   in v:
        return [_decode(i) for i in v["arrayValue"].get("values", [])]
    if "mapValue"     in v:
        fields = v["mapValue"].get("fields") or {}  # guard against None / missing
        return {k: _decode(vv) for k, vv in fields.items()}
    return str(v)


def _get_doc(run_id: str, token: str) -> dict:
    url  = f"{FS_BASE}/{COLLECTION}/{run_id}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if resp.status_code != 200:
        sys.exit(f"ERROR {resp.status_code}: {resp.text[:300]}")
    raw = resp.json()
    return {k: _decode(v) for k, v in raw.get("fields", {}).items()}

def _list_docs(token: str) -> list:
    url  = f"{FS_BASE}/{COLLECTION}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if resp.status_code != 200:
        sys.exit(f"ERROR {resp.status_code}: {resp.text[:300]}")
    docs = resp.json().get("documents", [])
    return [d["name"].rsplit("/", 1)[-1] for d in docs]

# ── CSV export ─────────────────────────────────────────────────────────────────
# ── CSV column spec (dense — every row populated) ─────────────────────────────
# Core identity (stamped on every row)
COLUMNS = [
    "runId", "date", "timestamp",
    # Measurement identity
    "group", "group_name", "code", "sub",
    # Generic values (normalized by _norm() in step_physio)
    "left_value", "right_value", "diff_value", "value_unit",
    # Clinical result
    "auto_result_left", "auto_result_right",
    "manual_required", "asymmetry",
    # Options / dropdown
    "options_str",
    # Traceability
    "source", "note",
]

def _flat_row(row: dict) -> dict:
    """Return a row dict with exactly COLUMNS populated — no blanks."""
    out = {}
    for col in COLUMNS:
        val = row.get(col, "")
        if val is None:
            val = ""
        elif isinstance(val, bool):
            val = str(val)
        elif isinstance(val, (dict, list)):
            import json
            val = json.dumps(val, ensure_ascii=False)
        out[col] = val
    return out


def export_run(run_id: str, token: str):
    print(f"  Downloading {COLLECTION}/{run_id} ...")
    doc      = _get_doc(run_id, token)
    flat_tbl = doc.get("flat_table", [])

    out_file = f"physio_export_{run_id}.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in flat_tbl:
            writer.writerow(_flat_row(row))

    print(f"  ✓ Saved: {out_file}  ({len(flat_tbl)} rows)")
    return out_file


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export physio Firestore data to CSV")
    parser.add_argument("--run-id", default=None, help="Specific run ID to export")
    parser.add_argument("--all",    action="store_true", help="Export all runs")
    args = parser.parse_args()

    token = _token()

    if args.all:
        run_ids = _list_docs(token)
        print(f"Found {len(run_ids)} run(s)")
        for rid in run_ids:
            export_run(rid, token)

    elif args.run_id:
        export_run(args.run_id, token)

    else:
        # Auto pick the latest run
        run_ids = sorted(_list_docs(token), reverse=True)
        if not run_ids:
            sys.exit("No documents found in physio_measurements collection.")
        print(f"Auto-selecting latest run: {run_ids[0]}")
        f = export_run(run_ids[0], token)
        print(f"\n  Open in Excel:  open {f}")

if __name__ == "__main__":
    main()
