"""
firestore_writer.py
===================
Writes physio measurements directly to a dedicated Firestore collection
using the Firestore REST API (no google-cloud-firestore SDK needed —
avoids protobuf conflict with mediapipe).

Collection : physio_measurements
Document   : {runId}

Auth: uses a short-lived identity token from `gcloud auth print-access-token`
      OR the env variable GOOGLE_ACCESS_TOKEN if pre-supplied.

Set project: export FIRESTORE_PROJECT=ml-ai-001-414605
"""

import os
import subprocess
import datetime
import json
import requests

GCP_PROJECT = os.environ.get("FIRESTORE_PROJECT", "ml-ai-001-414605")
COLLECTION   = "physio_measurements"
FS_BASE      = (
    f"https://firestore.googleapis.com/v1/"
    f"projects/{GCP_PROJECT}/databases/(default)/documents"
)


# ── Auth token ─────────────────────────────────────────────────────────────────
def _get_token() -> str:
    """Get GCP access token from env var or gcloud CLI."""
    token = os.environ.get("GOOGLE_ACCESS_TOKEN", "").strip()
    if token:
        return token
    try:
        out = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return ""


# ── Firestore value encoding ───────────────────────────────────────────────────
def _to_fs_value(v):
    """Convert Python value to Firestore REST API value object."""
    if v is None:
        return {"nullValue": None}
    if isinstance(v, bool):
        return {"booleanValue": v}
    if isinstance(v, int):
        return {"integerValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, str):
        return {"stringValue": v}
    if isinstance(v, dict):
        return {"mapValue": {"fields": {k: _to_fs_value(vv) for k, vv in v.items()}}}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_to_fs_value(i) for i in v]}}
    # Fallback: stringify numpy types etc.
    try:
        import numpy as np
        if isinstance(v, np.integer): return {"integerValue": str(int(v))}
        if isinstance(v, np.floating): return {"doubleValue": float(v)}
        if isinstance(v, np.ndarray): return _to_fs_value(v.tolist())
    except ImportError:
        pass
    return {"stringValue": str(v)}


def _build_fs_doc(fields: dict) -> dict:
    return {"fields": {k: _to_fs_value(v) for k, v in fields.items()}}


# ── Public API ─────────────────────────────────────────────────────────────────
def write_physio_measurements(run_id: str, date: str, physio_data: dict) -> bool:
    """
    Write physio measurements to Firestore collection `physio_measurements`.
    Document ID = run_id  (PATCH with updateMask so it's idempotent).

    Returns True on success, False on any error (non-blocking).
    """
    token = _get_token()
    if not token:
        print("  [Firestore] WARNING: no auth token — run `gcloud auth print-access-token`")
        return False

    ts  = datetime.datetime.utcnow().isoformat() + "Z"

    # Stamp runId + date + timestamp on every flat_table row
    # so each row is completely self-contained (Excel-friendly)
    flat_table = []
    for row in physio_data.get("flat_table", []):
        stamped = {
            "runId":     run_id,
            "date":      date,
            "timestamp": ts,
            **row,
        }
        flat_table.append(stamped)

    fields = {
        "runId":             run_id,
        "date":              date,
        "timestamp":         ts,
        "project_id":        GCP_PROJECT,
        "groups":            physio_data.get("groups", {}),
        "flat_table":        flat_table,          # rows now include runId/date/timestamp
        "asymmetry_summary": physio_data.get("asymmetry_summary", {}),
    }


    url  = f"{FS_BASE}/{COLLECTION}/{run_id}"
    body = _build_fs_doc(fields)

    try:
        resp = requests.patch(
            url, json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            timeout=30,
        )
        if resp.status_code in (200, 201):
            print(f"  [Firestore] ✓ {COLLECTION}/{run_id} → {resp.status_code} (project: {GCP_PROJECT})")
            return True
        else:
            print(f"  [Firestore] WARNING: {resp.status_code} — {resp.text[:300]}")
            return False
    except Exception as e:
        print(f"  [Firestore] WARNING: REST write failed — {e}")
        return False


# ── Mesh physio landmark table ─────────────────────────────────────────────────
MESH_COLLECTION = "human_mesh_physio_specified_measurements"

def write_physio_mesh_doc(run_id: str, date: str,
                          placed_landmarks: dict,
                          measurements: dict,
                          physio_data: dict,
                          image_gcs_path: str = "") -> bool:
    """
    Write physio mesh landmark data to a dedicated Firestore collection:
      human_mesh_physio_specified_measurements/{runId}

    Fields:
      runId, date, timestamp
      image_gcs_path   — GCS path of physio_annotated.png
      landmarks        — map of {label → {number, side, smpl_joint, x_mm, y_mm, z_mm}}
      measurements     — flat body measurements {name → cm}
      asymmetry_summary — flagged items from physio_data
    """
    token = _get_token()
    if not token:
        print("  [Firestore] WARNING: no auth token for mesh physio write")
        return False

    # Flatten measurements to {label: cm_value}
    meas_flat = {k: v.get("cm") for k, v in measurements.items()
                 if isinstance(v, dict) and "cm" in v}

    fields = {
        "runId":             run_id,
        "date":              date,
        "timestamp":         datetime.datetime.utcnow().isoformat() + "Z",
        "project_id":        GCP_PROJECT,
        "image_gcs_path":    image_gcs_path,
        "landmarks":         placed_landmarks,     # {label → metadata dict}
        "measurements_cm":   meas_flat,            # flat {name: cm}
        "asymmetry_summary": physio_data.get("asymmetry_summary", {}),
        "total_landmarks":   len(placed_landmarks),
    }

    url  = f"{FS_BASE}/{MESH_COLLECTION}/{run_id}"
    body = _build_fs_doc(fields)

    try:
        resp = requests.patch(
            url, json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            timeout=30,
        )
        if resp.status_code in (200, 201):
            print(f"  [Firestore] ✓ {MESH_COLLECTION}/{run_id} → {resp.status_code}")
            return True
        else:
            print(f"  [Firestore] WARNING mesh: {resp.status_code} — {resp.text[:300]}")
            return False
    except Exception as e:
        print(f"  [Firestore] WARNING: mesh write failed — {e}")
        return False

