"""
step_physio.py
==============
Computes physiotherapy measurements from SMPL joint data.

Maps to the 5-group physio assessment framework:
  Group 1 — Foot/Ankle/Knee/Hip axis alignment
  Group 2 — Postural alignment (Shoulders/ASIS/Ankle/Knee/Foot)
  Group 3 — Head/Cervical spine alignment
  Group 4 — (reserved / manual assessment)
  Group 5 — Supine assessment (derived from standing 3D data)

SMPL joint indices used:
   0=pelvis  1=l_hip    2=r_hip    3=spine1
   4=l_knee  5=r_knee   6=spine2   7=l_ankle
   8=r_ankle 9=spine3  10=l_foot  11=r_foot
  12=neck   13=l_collar 14=r_collar 15=head
  16=l_shoulder 17=r_shoulder 18=l_elbow 19=r_elbow
  20=l_wrist    21=r_wrist    22=l_hand  23=r_hand

Coordinate system (after step5 flip-back = camera coords):
  X  — lateral: positive → camera's right
  Y  — vertical: negative = above camera, positive = below (toward feet)
  Z  — depth from camera pair (mm)

All distance values are in mm unless label says _cm.
Positive diff = Left > Right (for X: left is further right on image).
"""

import numpy as np


# ── Asymmetry flag threshold ───────────────────────────────────────────────────
ASYM_THRESH_MM  = 10.0   # ≥ 10 mm → flagged as asymmetric
ASYM_THRESH_DEG = 5.0    # ≥ 5 °  → flagged for angle differences


def _r(v, decimals=1):
    """Round a float or None."""
    return round(float(v), decimals) if v is not None else None


def _diff(left, right):
    return _r(left - right)


def _asym(diff_val, threshold=ASYM_THRESH_MM):
    return abs(diff_val) >= threshold if diff_val is not None else False


def _midpoint(a, b):
    return (a + b) / 2.0


def compute_physio_measurements(fitted_joints, measurements):
    """
    Args:
        fitted_joints : np.ndarray (24, 3) in mm, camera coord (Y-down)
        measurements  : dict from step6  { label: {"cm": float, "in": float} }

    Returns:
        dict: structured physio measurement groups ready for Firestore
    """
    j = fitted_joints   # shorthand

    def meas_cm(key):
        return _r(measurements.get(key, {}).get("cm"))

    # ── Helper: valgus/varus deviation ────────────────────────────────────────
    # How far the knee deviates (X) from the straight line hip→ankle
    def valgus_x(hip_idx, knee_idx, ankle_idx):
        mid_x = _midpoint(j[hip_idx][0], j[ankle_idx][0])
        return _r(j[knee_idx][0] - mid_x)

    # ─────────────────────────────────────────────────────────────────────────
    # Group 1 — Foot / Ankle / Knee / Hip Axis
    # ─────────────────────────────────────────────────────────────────────────
    g1 = []

    # 1c  Lower X Axis — Knee lateral position (L vs R)
    kx_diff = _diff(j[4][0], j[5][0])
    g1.append({
        "code": "1c", "axis": "Lower X Axis", "landmark": "Knee",
        "triangle_links": [4, 5],
        "left_mm":  _r(j[4][0]),  "right_mm": _r(j[5][0]),
        "diff_mm":  kx_diff,       "asymmetry": _asym(kx_diff),
        "note": "X-axis position; diff > 0 = left knee further right"
    })

    # 1c  Foot X
    fx_diff = _diff(j[10][0], j[11][0])
    g1.append({
        "code": "1c", "axis": "Lower X Axis", "landmark": "Foot",
        "triangle_links": [4, 5],
        "left_mm":  _r(j[10][0]), "right_mm": _r(j[11][0]),
        "diff_mm":  fx_diff,       "asymmetry": _asym(fx_diff),
    })

    # 1d  Lower Y Axis — Knee height asymmetry (Y-down: lower Y = higher)
    ky_diff = _diff(j[4][1], j[5][1])
    g1.append({
        "code": "1d", "axis": "Lower Y Axis", "landmark": "Knee",
        "triangle_links": [4],
        "left_mm":  _r(j[4][1]),  "right_mm": _r(j[5][1]),
        "diff_mm":  ky_diff,       "asymmetry": _asym(ky_diff),
        "note": "Knee height (Y); more negative = higher knee"
    })

    # 1e  Lower Y Axis — Hip to Toe (full leg length)
    ll_diff = _diff(meas_cm("Left Full Leg"), meas_cm("Right Full Leg"))
    g1.append({
        "code": "1e", "axis": "Lower Y Axis", "landmark": "Hip to Toe (TPT2)",
        "triangle_links": [3, 4, 5],
        "left_cm":  meas_cm("Left Full Leg"),
        "right_cm": meas_cm("Right Full Leg"),
        "diff_cm":  ll_diff,
        "asymmetry": _asym(ll_diff * 10 if ll_diff else 0),  # convert to mm
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Group 2 — Postural Alignment
    # ─────────────────────────────────────────────────────────────────────────
    g2 = []

    # 2a  Upper X Axis — Shoulders
    sx_diff = _diff(j[16][0], j[17][0])
    g2.append({
        "code": "2a", "axis": "Upper X Axis", "landmark": "Shoulders",
        "triangle_links": [2],
        "shoulder_width_cm": meas_cm("Shoulder Width"),
        "left_mm":  _r(j[16][0]), "right_mm": _r(j[17][0]),
        "diff_mm":  sx_diff,       "asymmetry": _asym(sx_diff),
        "note": "Lateral position of each shoulder"
    })

    # 2b  Upper Y Axis — Shoulder height
    sy_diff = _diff(j[16][1], j[17][1])
    g2.append({
        "code": "2b", "axis": "Upper Y Axis", "landmark": "Shoulders",
        "triangle_links": [2, 3],
        "left_mm":  _r(j[16][1]), "right_mm": _r(j[17][1]),
        "diff_mm":  sy_diff,       "asymmetry": _asym(sy_diff),
        "note": "Height (Y) of each shoulder; diff > 0 = left shoulder lower"
    })

    # 2c  Lower X Axis — ASIS/Hip
    hx_diff = _diff(j[1][0], j[2][0])
    g2.append({
        "code": "2c", "axis": "Lower X Axis", "landmark": "ASIS (Hip)",
        "triangle_links": [3],
        "hip_width_cm": meas_cm("Hip Width"),
        "left_mm":  _r(j[1][0]),  "right_mm": _r(j[2][0]),
        "diff_mm":  hx_diff,       "asymmetry": _asym(hx_diff),
    })

    # 2d  Lower Y Axis — Ankle height
    ay_diff = _diff(j[7][1], j[8][1])
    g2.append({
        "code": "2d", "axis": "Lower Y Axis", "landmark": "Ankle",
        "triangle_links": [5],
        "left_mm":  _r(j[7][1]),  "right_mm": _r(j[8][1]),
        "diff_mm":  ay_diff,       "asymmetry": _asym(ay_diff),
    })

    # 2e  Lower X Axis — Valgus/Varus (knee deviation from hip-ankle line)
    vl = valgus_x(1, 4, 7)   # left
    vr = valgus_x(2, 5, 8)   # right
    vdiff = _diff(vl, vr)
    g2.append({
        "code": "2e", "axis": "Lower X Axis", "landmark": "Valgus/Varus (Knee)",
        "triangle_links": [4],
        "left_knee_deviation_mm":  vl,
        "right_knee_deviation_mm": vr,
        "diff_mm":  vdiff,  "asymmetry": _asym(vdiff),
        "note": "+ve = valgus (knee inward), -ve = varus (knee outward)"
    })

    # 2f  Lower Y Axis — ASIS/Hip height
    hy_diff = _diff(j[1][1], j[2][1])
    g2.append({
        "code": "2f", "axis": "Lower Y Axis", "landmark": "ASIS (Hip)",
        "triangle_links": [3, 4],
        "left_mm":  _r(j[1][1]),  "right_mm": _r(j[2][1]),
        "diff_mm":  hy_diff,       "asymmetry": _asym(hy_diff),
    })

    # 2g  Lower Y Axis — Knee height
    ky2_diff = _diff(j[4][1], j[5][1])
    g2.append({
        "code": "2g", "axis": "Lower Y Axis", "landmark": "Knee",
        "triangle_links": [4, 5],
        "left_mm":  _r(j[4][1]),  "right_mm": _r(j[5][1]),
        "diff_mm":  ky2_diff,      "asymmetry": _asym(ky2_diff),
    })

    # 2h  Lower Y Axis — Foot height
    fy_diff = _diff(j[10][1], j[11][1])
    g2.append({
        "code": "2h", "axis": "Lower Y Axis", "landmark": "Foot",
        "triangle_links": [5],
        "left_mm":  _r(j[10][1]), "right_mm": _r(j[11][1]),
        "diff_mm":  fy_diff,       "asymmetry": _asym(fy_diff),
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Group 3 — Head / Cervical Spine
    # ─────────────────────────────────────────────────────────────────────────
    g3 = []

    shoulder_mid_x = _r(_midpoint(j[16][0], j[17][0]))
    head_offset    = _r(j[15][0] - _midpoint(j[16][0], j[17][0]))

    # 3a  Upper X Axis — Head offset from shoulder midpoint
    g3.append({
        "code": "3a", "axis": "Upper X Axis", "landmark": "Head/Shoulders",
        "triangle_links": [1, 2],
        "head_x_mm":            _r(j[15][0]),
        "shoulder_midpoint_x_mm": shoulder_mid_x,
        "head_lateral_offset_mm": head_offset,
        "asymmetry": _asym(head_offset),
        "note": "+ve = head shifted toward left (camera) side"
    })

    # 3b  Upper Y Axis — Head/shoulder height + neck
    g3.append({
        "code": "3b", "axis": "Upper Y Axis", "landmark": "Head/Shoulders",
        "triangle_links": [1, 2],
        "neck_to_head_cm":         meas_cm("Neck to Head"),
        "head_y_mm":               _r(j[15][1]),
        "neck_y_mm":               _r(j[12][1]),
        "left_shoulder_y_mm":      _r(j[16][1]),
        "right_shoulder_y_mm":     _r(j[17][1]),
        "shoulder_height_diff_mm": _diff(j[16][1], j[17][1]),
        "note": "Y-axis; more negative = higher"
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Group 4 — (Manual clinical assessment, not computable from images)
    # ─────────────────────────────────────────────────────────────────────────
    g4 = [{
        "code": "4", "axis": "N/A",
        "landmark": "Manual Assessment Required",
        "note": "Group 4 measurements require in-person clinical evaluation"
    }]

    # ─────────────────────────────────────────────────────────────────────────
    # Group 5 — Supine Assessment (derived from standing 3D data)
    # ─────────────────────────────────────────────────────────────────────────
    g5 = []
    note_supine = ("Derived from standing 3D pose. "
                   "True supine values require images taken lying down.")

    # 5a  Supine X Axis — Foot
    g5.append({
        "code": "5a", "axis": "Supine X Axis", "landmark": "Foot",
        "triangle_links": [4, 5],
        "left_mm":  _r(j[10][0]), "right_mm": _r(j[11][0]),
        "diff_mm":  _diff(j[10][0], j[11][0]),
        "note": note_supine
    })

    # 5b  Supine X Axis — Hip
    g5.append({
        "code": "5b", "axis": "Supine X Axis", "landmark": "Hip",
        "triangle_links": [3],
        "left_mm":  _r(j[1][0]),  "right_mm": _r(j[2][0]),
        "diff_mm":  _diff(j[1][0], j[2][0]),
        "note": note_supine
    })

    # 5c  Supine X Axis — Knee
    g5.append({
        "code": "5c", "axis": "Supine X Axis", "landmark": "Knee",
        "triangle_links": [4],
        "left_mm":  _r(j[4][0]),  "right_mm": _r(j[5][0]),
        "diff_mm":  _diff(j[4][0], j[5][0]),
        "note": note_supine
    })

    # 5d  Lower Y Axis — Foot
    g5.append({
        "code": "5d", "axis": "Lower Y Axis", "landmark": "Foot",
        "triangle_links": [5],
        "left_mm":  _r(j[10][1]), "right_mm": _r(j[11][1]),
        "diff_mm":  _diff(j[10][1], j[11][1]),
        "note": note_supine
    })

    # 5e  Lower Y Axis — Hip
    g5.append({
        "code": "5e", "axis": "Lower Y Axis", "landmark": "Hip",
        "triangle_links": [3],
        "left_mm":  _r(j[1][1]),  "right_mm": _r(j[2][1]),
        "diff_mm":  _diff(j[1][1], j[2][1]),
        "note": note_supine
    })

    # 5f  Lower Y Axis — Knee
    g5.append({
        "code": "5f", "axis": "Lower Y Axis", "landmark": "Knee",
        "triangle_links": [4],
        "left_mm":  _r(j[4][1]),  "right_mm": _r(j[5][1]),
        "diff_mm":  _diff(j[4][1], j[5][1]),
        "note": note_supine
    })

    # ─────────────────────────────────────────────────────────────────────────
    # Assemble flat table (one row per measurement code)
    # ─────────────────────────────────────────────────────────────────────────
    flat_rows = []
    for group_num, rows in [(1, g1), (2, g2), (3, g3), (4, g4), (5, g5)]:
        for row in rows:
            flat_rows.append({"group": group_num, **row})

    return {
        "groups": {
            "group_1": {"name": "Foot/Ankle/Knee/Hip Alignment",   "rows": g1},
            "group_2": {"name": "Postural Alignment",               "rows": g2},
            "group_3": {"name": "Head/Cervical Spine",              "rows": g3},
            "group_4": {"name": "Manual Assessment Required",       "rows": g4},
            "group_5": {"name": "Supine Assessment (Derived)",      "rows": g5},
        },
        "flat_table": flat_rows,
        "asymmetry_summary": _asymmetry_summary(flat_rows),
    }


def _asymmetry_summary(flat_rows):
    """Return a condensed dict of all flagged asymmetries."""
    flagged = [
        {"code": r["code"], "landmark": r.get("landmark"), "diff_mm": r.get("diff_mm")}
        for r in flat_rows
        if r.get("asymmetry") is True
    ]
    return {
        "total_flagged": len(flagged),
        "flagged_items": flagged,
    }
