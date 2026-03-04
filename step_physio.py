"""
step_physio.py
==============
Computes physiotherapy measurements from SMPL joint data.

For each measurement code the module stores:
  options       — the clinical dropdown choices (from the helper sheet)
  auto_result   — value auto-classified from SMPL data (None = manual required)
  auto_source   — which computed value drove the classification
  manual_required — True when classification needs in-person clinical assessment

SMPL joint indices (camera coords Y-down after step5 flip-back):
   0=pelvis  1=l_hip    2=r_hip    3=spine1
   4=l_knee  5=r_knee   7=l_ankle  8=r_ankle
  10=l_foot 11=r_foot  12=neck    15=head
  16=l_shoulder 17=r_shoulder

All distances in mm unless label says _cm.
Positive diff = Left > Right.
"""

import numpy as np

# ── Thresholds ─────────────────────────────────────────────────────────────────
T_SLOPE_EXCESS  = 20.0   # mm height diff → Excessive
T_SLOPE_MOD     = 10.0   # mm height diff → Moderate
T_VALGUS_EXCESS = 20.0   # mm knee deviation → Excessive
T_VALGUS_MOD    = 10.0   # mm knee deviation → Non-Excessive
T_HEAD_OFFSET   = 15.0   # mm head lateral offset
T_ASYM          = 10.0   # mm general asymmetry flag

# ── All dropdown options per code (from helper sheet) ─────────────────────────
PHYSIO_OPTIONS = {
    "1c": {
        "Weight":        ["Normal", "Right +8lbs", "Left +8lbs"],
        "Foot":          ["Excessive Pronation", "Moderate Pronation", "Supination"],
        "Knee Position": ["Neutral", "Varus", "Valgus"],
    },
    "1d": {
        "Lower Body":    ["Inverted", "Neutral", "Everted"],
    },
    "1e": {
        "Hip to Toe (Tip Observation)": [
            "Internal Rotation", "Neutral", "External Rotation"],
        "Hip to Toe (FEM Observation)": [
            "Excessive Internal Rotation", "Moderate Internal Rotation",
            "Neutral", "Moderate External Rotation", "Extreme External Rotation"],
    },
    "2a": {
        "Shoulder Girdle": ["Slope Right", "Neutral", "Slope Left"],
    },
    "2c": {
        "Slope Right ASIS": ["Excessive", "Moderate", "Neutral"],
        "Neutral":           ["Neutral"],
        "Slope Left ASIS":   ["Excessive", "Moderate", "Neutral"],
    },
    "2d": {
        "Right": ["Slope Right", "Moderate", "Spinal Extension", "Neutral"],
        "Left":  ["Slope Right", "Moderate", "Spinal Extension", "Neutral"],
    },
    "2e": {
        "Valgus":  ["Excessive", "Non-Excessive"],
        "Neutral": ["Neutral"],
        "Varus":   ["Excessive", "Non-Excessive"],
    },
    "2f": {
        "Right":   ["Posterior", "Anterior"],
        "Neutral": ["Neutral"],
        "Left":    ["Posterior", "Anterior"],
    },
    "2g": {
        "Right Side Prox TIB": ["Plus", "Neutral", "Minus"],
        "Left Side Prox TIB":  ["Plus", "Neutral", "Minus"],
    },
    "2h": {
        "Right": ["Excessive Heel Raise", "Moderate Heel Raise", "Neutral"],
        "Left":  ["Excessive Heel Raise", "Moderate Heel Raise", "Neutral"],
    },
    "3a": {
        "Right": ["Slope Right", "Neutral", "Slope Left"],
        "Left":  ["Slope Right", "Neutral", "Slope Left"],
    },
    "3b": {
        "Right": ["Forward Head (Ant)", "Neutral", "Head Back (Post)"],
        "Left":  ["Forward Head (Ant)", "Neutral", "Head Back (Post)"],
    },
    "3c": {
        "Right": ["Forward Head (Ant)", "Neutral", "Head Back (Post)"],
        "Left":  ["Forward Head (Ant)", "Neutral", "Head Back (Post)"],
    },
    "5a": {
        "Right": ["Excessive External Rotation", "External Rotation",
                  "Neutral", "Internal Rotation", "Excessive Internal Rotation"],
        "Left":  ["Excessive External Rotation", "External Rotation",
                  "Neutral", "Internal Rotation", "Excessive Internal Rotation"],
    },
    "5b": {
        "Slope Right": ["Excessive", "Moderate"],
        "Neutral":     ["Neutral"],
        "Slope Left":  ["Excessive", "Moderate"],
    },
    "5c": {
        "Right": ["IR", "Neutral", "Ext ROT"],
        "Left":  ["IR", "Neutral", "Ext ROT"],
    },
    "5d": {
        "Right": ["Dorsi Flex", "Neutral", "Plantar Flex"],
        "Left":  ["Dorsi Flex", "Neutral", "Plantar Flex"],
    },
    "5e_hip": {
        "Right": ["Ant", "Neutral", "Post"],
        "Left":  ["Ant", "Neutral", "Post"],
    },
    "5e_knee": {
        "Right": ["Hyper Ext", "Extension", "Neutral", "Flexion", "Hyper Flex"],
        "Left":  ["Hyper Ext", "Extension", "Neutral", "Flexion", "Hyper Flex"],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _r(v, d=1):
    return round(float(v), d) if v is not None else None

def _diff(l, r):
    return _r(l - r) if (l is not None and r is not None) else None

def _asym(diff_val, threshold=T_ASYM):
    return abs(diff_val) >= threshold if diff_val is not None else False

def _mid(a, b):
    return (a + b) / 2.0

def _slope_label(diff_mm, left_is_positive=True):
    """
    Map a height difference (mm) to Slope Left/Right/Neutral.
    diff_mm = left_val - right_val (Y-down coords: more positive = lower)
    In Y-down: left_mm > right_mm means left is LOWER → slopes toward left.
    """
    if diff_mm is None:
        return "unknown"
    if abs(diff_mm) < T_SLOPE_MOD:
        return "Neutral"
    if diff_mm > 0:
        return f"Slope Left {'Excessive' if abs(diff_mm) >= T_SLOPE_EXCESS else 'Moderate'}"
    else:
        return f"Slope Right {'Excessive' if abs(diff_mm) >= T_SLOPE_EXCESS else 'Moderate'}"

def _valgus_label(deviation_mm):
    """Classify knee valgus deviation."""
    if deviation_mm is None:
        return "unknown"
    if abs(deviation_mm) < T_VALGUS_MOD:
        return "Neutral"
    if deviation_mm > 0:
        return f"Valgus {'Excessive' if deviation_mm >= T_VALGUS_EXCESS else 'Non-Excessive'}"
    else:
        return f"Varus {'Excessive' if abs(deviation_mm) >= T_VALGUS_EXCESS else 'Non-Excessive'}"

def _knee_position_label(deviation_mm):
    """Simple three-way Valgus/Neutral/Varus for 1c."""
    if abs(deviation_mm) < T_VALGUS_MOD:
        return "Neutral"
    return "Valgus" if deviation_mm > 0 else "Varus"

def _valgus_x(j, hip_idx, knee_idx, ankle_idx):
    mid_x = _mid(j[hip_idx][0], j[ankle_idx][0])
    return _r(j[knee_idx][0] - mid_x)

def _manual(options_key, sub_key=None):
    """Return a manual-required entry with options listed."""
    opts = PHYSIO_OPTIONS.get(options_key, {})
    sub  = opts.get(sub_key, list(opts.values())[0] if opts else []) if sub_key else opts
    return {"auto_result": None, "manual_required": True,
            "options": sub if isinstance(sub, list) else opts}


# ── Main computation ─────────────────────────────────────────────────────────
def compute_physio_measurements(fitted_joints, measurements):
    """
    Args:
        fitted_joints : np.ndarray (24, 3) mm, camera coords Y-down
        measurements  : dict {label: {"cm": float, "in": float}} from step6
    Returns:
        dict with groups, flat_table, asymmetry_summary
    """
    j = fitted_joints

    def meas(key):
        return _r(measurements.get(key, {}).get("cm"))

    # Pre-compute key derived values
    sh_y_diff  = _diff(j[16][1], j[17][1])   # shoulder height (Y): left - right
    hip_y_diff = _diff(j[1][1],  j[2][1])    # hip Y: left - right
    ankle_y_diff = _diff(j[7][1], j[8][1])
    knee_y_diff  = _diff(j[4][1], j[5][1])
    foot_y_diff  = _diff(j[10][1], j[11][1])

    vl = _valgus_x(j, 1,  4,  7)   # left  knee valgus deviation mm
    vr = _valgus_x(j, 2,  5,  8)   # right knee valgus deviation mm

    head_x_offset = _r(j[15][0] - _mid(j[16][0], j[17][0]))

    # ── Group 1 ──────────────────────────────────────────────────────────────
    g1 = [
        # 1c – Weight (force plate needed)
        {
            "code": "1c", "sub": "Weight",
            "options":        PHYSIO_OPTIONS["1c"]["Weight"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Force plate required",
        },
        # 1c – Foot (visual/scan)
        {
            "code": "1c", "sub": "Foot",
            "options":        PHYSIO_OPTIONS["1c"]["Foot"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Foot scan or visual required",
        },
        # 1c – Knee Position ← AUTO from valgus deviation (average L+R)
        {
            "code": "1c", "sub": "Knee Position",
            "options":        PHYSIO_OPTIONS["1c"]["Knee Position"],
            "left_knee_deviation_mm":  vl,
            "right_knee_deviation_mm": vr,
            "auto_result": {
                "Right": _knee_position_label(vr),
                "Left":  _knee_position_label(vl),
            },
            "manual_required": False,
            "source":         "Knee X deviation from hip-ankle line",
            "asymmetry":      _asym(_diff(vl, vr) or 0),
        },

        # 1d – Lower Body arch (visual)
        {
            "code": "1d", "sub": "Lower Body",
            "options":        PHYSIO_OPTIONS["1d"]["Lower Body"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Visual/sagittal view required",
        },

        # 1e – Hip to Toe Tip Observation (foot rotation)
        {
            "code": "1e", "sub": "Hip to Toe (Tip Observation)",
            "options":        PHYSIO_OPTIONS["1e"]["Hip to Toe (Tip Observation)"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Foot Z-rotation requires top-down or sagittal camera",
        },
        # 1e – Hip to Toe FEM Observation (femoral rotation)
        {
            "code": "1e", "sub": "Hip to Toe (FEM Observation)",
            "options":        PHYSIO_OPTIONS["1e"]["Hip to Toe (FEM Observation)"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Femoral rotation requires lateral/top-down camera",
        },
    ]

    # ── Group 2 ──────────────────────────────────────────────────────────────
    g2 = [
        # 2a – Shoulder Girdle slope ← AUTO
        {
            "code": "2a", "sub": "Shoulder Girdle",
            "options":        PHYSIO_OPTIONS["2a"]["Shoulder Girdle"],
            "shoulder_width_cm": meas("Shoulder Width"),
            "left_shoulder_y_mm":  _r(j[16][1]),
            "right_shoulder_y_mm": _r(j[17][1]),
            "diff_mm":        sh_y_diff,
            "auto_result":    _slope_label(sh_y_diff),
            "manual_required": False,
            "source":         "Shoulder Y-height difference",
            "asymmetry":      _asym(sh_y_diff),
        },

        # 2c – ASIS slope ← AUTO
        {
            "code": "2c", "sub": "ASIS Slope",
            "options":        {k: v for k, v in PHYSIO_OPTIONS["2c"].items()},
            "hip_width_cm":   meas("Hip Width"),
            "left_hip_y_mm":  _r(j[1][1]),
            "right_hip_y_mm": _r(j[2][1]),
            "diff_mm":        hip_y_diff,
            "auto_result":    _slope_label(hip_y_diff),
            "manual_required": False,
            "source":         "Hip/ASIS Y-height difference",
            "asymmetry":      _asym(hip_y_diff),
        },

        # 2d – Spinal tilt (requires sagittal)
        {
            "code": "2d", "sub": "Right",
            "options":        PHYSIO_OPTIONS["2d"]["Right"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Sagittal spine view required",
        },
        {
            "code": "2d", "sub": "Left",
            "options":        PHYSIO_OPTIONS["2d"]["Left"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Sagittal spine view required",
        },

        # 2e – Valgus/Varus ← AUTO
        {
            "code": "2e", "sub": "Valgus/Varus",
            "options":        PHYSIO_OPTIONS["2e"],
            "left_knee_deviation_mm":  vl,
            "right_knee_deviation_mm": vr,
            "auto_result": {
                "Right": _valgus_label(vr),
                "Left":  _valgus_label(vl),
            },
            "manual_required": False,
            "source":         "Knee X deviation from hip-ankle line",
            "asymmetry":      _asym(_diff(vl, vr) or 0),
        },

        # 2f – Pelvic tilt Posterior/Anterior (sagittal)
        {
            "code": "2f", "sub": "Right/Left Pelvic Tilt",
            "options":        PHYSIO_OPTIONS["2f"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Anterior/posterior pelvic tilt requires lateral view",
        },

        # 2g – Proximal Tibia ← AUTO (tibia angle from knee to ankle X)
        {
            "code": "2g", "sub": "Prox TIB",
            "options":        PHYSIO_OPTIONS["2g"],
            "left_knee_x_mm":   _r(j[4][0]),
            "right_knee_x_mm":  _r(j[5][0]),
            "left_ankle_x_mm":  _r(j[7][0]),
            "right_ankle_x_mm": _r(j[8][0]),
            "auto_result": {
                "Right Side Prox TIB": (
                    "Plus"  if (j[5][0] - j[8][0]) >  T_SLOPE_MOD else
                    "Minus" if (j[5][0] - j[8][0]) < -T_SLOPE_MOD else "Neutral"
                ),
                "Left Side Prox TIB": (
                    "Plus"  if (j[4][0] - j[7][0]) >  T_SLOPE_MOD else
                    "Minus" if (j[4][0] - j[7][0]) < -T_SLOPE_MOD else "Neutral"
                ),
            },
            "manual_required": False,
            "source":         "Knee vs ankle X-axis lateral offset",
        },

        # 2h – Heel raise (gait/supine needed)
        {
            "code": "2h", "sub": "Heel Raise",
            "options":        PHYSIO_OPTIONS["2h"],
            "left_foot_y_mm":  _r(j[10][1]),
            "right_foot_y_mm": _r(j[11][1]),
            "foot_y_diff_mm":  foot_y_diff,
            "auto_result": {
                "Right": (
                    "Excessive Heel Raise" if (foot_y_diff or 0) < -T_SLOPE_EXCESS else
                    "Moderate Heel Raise"  if (foot_y_diff or 0) < -T_SLOPE_MOD else "Neutral"
                ),
                "Left": (
                    "Excessive Heel Raise" if (foot_y_diff or 0) > T_SLOPE_EXCESS else
                    "Moderate Heel Raise"  if (foot_y_diff or 0) > T_SLOPE_MOD else "Neutral"
                ),
            },
            "manual_required": False,
            "source":         "Foot Y-height difference left vs right",
            "asymmetry":      _asym(foot_y_diff),
        },
    ]

    # ── Group 3 ──────────────────────────────────────────────────────────────
    g3 = [
        # 3a – Shoulder slope (same derivation as 2a) ← AUTO
        {
            "code": "3a", "sub": "Shoulder Slope",
            "options":        PHYSIO_OPTIONS["3a"],
            "left_shoulder_y_mm":  _r(j[16][1]),
            "right_shoulder_y_mm": _r(j[17][1]),
            "diff_mm":        sh_y_diff,
            "auto_result": {
                "Right": _slope_label(-sh_y_diff if sh_y_diff else 0),
                "Left":  _slope_label(sh_y_diff),
            },
            "manual_required": False,
            "source":         "Shoulder Y-height difference",
            "asymmetry":      _asym(sh_y_diff),
        },

        # 3b – Head position sagittal ← partial AUTO (lateral offset only)
        {
            "code": "3b", "sub": "Head Position (Lateral)",
            "options":        PHYSIO_OPTIONS["3b"],
            "head_x_mm":            _r(j[15][0]),
            "shoulder_midpoint_x_mm": _r(_mid(j[16][0], j[17][0])),
            "head_lateral_offset_mm": head_x_offset,
            "auto_result": {
                "Right": (
                    "Slope Left" if (head_x_offset or 0) >  T_HEAD_OFFSET else
                    "Slope Right" if (head_x_offset or 0) < -T_HEAD_OFFSET else "Neutral"
                ),
                "Left": (
                    "Slope Right" if (head_x_offset or 0) < -T_HEAD_OFFSET else
                    "Slope Left"  if (head_x_offset or 0) >  T_HEAD_OFFSET else "Neutral"
                ),
            },
            "manual_required": False,
            "source":         "Head X-offset from shoulder midpoint",
            "note":           "Forward/backward head tilt requires lateral/sagittal camera",
            "asymmetry":      _asym(head_x_offset),
        },

        # 3c – Head forward/back tilt (sagittal)
        {
            "code": "3c", "sub": "Head Forward/Back",
            "options":        PHYSIO_OPTIONS["3c"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Sagittal/lateral view required for anterior-posterior head tilt",
        },
    ]

    # ── Group 4 — manual ─────────────────────────────────────────────────────
    g4 = [{
        "code": "4", "sub": "Manual Assessment",
        "options":        {},
        "auto_result":    None,
        "manual_required": True,
        "source":         "In-person clinical evaluation required",
    }]

    # ── Group 5 — Supine (derived from standing 3D) ───────────────────────────
    note_supine = "Derived from standing 3D. True supine requires lying-down images."

    g5 = [
        # 5a – Foot rotation (supine)
        {
            "code": "5a", "sub": "Foot Rotation",
            "options":        PHYSIO_OPTIONS["5a"],
            "left_foot_x_mm":  _r(j[10][0]),
            "right_foot_x_mm": _r(j[11][0]),
            "auto_result":     None,
            "manual_required": True,
            "source":         "Foot Z-rotation not measurable from frontal camera alone",
            "note":           note_supine,
        },

        # 5b – Hip tilt supine ← AUTO (same as 2c)
        {
            "code": "5b", "sub": "Hip Tilt (Supine)",
            "options":        PHYSIO_OPTIONS["5b"],
            "left_hip_y_mm":  _r(j[1][1]),
            "right_hip_y_mm": _r(j[2][1]),
            "diff_mm":        hip_y_diff,
            "auto_result":    _slope_label(hip_y_diff),
            "manual_required": False,
            "source":         "Hip Y-height difference (same as 2c ASIS)",
            "note":           note_supine,
            "asymmetry":      _asym(hip_y_diff),
        },

        # 5c – Shoulder rotation (requires axial view)
        {
            "code": "5c", "sub": "Shoulder Rotation",
            "options":        PHYSIO_OPTIONS["5c"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Internal/external rotation requires axial or top-down view",
            "note":           note_supine,
        },

        # 5d – Ankle dorsi/plantar flex (sagittal)
        {
            "code": "5d", "sub": "Ankle Flex",
            "options":        PHYSIO_OPTIONS["5d"],
            "left_ankle_y_mm":  _r(j[7][1]),
            "right_ankle_y_mm": _r(j[8][1]),
            "diff_mm":          ankle_y_diff,
            "auto_result":    None,
            "manual_required": True,
            "source":         "Dorsi/plantar flexion requires lateral/sagittal view",
            "note":           note_supine,
        },

        # 5e – Hip Ant/Post (supine)
        {
            "code": "5e_hip", "sub": "Hip (Supine)",
            "options":        PHYSIO_OPTIONS["5e_hip"],
            "auto_result":    None,
            "manual_required": True,
            "source":         "Hip Ant/Post requires sagittal view",
            "note":           note_supine,
        },

        # 5e – Knee flex/ext (supine) ← partial AUTO from knee Y diff
        {
            "code": "5e_knee", "sub": "Knee (Supine)",
            "options":        PHYSIO_OPTIONS["5e_knee"],
            "left_knee_y_mm":  _r(j[4][1]),
            "right_knee_y_mm": _r(j[5][1]),
            "diff_mm":         knee_y_diff,
            "auto_result": {
                "Right": "Neutral",  # standing = approximately neutral
                "Left":  "Neutral",
            },
            "manual_required": False,
            "source":         "Standing pose assumed neutral; supine flex requires lying image",
            "note":           note_supine,
            "asymmetry":      _asym(knee_y_diff),
        },
    ]

    # ── Flat table ────────────────────────────────────────────────────────────
    flat_rows = []
    for group_num, rows in [(1, g1), (2, g2), (3, g3), (4, g4), (5, g5)]:
        for row in rows:
            flat_rows.append(_norm({
                "group": group_num,
                "group_name": {1:"Foot/Ankle/Knee/Hip", 2:"Postural Alignment",
                               3:"Head/Cervical", 4:"Manual Required",
                               5:"Supine (Derived)"}[group_num],
                **row,
            }))

    # ── Asymmetry summary ─────────────────────────────────────────────────────
    flagged = [
        {"code": r["code"], "sub": r.get("sub"), "diff_mm": r.get("diff_mm")}
        for r in flat_rows if r.get("asymmetry") is True
    ]

    return {
        "groups": {
            "group_1": {"name": "Foot/Ankle/Knee/Hip Alignment",               "rows": g1},
            "group_2": {"name": "Postural Alignment (Standing)",               "rows": g2},
            "group_3": {"name": "Head/Cervical Spine",                         "rows": g3},
            "group_4": {"name": "Manual Assessment Required",                  "rows": g4},
            "group_5": {"name": "Supine Assessment (Derived from Standing)",   "rows": g5},
        },
        "flat_table": flat_rows,
        "asymmetry_summary": {
            "total_flagged": len(flagged),
            "flagged_items": flagged,
        },
    }


def _norm(row: dict) -> dict:
    """
    Normalize a row to always have standard columns for CSV export.
    Picks the first numeric left/right pair found and maps it to
    left_value / right_value / diff_value / value_unit.
    Expands auto_result dict → auto_result_left, auto_result_right strings.
    """
    r = dict(row)

    # ── Standard value columns ─────────────────────────────────────────────────
    # Priority order: try mm fields first, then cm fields
    left_val = right_val = diff_val = None
    unit = ""

    for (lk, rk, dk, u) in [
        ("left_mm",              "right_mm",              "diff_mm",      "mm"),
        ("left_knee_y_mm",       "right_knee_y_mm",       "diff_mm",      "mm"),
        ("left_shoulder_y_mm",   "right_shoulder_y_mm",   "diff_mm",      "mm"),
        ("left_hip_y_mm",        "right_hip_y_mm",        "diff_mm",      "mm"),
        ("left_ankle_y_mm",      "right_ankle_y_mm",      "diff_mm",      "mm"),
        ("left_foot_y_mm",       "right_foot_y_mm",       "diff_mm",      "mm"),
        ("left_knee_deviation_mm","right_knee_deviation_mm","diff_mm",    "mm"),
        ("left_knee_x_mm",       "right_knee_x_mm",       None,           "mm"),
        ("left_foot_x_mm",       "right_foot_x_mm",       None,           "mm"),
        ("head_lateral_offset_mm", None,                  None,           "mm"),
        ("left_cm",              "right_cm",              "diff_cm",       "cm"),
    ]:
        lv = r.get(lk)
        rv = r.get(rk) if rk else None
        dv = r.get(dk) if dk else (_diff(lv, rv) if (lv is not None and rv is not None) else None)
        if lv is not None or rv is not None or dv is not None:
            left_val  = lv
            right_val = rv
            diff_val  = dv
            unit      = u
            break

    r["left_value"]  = left_val
    r["right_value"] = right_val
    r["diff_value"]  = diff_val
    r["value_unit"]  = unit

    # ── Expand auto_result ─────────────────────────────────────────────────────
    ar = r.get("auto_result")
    if isinstance(ar, dict):
        r["auto_result_right"] = ar.get("Right") or ar.get("Right Side Prox TIB", "")
        r["auto_result_left"]  = ar.get("Left")  or ar.get("Left Side Prox TIB",  "")
    elif isinstance(ar, str):
        r["auto_result_right"] = ar
        r["auto_result_left"]  = ar
    else:
        r["auto_result_right"] = ""
        r["auto_result_left"]  = ""

    # ── Options as flat pipe-separated string ──────────────────────────────────
    opts = r.get("options", {})
    if isinstance(opts, list):
        r["options_str"] = " | ".join(str(o) for o in opts)
    elif isinstance(opts, dict):
        parts = []
        for side, choices in opts.items():
            if isinstance(choices, list):
                parts.append(f"{side}: {' | '.join(str(c) for c in choices)}")
            else:
                parts.append(f"{side}: {choices}")
        r["options_str"] = "  ||  ".join(parts)
    else:
        r["options_str"] = str(opts) if opts else ""

    return r

