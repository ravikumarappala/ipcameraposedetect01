"""
step_physio_render.py
=====================
Renders a front-view SMPL mesh annotated with the physiotherapy
landmark numbering system (1R/1L – 27R/27L) from the Bronson
physio assessment framework, plus key measurement lines.

Outputs:
  physio_annotated.png   — mesh image with all landmarks labelled
  physio_landmarks.json  — 2D/3D coordinates of each numbered point

Landmark→SMPL mapping (approximated from diagram):
  Upper zone (above waistline = spine1 Y level):
    21R/21L  → head exterior (project from joint 15)
    20R/20L  → skull base / occiput
    19R/19L  → acromion / outer shoulder
    18R/18L  → neck base (joint 12)
    17R/17L  → inner shoulder (collar joints 13, 14)
    16R/16L  → shoulder joint (joints 16, 17)
    23R/23L  → far shoulder / deltoid outer
    22R/22L  → shoulder level mid
    15R/15L  → axilla / upper chest
    14R/14L  → lower ribcage
    24R/24L  → body side upper
    25R/25L  → body side mid
    13R/13L  → waistline (spine1 Y level)
  Lower zone (below waistline):
    26R/26L  → ASIS / hip brim (joints 1, 2)
    11R/11L  → inguinal / groin
    12R/12L  → navel (spine1 region)
     9R/ 9L  → pelvis (joint 0)
             → (mirrored point)
     8R/ 8L  → inner thigh
     7R/ 7L  → thigh mid
    10R       → body side at hip
     4R/ 4L  → knee (joints 4, 5)
     5R/ 5L  → lateral knee
     6R/ 6L  → popliteal / calf inner
     3R/ 3L  → lower shin
     2R/ 2L  → ankle (joints 7, 8)
     1R/ 1L  → foot / heel (joints 10, 11)
    27R/27L  → hand (joints 22, 23)
    10R/10L  → body side at hip level
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import json

# ── Physio group colours (match the image) ─────────────────────────────────
C_RIGHT  = "#1565C0"   # blue — right side labels
C_LEFT   = "#C62828"   # red  — left side labels
C_AXIS   = "#1565C0"   # blue — vertical centre line
C_WAIST  = "#1565C0"   # blue — waistline horizontal
C_MESH   = "#B0BEC5"   # light grey mesh
C_JOINT  = "#263238"   # dark joint dots

# ── Landmark definitions ────────────────────────────────────────────────────
# Each entry: (number, side, smpl_joint_idx, x_offset_frac, y_offset_frac)
# x_offset_frac: fraction of body half-width added outward (+ = away from centre)
# y_offset_frac: fraction of body height added upward from the joint
LANDMARK_DEF = [
    # ── Feet / ankle / shin ───────────────────────────────────────────────
    ( 1, "R", 11,  0.05,  0.00),   # r_foot
    ( 1, "L", 10, -0.05,  0.00),   # l_foot
    ( 2, "R",  8,  0.10,  0.05),   # r_ankle outer
    ( 2, "L",  7, -0.10,  0.05),   # l_ankle outer
    ( 3, "R",  5,  0.20, -0.12),   # lower r_shin (between knee & ankle)
    ( 3, "L",  4, -0.20, -0.12),   # lower l_shin
    # ── Knee ──────────────────────────────────────────────────────────────
    ( 4, "R",  5,  0.00,  0.00),   # r_knee
    ( 4, "L",  4,  0.00,  0.00),   # l_knee
    ( 5, "R",  5,  0.35,  0.00),   # lateral r_knee
    ( 5, "L",  4, -0.35,  0.00),   # lateral l_knee
    ( 6, "R",  5,  0.15, -0.08),   # popliteal / calf r side (inner)
    ( 6, "L",  4, -0.15, -0.08),   # popliteal / calf l side
    # ── Thigh / hip ───────────────────────────────────────────────────────
    ( 7, "R",  2,  0.10, -0.40),   # r_thigh mid (interpolated hip→knee)
    ( 7, "L",  1, -0.10, -0.40),   # l_thigh mid
    ( 8, "R",  2,  0.00, -0.20),   # inner r_thigh / groin
    ( 8, "L",  1,  0.00, -0.20),   # inner l_thigh
    ( 9, "R",  0,  0.15,  0.00),   # pelvis r
    ( 9, "L",  0, -0.15,  0.00),   # pelvis l
    (10, "R",  2,  0.60,  0.10),   # body side at hip level r
    (10, "L",  1, -0.60,  0.10),   # body side at hip level l
    (11, "R",  2,  0.30, -0.05),   # inguinal / ASIS lower r
    (11, "L",  1, -0.30, -0.05),   # inguinal / ASIS lower l
    # ── Abdomen / waist ───────────────────────────────────────────────────
    (12, "R",  3,  0.20, -0.15),   # navel r
    (12, "L",  3, -0.20, -0.15),   # navel l
    (13, "R",  3,  0.25,  0.00),   # waistline r (spine1 level)
    (13, "L",  3, -0.25,  0.00),   # waistline l
    # ── Mid/upper torso ───────────────────────────────────────────────────
    (14, "R",  6,  0.25, -0.05),   # lower ribcage r (spine2)
    (14, "L",  6, -0.25, -0.05),   # lower ribcage l
    (15, "R",  6,  0.35,  0.15),   # upper chest/axilla r
    (15, "L",  6, -0.35,  0.15),   # upper chest/axilla l
    (24, "R",  6,  0.70,  0.00),   # body side upper r
    (24, "L",  6, -0.70,  0.00),   # body side upper l
    (25, "R",  3,  0.65,  0.05),   # body side mid r
    (25, "L",  3, -0.65,  0.05),   # body side mid l
    (26, "R",  1,  0.75,  0.10),   # body side lower r (ASIS brim)
    (26, "L",  2, -0.75,  0.10),   # body side lower l
    # ── Shoulder zone ─────────────────────────────────────────────────────
    (16, "R", 17,  0.00,  0.00),   # r_shoulder
    (16, "L", 16,  0.00,  0.00),   # l_shoulder
    (17, "R", 14,  0.00,  0.00),   # r_collar
    (17, "L", 13,  0.00,  0.00),   # l_collar
    (18, "R", 12,  0.15,  0.00),   # neck base r
    (18, "L", 12, -0.15,  0.00),   # neck base l
    (19, "R", 17,  0.45,  0.05),   # acromion r
    (19, "L", 16, -0.45,  0.05),   # acromion l
    (22, "R", 17,  0.25,  0.00),   # shoulder mid r
    (22, "L", 16, -0.25,  0.00),   # shoulder mid l
    (23, "R", 17,  0.65,  0.02),   # far shoulder r
    (23, "L", 16, -0.65,  0.02),   # far shoulder l
    # ── Head / neck ───────────────────────────────────────────────────────
    (20, "R", 15,  0.20, -0.05),   # skull base r
    (20, "L", 15, -0.20, -0.05),   # skull base l
    (21, "R", 15,  0.30,  0.15),   # head top r
    (21, "L", 15, -0.30,  0.15),   # head top l
    # ── Hands ─────────────────────────────────────────────────────────────
    (27, "R", 23,  0.00,  0.00),   # r_hand
    (27, "L", 22,  0.00,  0.00),   # l_hand
]


def _project_joints(fitted_joints):
    """
    Project 3D joints to 2D front view (X = lateral, Z_up = -Y_camera).
    Returns (24, 2) array: column 0 = X, column 1 = height (up = positive).
    """
    j = fitted_joints.copy()
    # Camera coords: Y-down → flip to height (up = positive)
    x2d = j[:, 0]
    y2d = -j[:, 1]  # Y-up
    return np.column_stack([x2d, y2d])


def _body_scale(j2d):
    """Return (half_width, body_height) for offset scaling."""
    x_range = j2d[:, 0].max() - j2d[:, 0].min()
    y_range = j2d[:, 1].max() - j2d[:, 1].min()
    half_w  = x_range / 2.0
    return half_w, y_range


def render_physio_landmarks(fitted_joints, fitted_verts, faces,
                            measurements, physio_data,
                            out_path="physio_annotated.png"):
    """
    Render front-view SMPL mesh with physio landmark numbers overlaid.

    Parameters
    ----------
    fitted_joints : (24, 3) mm  camera-coords Y-down
    fitted_verts  : (6890, 3) mm
    faces         : (N, 3) int
    measurements  : dict from step6
    physio_data   : dict from step_physio
    out_path      : save path for PNG

    Returns
    -------
    str  path to saved image
    dict landmark metadata (pixel coords + 3D coords)
    """
    j2d   = _project_joints(fitted_joints)
    half_w, body_h = _body_scale(j2d)

    # Project vertices
    vx = fitted_verts[:, 0]
    vy = -fitted_verts[:, 1]   # flip Y for height-up

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 16), dpi=120)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    # ── Draw mesh silhouette (scatter of vertices, density = body) ────────
    ax.scatter(vx, vy, s=0.3, c=C_MESH, alpha=0.35, linewidths=0, zorder=1)

    # ── Draw skeleton connections ──────────────────────────────────────────
    CONNECTIONS = [
        (15, 12), (12, 9), (9, 6), (6, 3), (3, 0),   # spine
        (0, 1), (0, 2),                                # hip
        (1, 4), (4, 7), (7, 10),                      # left leg
        (2, 5), (5, 8), (8, 11),                      # right leg
        (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # left arm
        (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),  # right arm
        (12, 15),                                       # neck→head
    ]
    for (a, b) in CONNECTIONS:
        ax.plot([j2d[a, 0], j2d[b, 0]], [j2d[a, 1], j2d[b, 1]],
                color="#607D8B", lw=1.2, zorder=2, alpha=0.6)

    # ── Draw joint dots ────────────────────────────────────────────────────
    ax.scatter(j2d[:, 0], j2d[:, 1], s=18, c=C_JOINT, zorder=3)

    # ── Derived body geometry ──────────────────────────────────────────────
    x_centre = (j2d[:, 0].max() + j2d[:, 0].min()) / 2
    waist_y  = j2d[3, 1]   # spine1 level
    foot_y   = j2d[[10, 11], 1].min()
    head_y   = j2d[15, 1] + 0.1 * body_h

    # ── Vertical centre axis ───────────────────────────────────────────────
    ax.axvline(x=x_centre, color=C_AXIS, lw=1.5, ls="--", alpha=0.7,
               zorder=2, label="Centre axis")

    # ── Waistline horizontal ───────────────────────────────────────────────
    ax.axhline(y=waist_y, color=C_WAIST, lw=1.8, ls="-", alpha=0.7, zorder=2)
    ax.text(j2d[:, 0].min() - half_w * 0.3, waist_y + body_h * 0.01,
            "Waistline", color=C_WAIST, fontsize=7, va="bottom", ha="left")

    # ── Measurement annotation lines ───────────────────────────────────────
    sw_cm  = measurements.get("Shoulder Width", {}).get("cm", 0)
    hw_cm  = measurements.get("Hip Width",      {}).get("cm", 0)
    sh_y   = j2d[16, 1]
    hip_y  = j2d[[1, 2], 1].mean()
    sh_l   = j2d[16, 0];  sh_r = j2d[17, 0]
    hip_l  = j2d[1, 0];   hip_r = j2d[2, 0]

    def _measurement_bar(ax, x0, x1, y, label_cm, color, linestyle="-"):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.3))
        ax.text((x0 + x1) / 2, y + body_h * 0.012, f"{label_cm:.1f}cm",
                ha="center", va="bottom", fontsize=7, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.5))

    _measurement_bar(ax, sh_l,  sh_r,  sh_y  + body_h * 0.04, sw_cm,  "#F44336")
    _measurement_bar(ax, hip_l, hip_r, hip_y + body_h * 0.04, hw_cm,  "#9C27B0")

    # ── Place landmark numbers ─────────────────────────────────────────────
    placed = {}   # key: f"{num}{side}" → (px, py, 3d_mm)

    for (num, side, jidx, xoff_frac, yoff_frac) in LANDMARK_DEF:
        px = j2d[jidx, 0] + xoff_frac * half_w
        py = j2d[jidx, 1] + yoff_frac * body_h
        label = f"{num}{side}"
        color = C_RIGHT if side == "R" else C_LEFT

        ax.annotate(label,
                    xy=(px, py), fontsize=6.5, fontweight="bold",
                    color=color, ha="center", va="center",
                    bbox=dict(boxstyle="square,pad=0.18", fc="white",
                              ec=color, lw=0.8, alpha=0.88),
                    zorder=5)

        placed[label] = {
            "landmark": label,
            "number":   num,
            "side":     side,
            "smpl_joint_idx": jidx,
            "smpl_joint_name": _SMPL_NAMES[jidx],
            "proj_x_mm":  round(float(px), 1),
            "proj_y_mm":  round(float(py), 1),
            "world_x_mm": round(float(fitted_joints[jidx, 0]), 1),
            "world_y_mm": round(float(fitted_joints[jidx, 1]), 1),
            "world_z_mm": round(float(fitted_joints[jidx, 2]), 1),
        }

    # ── Asymmetry flags from physio_data ──────────────────────────────────
    flagged = physio_data.get("asymmetry_summary", {}).get("flagged_items", [])
    if flagged:
        flag_txt = "⚠ Asymmetry flags: " + ", ".join(
            f"{f['code']} {f.get('sub','')} ({f.get('diff_mm',''):.1f}mm)"
            for f in flagged if f.get("diff_mm") is not None
        )
        ax.text(0.02, 0.01, flag_txt, transform=ax.transAxes,
                fontsize=7, color="#E65100", va="bottom",
                bbox=dict(boxstyle="round", fc="#FFF3E0", ec="#E65100", lw=0.8))

    # ── Legend ────────────────────────────────────────────────────────────
    legend_els = [
        mpatches.Patch(color=C_RIGHT, label="Right side (R)"),
        mpatches.Patch(color=C_LEFT,  label="Left side (L)"),
        Line2D([0], [0], color=C_AXIS,  ls="--", lw=1.5, label="Centre axis"),
        Line2D([0], [0], color=C_WAIST, ls="-",  lw=1.8, label="Waistline"),
        Line2D([0], [0], color="#F44336", marker=None, lw=1.5, label="Shoulder width"),
        Line2D([0], [0], color="#9C27B0", marker=None, lw=1.5, label="Hip width"),
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=7, framealpha=0.9)

    # ── Axis formatting ────────────────────────────────────────────────────
    ax.set_aspect("equal")
    margin = half_w * 1.4
    ax.set_xlim(x_centre - margin * 2, x_centre + margin * 2)
    ax.set_ylim(foot_y - body_h * 0.05, head_y + body_h * 0.05)
    ax.axis("off")
    ax.set_title("Physiotherapy Landmark Assessment\n(1R/1L – 27R/27L  |  Front View)",
                 fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Physio Render] Saved: {out_path}")
    return out_path, placed


# ── SMPL joint names lookup (shortened) ────────────────────────────────────
_SMPL_NAMES = [
    "pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2",
    "l_ankle","r_ankle","spine3","l_foot","r_foot","neck",
    "l_collar","r_collar","head","l_shoulder","r_shoulder",
    "l_elbow","r_elbow","l_wrist","r_wrist","l_hand","r_hand"
]


def run(logger, fitted_joints, fitted_verts, faces, measurements, physio_data):
    """
    Called from run_pipeline.py after physio computation.
    Renders image, saves locally, returns (out_path, placed_dict).
    """
    out_dir = os.path.join(logger.run_dir, "physio-render")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "physio_annotated.png")

    img_path, placed = render_physio_landmarks(
        fitted_joints, fitted_verts, faces,
        measurements, physio_data, out_path=out_path
    )
    return img_path, placed
