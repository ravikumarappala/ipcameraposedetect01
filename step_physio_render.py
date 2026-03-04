"""
step_physio_render.py  (v2)
===========================
Produces a front-view SMPL mesh image that closely matches the
Bronson physiotherapy assessment diagram:

  • 6 numbered spine nodes  (red filled squares: 1–6)
  • 6 alternating triangles drawn at actual body silhouette width
      Δ1 inverted  head → shoulders
      Δ2 upright   shoulders → waist
      Δ3 inverted  waist → hip
      Δ4 upright   hip → knee
      Δ5 inverted  knee → ankle
      Δ6 semi      ankle → floor
  • Horizontal reference lines at every zone junction
  • Vertical spine axis + blue waist divider
  • Numbered landmark boxes (1R/1L – 27R/27L) in boxed labels
  • Measurement bars (shoulder width, hip width, leg lengths)
  • Angle labels at key joints (elbow, knee, shoulder)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon as MplPolygon
from matplotlib.lines import Line2D
import json

# ── Colours ──────────────────────────────────────────────────────────────────
RED_EDGE   = "#CC0000"
RED_FILL   = (0.80, 0.00, 0.00, 0.18)   # RGBA – light red fill
RED_SOLID  = "#CC0000"
BLUE_AXIS  = "#1565C0"
BLUE_WAIST = "#1565C0"
SKELETON   = "#607D8B"
MESH_DOT   = "#90A4AE"
SPINE_DOT  = "#212121"
C_RIGHT    = "#1565C0"   # blue labels = RIGHT side
C_LEFT     = "#C62828"   # red labels  = LEFT side
BG         = "#F5F5F5"

# SMPL joint names
JNAMES = [
    "pelvis","l_hip","r_hip","spine1","l_knee","r_knee","spine2",
    "l_ankle","r_ankle","spine3","l_foot","r_foot","neck",
    "l_collar","r_collar","head","l_shoulder","r_shoulder",
    "l_elbow","r_elbow","l_wrist","r_wrist","l_hand","r_hand"
]

# ── Landmark definitions (number, side, smpl_joint_idx, x_off_frac, y_off_frac) ──
# x_off_frac: × body half-width;  y_off_frac: × body height
LANDMARKS = [
    ( 1,"R",11,  0.06, 0.00), ( 1,"L",10, -0.06, 0.00),
    ( 2,"R", 8,  0.12, 0.04), ( 2,"L", 7, -0.12, 0.04),
    ( 3,"R", 5,  0.28,-0.14), ( 3,"L", 4, -0.28,-0.14),
    ( 4,"R", 5,  0.04, 0.00), ( 4,"L", 4, -0.04, 0.00),
    ( 5,"R", 5,  0.42, 0.00), ( 5,"L", 4, -0.42, 0.00),
    ( 6,"R", 5,  0.18,-0.08), ( 6,"L", 4, -0.18,-0.08),
    ( 7,"R", 2,  0.12,-0.40), ( 7,"L", 1, -0.12,-0.40),
    ( 8,"R", 2,  0.02,-0.20), ( 8,"L", 1, -0.02,-0.20),
    ( 9,"R", 0,  0.18, 0.00), ( 9,"L", 0, -0.18, 0.00),
    (10,"R", 2,  0.65, 0.08), (10,"L", 1, -0.65, 0.08),
    (11,"R", 2,  0.35,-0.04), (11,"L", 1, -0.35,-0.04),
    (12,"R", 3,  0.22,-0.14), (12,"L", 3, -0.22,-0.14),
    (13,"R", 3,  0.28, 0.00), (13,"L", 3, -0.28, 0.00),
    (14,"R", 6,  0.26,-0.04), (14,"L", 6, -0.26,-0.04),
    (15,"R", 6,  0.38, 0.14), (15,"L", 6, -0.38, 0.14),
    (16,"R",17,  0.02, 0.00), (16,"L",16, -0.02, 0.00),
    (17,"R",14,  0.02, 0.00), (17,"L",13, -0.02, 0.00),
    (18,"R",12,  0.16, 0.00), (18,"L",12, -0.16, 0.00),
    (19,"R",17,  0.48, 0.04), (19,"L",16, -0.48, 0.04),
    (20,"R",15,  0.22,-0.06), (20,"L",15, -0.22,-0.06),
    (21,"R",15,  0.32, 0.16), (21,"L",15, -0.32, 0.16),
    (22,"R",17,  0.28, 0.00), (22,"L",16, -0.28, 0.00),
    (23,"R",17,  0.68, 0.02), (23,"L",16, -0.68, 0.02),
    (24,"R", 6,  0.72, 0.00), (24,"L", 6, -0.72, 0.00),
    (25,"R", 3,  0.68, 0.04), (25,"L", 3, -0.68, 0.04),
    (26,"R", 1,  0.78, 0.08), (26,"L", 2, -0.78, 0.08),
    (27,"R",23,  0.00, 0.00), (27,"L",22,  0.00, 0.00),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _proj(j3d):
    """3D (mm, Y-down) → 2D (x, height_up)."""
    return np.column_stack([j3d[:, 0], -j3d[:, 1]])

def _body_extent(vx, vy, y_center, band=35):
    """Return (left_x, right_x) of the body silhouette at y_center ± band."""
    m = np.abs(vy - y_center) < band
    if m.sum() < 8:
        return None, None
    return float(vx[m].min()), float(vx[m].max())

def _arrowbar(ax, x0, x1, y, label, color, fontsize=7.5):
    """Draw ↔ measurement bar with label."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.4))
    ax.text((x0+x1)/2, y, label, ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.7))

def _vline(ax, y, xmin, xmax, color="#AAAAAA", lw=0.8, ls="--"):
    ax.plot([xmin, xmax], [y, y], color=color, lw=lw, ls=ls, zorder=2)

def _spine_node(ax, x, y, num, size=14):
    """Draw a red filled square with a white number at (x, y)."""
    sq = FancyBboxPatch((x - size/2, y - size/2), size, size,
                        boxstyle="square,pad=0", linewidth=0,
                        facecolor=RED_SOLID, zorder=8)
    ax.add_patch(sq)
    ax.text(x, y, str(num), ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=9)

def _draw_triangle(ax, pts, filled=True):
    """Draw a single triangle outline (+ optional light fill)."""
    tri = MplPolygon(pts, closed=True,
                     edgecolor=RED_EDGE, linewidth=1.8,
                     facecolor=RED_FILL if filled else "none",
                     zorder=4)
    ax.add_patch(tri)

def _label_box(ax, x, y, text, color):
    """Place a small boxed label at (x, y)."""
    ax.text(x, y, text, ha="center", va="center",
            fontsize=6.3, fontweight="bold", color=color, zorder=6,
            bbox=dict(boxstyle="square,pad=0.18", fc="white",
                      ec=color, lw=0.8, alpha=0.92))


# ── Main render ───────────────────────────────────────────────────────────────
def render_physio_landmarks(fitted_joints, fitted_verts, faces,
                             measurements, physio_data,
                             out_path="physio_annotated.png"):
    j2d = _proj(fitted_joints)          # (24, 2): x, height_up
    vx  = fitted_verts[:, 0].astype(float)
    vy  = (-fitted_verts[:, 1]).astype(float)   # height-up

    body_h = j2d[:, 1].max() - j2d[:, 1].min()
    half_w = (j2d[:, 0].max() - j2d[:, 0].min()) / 2
    cx     = (j2d[:, 0].max() + j2d[:, 0].min()) / 2
    foot_y = j2d[[10, 11], 1].min()
    head_y = j2d[15,  1] + 0.12 * body_h

    # ── 6 Zone y-levels ────────────────────────────────────────────────────
    y_head     = j2d[15, 1]                           # node 1
    y_shoulder = j2d[[16, 17], 1].mean()              # node 2
    y_waist    = j2d[3,  1]                            # node 3
    y_hip      = j2d[[1, 2], 1].mean()                # node 4
    y_knee     = j2d[[4, 5], 1].mean()                # node 5
    y_ankle    = j2d[[7, 8], 1].mean()                # node 6

    zone_ys = [y_head, y_shoulder, y_waist, y_hip, y_knee, y_ankle]

    # Body silhouette x-extents at each zone level
    def ext(y): return _body_extent(vx, vy, y)
    lxH,  rxH  = _body_extent(vx, vy, y_head,     band=40)
    lxSh, rxSh = _body_extent(vx, vy, y_shoulder,  band=30)
    lxW,  rxW  = _body_extent(vx, vy, y_waist,     band=30)
    lxHp, rxHp = _body_extent(vx, vy, y_hip,       band=30)
    lxK,  rxK  = _body_extent(vx, vy, y_knee,      band=30)
    lxA,  rxA  = _body_extent(vx, vy, y_ankle,     band=28)

    # Fallbacks if vertex band empty
    def _fb(lx, rx, y_ref, frac=0.35):
        if lx is None: return cx - frac*body_h*0.5, cx + frac*body_h*0.5
        return lx, rx
    lxH, rxH   = _fb(lxH, rxH, y_head, 0.15)
    lxSh, rxSh = _fb(lxSh, rxSh, y_shoulder, 0.55)
    lxW, rxW   = _fb(lxW, rxW, y_waist, 0.30)
    lxHp, rxHp = _fb(lxHp, rxHp, y_hip, 0.45)
    lxK, rxK   = _fb(lxK, rxK, y_knee, 0.30)
    lxA, rxA   = _fb(lxA, rxA, y_ankle, 0.18)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 15), dpi=140)
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)

    # ── Mesh silhouette ───────────────────────────────────────────────────
    ax.scatter(vx, vy, s=0.4, c=MESH_DOT, alpha=0.25,
               linewidths=0, zorder=1, rasterized=True)

    # ── Skeleton lines ────────────────────────────────────────────────────
    SKEL = [(15,12),(12,9),(9,6),(6,3),(3,0),
            (0,1),(0,2),(1,4),(4,7),(7,10),(2,5),(5,8),(8,11),
            (9,13),(13,16),(16,18),(18,20),(20,22),
            (9,14),(14,17),(17,19),(19,21),(21,23),(12,15)]
    for a, b in SKEL:
        ax.plot([j2d[a,0], j2d[b,0]], [j2d[a,1], j2d[b,1]],
                color=SKELETON, lw=1.0, zorder=2, alpha=0.55)

    # ── Horizontal reference lines at each zone ───────────────────────────
    xL = min(lxH, lxSh, lxW, lxHp, lxK, lxA) - half_w * 0.6
    xR = max(rxH, rxSh, rxW, rxHp, rxK, rxA) + half_w * 0.6
    for y, lbl in zip(zone_ys,
                      ["Head","Shoulder","Waist","Hip","Knee","Ankle"]):
        ax.plot([xL, xR], [y, y], color="#9E9E9E", lw=0.9, ls="-", zorder=2)
        ax.text(xL - 5, y, lbl, ha="right", va="center",
                fontsize=6, color="#616161", style="italic")

    # ── Waistline (bold blue) ─────────────────────────────────────────────
    ax.plot([xL, xR], [y_waist, y_waist],
            color=BLUE_WAIST, lw=2.2, ls="-", zorder=3)
    ax.text(xL - 5, y_waist + body_h*0.012, "Waistline",
            ha="right", fontsize=7, color=BLUE_WAIST, fontweight="bold")

    # ── Vertical spine axis ───────────────────────────────────────────────
    ax.plot([cx, cx], [foot_y - body_h*0.03, head_y + body_h*0.04],
            color=BLUE_AXIS, lw=1.8, ls="--", zorder=3, alpha=0.8)

    # ── 6 Triangles ────────────────────────────────────────────────────────
    # Each triangle: (apex) → (left edge) → (right edge)
    # Zone 1: HEAD ↓ to shoulders (inverted ▽: apex=head, base=shoulders)
    _draw_triangle(ax, [(cx, y_head), (lxSh, y_shoulder), (rxSh, y_shoulder)])

    # Zone 2: shoulders ↑ converge to waist (▲: base=shoulders, apex=waist)
    _draw_triangle(ax, [(lxSh, y_shoulder), (rxSh, y_shoulder), (cx, y_waist)])

    # Zone 3: WAIST ↓ to hips (inverted ▽: apex=waist, base=hips)
    _draw_triangle(ax, [(cx, y_waist), (lxHp, y_hip), (rxHp, y_hip)])

    # Zone 4: hips ↑ converge to knee centre (▲: base=hips, apex=knee)
    _draw_triangle(ax, [(lxHp, y_hip), (rxHp, y_hip), (cx, y_knee)])

    # Zone 5: KNEE ↓ to ankles (inverted ▽: apex=knee, base=ankles)
    _draw_triangle(ax, [(cx, y_knee), (lxA, y_ankle), (rxA, y_ankle)])

    # Zone 6: ankles ↑ converge to floor centre (▲: small base triangle)
    floor_y = foot_y - body_h * 0.01
    _draw_triangle(ax, [(lxA, y_ankle), (rxA, y_ankle), (cx, floor_y)])

    # ── Zone labels "Triangles: 1&2" and "3,4,5" ─────────────────────────
    mid12_y = (y_head + y_waist) / 2
    mid345_y = (y_waist + floor_y) / 2
    ax.text(rxSh + half_w * 0.55, mid12_y,
            "Triangles:\n1 & 2", ha="left", va="center",
            fontsize=8.5, fontweight="bold", color="#212121",
            bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC", lw=0.7))
    ax.text(rxHp + half_w * 0.55, mid345_y,
            "3, 4,\n& 5", ha="left", va="center",
            fontsize=8.5, fontweight="bold", color="#212121",
            bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC", lw=0.7))

    # ── 6 Spine nodes (red squares) ───────────────────────────────────────
    sq_size = body_h * 0.022
    for i, (y, num) in enumerate(zip(zone_ys, range(1, 7))):
        _spine_node(ax, cx, y, num, size=sq_size)

    # ── Measurement bars ───────────────────────────────────────────────────
    sw  = measurements.get("Shoulder Width", {}).get("cm", 0)
    hw  = measurements.get("Hip Width",      {}).get("cm", 0)
    ll  = measurements.get("Left Full Leg",  {}).get("cm", 0)
    rl  = measurements.get("Right Full Leg", {}).get("cm", 0)
    lla = measurements.get("Left Full Arm",  {}).get("cm", 0)
    rla = measurements.get("Right Full Arm", {}).get("cm", 0)

    bar_y_sh  = y_shoulder + body_h * 0.055
    bar_y_hip = y_hip + body_h * 0.055

    _arrowbar(ax, lxSh, rxSh, bar_y_sh,
              f"Shoulder {sw:.1f}cm", "#E53935", 7)
    _arrowbar(ax, lxHp, rxHp, bar_y_hip,
              f"Hip {hw:.1f}cm", "#8E24AA", 7)

    # Vertical leg length lines
    for (jt, jb, label, xoff, col) in [
        (j2d[1,1], j2d[10,1], f"L Leg\n{ll:.1f}cm", lxHp - half_w*0.25, C_LEFT),
        (j2d[2,1], j2d[11,1], f"R Leg\n{rl:.1f}cm", rxHp + half_w*0.25, C_RIGHT),
        (j2d[16,1], j2d[22,1], f"L Arm\n{lla:.1f}cm", lxSh - half_w*0.50, C_LEFT),
        (j2d[17,1], j2d[23,1], f"R Arm\n{rla:.1f}cm", rxSh + half_w*0.50, C_RIGHT),
    ]:
        ax.annotate("", xy=(xoff, jb), xytext=(xoff, jt),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.1))
        ax.text(xoff, (jt+jb)/2, label, ha="center", va="center",
                fontsize=6, color=col, rotation=90,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", ec=col, lw=0.5))

    # ── Landmark numbered boxes ────────────────────────────────────────────
    placed = {}
    for (num, side, jidx, xoff_f, yoff_f) in LANDMARKS:
        px = j2d[jidx, 0] + xoff_f * half_w
        py = j2d[jidx, 1] + yoff_f * body_h
        lbl = f"{num}{side}"
        col = C_RIGHT if side == "R" else C_LEFT
        _label_box(ax, px, py, lbl, col)
        placed[lbl] = {
            "landmark": lbl, "number": num, "side": side,
            "smpl_joint_idx": jidx,
            "smpl_joint_name": JNAMES[jidx],
            "proj_x_mm": round(float(px), 1),
            "proj_y_mm": round(float(py), 1),
            "world_x_mm": round(float(fitted_joints[jidx, 0]), 1),
            "world_y_mm": round(float(fitted_joints[jidx, 1]), 1),
            "world_z_mm": round(float(fitted_joints[jidx, 2]), 1),
        }

    # ── Joint angle labels ────────────────────────────────────────────────
    def _angle_label(ax, jidx, label, yoff=0, xoff=0):
        x = j2d[jidx, 0] + xoff * half_w
        y = j2d[jidx, 1] + yoff * body_h
        ax.text(x, y, label, ha="center", va="center",
                fontsize=6.2, color="#FF6F00",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="#FFF8E1", ec="#FF8F00", lw=0.8, alpha=0.9),
                zorder=7)

    for row in physio_data.get("flat_table", []):
        ar = row.get("auto_result", "")
        if not ar or row.get("manual_required"):
            continue
        code = row.get("code", "")
        sub  = row.get("sub", "")
        if isinstance(ar, dict):
            txt = f"{code}: R={ar.get('Right','')}  L={ar.get('Left','')}"
        else:
            txt = f"{code}: {ar}"

        # place angle/slope labels at relevant joint
        if code == "2a":   _angle_label(ax, 16, txt, yoff=0.12, xoff=0)
        elif code == "2c": _angle_label(ax,  1, txt, yoff=0.11, xoff=-0.5)
        elif code == "2e": _angle_label(ax,  4, txt, yoff= 0.0, xoff=-0.8)
        elif code == "1c": _angle_label(ax,  4, txt, yoff=-0.04, xoff=-0.8)
        elif code == "3b": _angle_label(ax, 15, txt, yoff= 0.10, xoff=0)

    # ── Asymmetry banner ─────────────────────────────────────────────────
    flagged = physio_data.get("asymmetry_summary",{}).get("flagged_items", [])
    if flagged:
        ftxt = "⚠ Asymmetry: " + "  •  ".join(
            f"{f['code']} {f.get('sub','')}"
            for f in flagged[:6])
        ax.text(0.02, 0.005, ftxt, transform=ax.transAxes,
                fontsize=6.5, color="#BF360C", va="bottom",
                bbox=dict(boxstyle="round", fc="#FBE9E7", ec="#BF360C", lw=0.8))

    # ── Right/Left side headings ──────────────────────────────────────────
    ax.text(cx - half_w * 1.1, head_y, "Right\nSide",
            ha="center", va="bottom", fontsize=9,
            color=C_RIGHT, fontweight="bold")
    ax.text(cx + half_w * 1.1, head_y, "Left\nSide",
            ha="center", va="bottom", fontsize=9,
            color=C_LEFT, fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_h = [
        mpatches.Patch(facecolor=RED_FILL, edgecolor=RED_EDGE, lw=1.5,
                       label="Physiotherapy triangles (zones 1–6)"),
        mpatches.Patch(color=C_RIGHT, label="Right side landmarks"),
        mpatches.Patch(color=C_LEFT,  label="Left side landmarks"),
        Line2D([0],[0], color=BLUE_AXIS, ls="--", lw=1.5, label="Centre axis"),
        Line2D([0],[0], color=BLUE_WAIST, lw=2.0, label="Waistline divider"),
        mpatches.Patch(facecolor="#FFF8E1", edgecolor="#FF8F00",
                       label="Auto-classification"),
    ]
    ax.legend(handles=legend_h, loc="upper right",
              fontsize=6.5, framealpha=0.92, ncol=1)

    # ── Axis / title ──────────────────────────────────────────────────────
    ax.set_aspect("equal")
    margin_x = half_w * 2.2
    ax.set_xlim(cx - margin_x, cx + margin_x)
    ax.set_ylim(floor_y - body_h*0.04, head_y + body_h*0.08)
    ax.axis("off")
    ax.set_title(
        "Physiotherapy Assessment — Triangular Framework\n"
        "(1R / 1L – 27R / 27L  ·  Front View  ·  Zones 1–6)",
        fontsize=11, fontweight="bold", pad=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Physio Render] Saved: {out_path}")
    return out_path, placed


# ── Entry point called from run_pipeline.py ──────────────────────────────────
def run(logger, fitted_joints, fitted_verts, faces,
        measurements, physio_data):
    out_dir = os.path.join(logger.run_dir, "physio-render")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "physio_annotated.png")
    return render_physio_landmarks(
        fitted_joints, fitted_verts, faces,
        measurements, physio_data,
        out_path=out_path)
