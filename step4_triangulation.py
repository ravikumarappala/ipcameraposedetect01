"""
Step 4: 3D Triangulation
=========================
Reads: step-2-out/ (P1.npy, P2.npy) + step-3-out/ (pts_left, pts_right, smpl_indices)
Writes: step-4-out/ (joints_3d_stereo)
"""
import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger
from constants import SMPL_JOINT_NAMES


def triangulate_points(P1, P2, pts_left, pts_right):
    """Triangulate 3D points from stereo correspondences using projection matrices."""
    pts_4d = cv2.triangulatePoints(
        P1, P2,
        pts_left.T.astype(np.float64),
        pts_right.T.astype(np.float64)
    )
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


def run(logger, pts_left=None, pts_right=None, smpl_indices=None, stereo=None):
    """Triangulate 3D joint coordinates from stereo correspondences."""
    print("\n[4/7] Triangulating 3D joints")

    # Load correspondences from step 3 if not provided
    if pts_left is None or pts_right is None or smpl_indices is None:
        step3 = logger.load_step_output(3)
        pts_left = np.array(step3["pts_left"])
        pts_right = np.array(step3["pts_right"])
        smpl_indices = step3["smpl_indices"]

    # Load projection matrices from step 2 output
    step2_dir = os.path.join(logger.run_dir, "step-2-out")
    P1_path = os.path.join(step2_dir, "P1.npy")
    P2_path = os.path.join(step2_dir, "P2.npy")

    if os.path.exists(P1_path) and os.path.exists(P2_path):
        P1 = np.load(P1_path)
        P2 = np.load(P2_path)
    elif stereo is not None:
        # Fallback: use stereo object directly (when called from pipeline)
        P1 = stereo.P1
        P2 = stereo.P2
    else:
        raise FileNotFoundError("P1.npy/P2.npy not found in step-2-out/ and no stereo object provided")

    # Triangulate
    joints_3d_stereo = triangulate_points(P1, P2, pts_left, pts_right)

    print(f"  Triangulated {len(joints_3d_stereo)} joints")
    for i, idx in enumerate(smpl_indices):
        name = SMPL_JOINT_NAMES[idx]
        x, y, z = joints_3d_stereo[i]
        print(f"    {name:>15s}: X={x:7.1f}  Y={y:7.1f}  Z={z:7.1f} mm")

    logger.log_step(4, "in", {
        "num_correspondences": len(pts_left),
        "smpl_indices": smpl_indices,
    })
    logger.log_step(4, "out", {
        "joints_3d_stereo": joints_3d_stereo,
    })

    return joints_3d_stereo, smpl_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: 3D triangulation")
    parser.add_argument("--run-dir", required=True, help="Run directory from step 3")
    args = parser.parse_args()

    logger = StepLogger.from_run_dir(args.run_dir)
    run(logger)
    print(f"  ✓ Step 4 complete. Run dir: {logger.run_dir}")
