"""
Step 2: Stereo Calibration & Rectification
============================================
Reads: step-1-out/ (img_left, img_right) + calibration .npz file
Writes: step-2-out/ (rect_left, rect_right, stereo_info, P1, P2)
"""
import os
import sys
import argparse
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger
from constants import DEFAULT_CALIB_PATH


# ============================================================
# StereoProcessor — fully contained in this step
# ============================================================
class StereoProcessor:
    """Handles stereo calibration, rectification, and triangulation."""

    def __init__(self, calib_path):
        print(f"Loading stereo calibration from {calib_path}...")
        data = np.load(calib_path)

        self.K1 = data['K1']
        self.dist1 = data['dist1']
        self.K2 = data['K2']
        self.dist2 = data['dist2']
        self.R = data['R']
        self.T = data['T']
        self.img_size = tuple(data['img_size'])

        baseline_mm = float(np.linalg.norm(self.T))
        print(f"  Image size: {self.img_size}")
        print(f"  Baseline: {baseline_mm:.2f} mm")

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.dist1, self.K2, self.dist2,
            self.img_size, self.R, self.T, alpha=0
        )

        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.dist1, self.R1, self.P1,
            self.img_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.dist2, self.R2, self.P2,
            self.img_size, cv2.CV_32FC1
        )

        self.stereo_info = {
            "baseline_mm": baseline_mm,
            "img_size": list(self.img_size),
            "rms": float(data['rms']) if 'rms' in data else None,
        }

        print("  Stereo rectification maps computed.")

    def rectify(self, img_left, img_right):
        rect_left = cv2.remap(img_left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right


def run(logger, calib_path=DEFAULT_CALIB_PATH, img_left=None, img_right=None):
    """Load calibration, rectify images, and save projection matrices."""
    print("\n[2/7] Loading stereo calibration")

    # Load images from step 1 if not provided
    if img_left is None or img_right is None:
        step1 = logger.load_step_output(1)
        img_left = step1["img_left"]
        img_right = step1["img_right"]

    stereo = StereoProcessor(calib_path)
    rect_left, rect_right = stereo.rectify(img_left, img_right)

    logger.log_step(2, "in", {"calib_path": calib_path})
    logger.log_step(2, "out", {
        "rect_left": rect_left,
        "rect_right": rect_right,
        "stereo_info": stereo.stereo_info,
    })

    # Save projection matrices for step 4 triangulation
    out_dir = os.path.join(logger.run_dir, "step-2-out")
    np.save(os.path.join(out_dir, "P1.npy"), stereo.P1)
    np.save(os.path.join(out_dir, "P2.npy"), stereo.P2)

    return rect_left, rect_right, stereo.stereo_info, stereo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Stereo calibration & rectification")
    parser.add_argument("--run-dir", required=True, help="Run directory from step 1")
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH, help="Stereo calibration .npz")
    args = parser.parse_args()

    logger = StepLogger.from_run_dir(args.run_dir)
    run(logger, calib_path=args.calib)
    print(f"  ✓ Step 2 complete. Run dir: {logger.run_dir}")
