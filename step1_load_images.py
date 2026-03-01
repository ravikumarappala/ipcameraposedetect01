"""
Step 1: Load Images
====================
Reads: left and right image paths
Writes: step-1-out/ (img_left, img_right)
"""
import os
import sys
import argparse
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger


def run(logger, left_path, right_path):
    """Load stereo images and log them."""
    print("\n[1/7] Loading images")
    print(f"  Left:  {left_path}")
    print(f"  Right: {right_path}")

    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None:
        raise FileNotFoundError(f"Cannot load left image: {left_path}")
    if img_right is None:
        raise FileNotFoundError(f"Cannot load right image: {right_path}")

    print(f"  Left size: {img_left.shape[1]}x{img_left.shape[0]}")
    print(f"  Right size: {img_right.shape[1]}x{img_right.shape[0]}")

    logger.log_step(1, "in", {"left_path": left_path, "right_path": right_path})
    logger.log_step(1, "out", {"img_left": img_left, "img_right": img_right})

    return img_left, img_right


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Load stereo images")
    parser.add_argument("--left", required=True, help="Left image path")
    parser.add_argument("--right", required=True, help="Right image path")
    parser.add_argument("--output", default="smpl_measurements", help="Output base dir")
    args = parser.parse_args()

    logger = StepLogger(base_dir=os.path.join(args.output, "run_logs"))
    run(logger, args.left, args.right)
    print(f"  ✓ Step 1 complete. Run dir: {logger.run_dir}")
