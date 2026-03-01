"""
SMPL Measurement Pipeline Orchestrator
=========================================
Runs all 7 steps sequentially. Each step is self-contained.
Supports resuming from any step with --from-step and --run-dir.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger
from constants import DEFAULT_CALIB_PATH, SMPL_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="SMPL Body Measurement Pipeline")
    parser.add_argument("--left", required=True, help="Left image path")
    parser.add_argument("--right", required=True, help="Right image path")
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH, help="Stereo calibration .npz")
    parser.add_argument("--smpl", default=SMPL_MODEL_PATH, help="SMPL model path")
    parser.add_argument("--output", default="smpl_measurements", help="Output base directory")
    parser.add_argument("--height", type=float, default=None, help="Known height in cm for scaling")
    parser.add_argument("--from-step", type=int, default=1, help="Resume from step N (1-7)")
    parser.add_argument("--run-dir", type=str, default=None, help="Resume into existing run dir")
    args = parser.parse_args()

    # Initialize logger
    if args.run_dir:
        logger = StepLogger.from_run_dir(args.run_dir)
    else:
        logger = StepLogger(base_dir=os.path.join(args.output, "run_logs"))

    print(f"\n{'='*60}")
    print(f"  SMPL Measurement Pipeline")
    print(f"  Run dir: {logger.run_dir}")
    print(f"  Starting from step: {args.from_step}")
    print(f"{'='*60}")

    # Step 1: Load images
    if args.from_step <= 1:
        from step1_load_images import run as run_step1
        img_left, img_right = run_step1(logger, args.left, args.right)
    else:
        img_left = img_right = None

    # Step 2: Stereo calibration & rectification
    if args.from_step <= 2:
        from step2_stereo_calib import run as run_step2
        rect_left, rect_right, stereo_info, stereo = run_step2(
            logger, calib_path=args.calib, img_left=img_left, img_right=img_right
        )
    else:
        rect_left = rect_right = stereo_info = stereo = None

    # Step 3: Pose detection
    if args.from_step <= 3:
        from step3_pose_detection import run as run_step3
        pts_left, pts_right, smpl_indices = run_step3(
            logger, rect_left=rect_left, rect_right=rect_right
        )
    else:
        pts_left = pts_right = smpl_indices = None

    # Step 4: Triangulation
    if args.from_step <= 4:
        from step4_triangulation import run as run_step4
        joints_3d_stereo, smpl_indices = run_step4(
            logger, pts_left=pts_left, pts_right=pts_right,
            smpl_indices=smpl_indices, stereo=stereo
        )
    else:
        joints_3d_stereo = None

    # Step 5: SMPL fitting
    if args.from_step <= 5:
        from step5_smpl_fitting import run as run_step5
        fitted_joints, fitted_verts, smpl_indices = run_step5(
            logger, smpl_path=args.smpl, known_height=args.height,
            joints_3d_stereo=joints_3d_stereo, smpl_indices=smpl_indices
        )
    else:
        fitted_joints = fitted_verts = None

    # Step 6: Measurements
    if args.from_step <= 6:
        from step6_measurements import run as run_step6
        measurements = run_step6(logger, fitted_joints=fitted_joints, fitted_verts=fitted_verts)
    else:
        measurements = None

    # Step 7: Rendering & summary
    if args.from_step <= 7:
        from step7_rendering import run as run_step7
        run_step7(
            logger, known_height=args.height,
            fitted_joints=fitted_joints, fitted_verts=fitted_verts,
            measurements=measurements, stereo_info=stereo_info,
            joints_3d_stereo=joints_3d_stereo, smpl_indices=smpl_indices,
        )

    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline complete!")
    print(f"  Results: {logger.run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
