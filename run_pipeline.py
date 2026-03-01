"""
SMPL Measurement Pipeline Orchestrator
=========================================
Runs all 7 steps sequentially. Each step is self-contained.
Supports resuming from any step with --from-step and --run-dir.

CF logging (exec-logger Cloud Function):
  • 1 run doc  → /run  (create_run)
  • 7 step docs → /step (one per step, with file attachments where produced)
  • 1 summary doc → /step stepNum=99 (log_summary with measurements + summary.csv)
  • run marked complete → /run (complete_run)

Disable: export CF_ENABLED=false
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger
from constants import DEFAULT_CALIB_PATH, SMPL_MODEL_PATH
from cf_logger import CFLogger


def _step_file(logger, step_num: int, filename: str) -> str:
    """Return path to a step output file; empty string if not found."""
    p = os.path.join(logger.run_dir, f"step-{step_num}-out", filename)
    return p if os.path.isfile(p) else ""


def main():
    parser = argparse.ArgumentParser(description="SMPL Body Measurement Pipeline")
    parser.add_argument("--left",      required=True,               help="Left image path")
    parser.add_argument("--right",     required=True,               help="Right image path")
    parser.add_argument("--calib",     default=DEFAULT_CALIB_PATH,  help="Stereo calibration .npz")
    parser.add_argument("--smpl",      default=SMPL_MODEL_PATH,     help="SMPL model path")
    parser.add_argument("--output",    default="smpl_measurements",  help="Output base directory")
    parser.add_argument("--height",    type=float, default=None,    help="Known height in cm")
    parser.add_argument("--from-step", type=int,   default=1,       help="Resume from step N (1-7)")
    parser.add_argument("--run-dir",   type=str,   default=None,    help="Resume into existing run dir")
    parser.add_argument("--cf-run-id", type=str,   default=None,    help="CF run ID (auto-generated if omitted)")
    args = parser.parse_args()

    # ── Prompt for height ──────────────────────────────────────────────────────
    if args.height is None:
        raw = input("Enter person's height in cm [default 168]: ").strip()
        args.height = float(raw) if raw else 168.0
        print(f"  Using height: {args.height} cm")

    # ── Local logger ───────────────────────────────────────────────────────────
    if args.run_dir:
        logger = StepLogger.from_run_dir(args.run_dir)
    else:
        logger = StepLogger(base_dir=os.path.join(args.output, "run_logs"))

    # ── CF logger + run entry ──────────────────────────────────────────────────
    cf = CFLogger(run_id=args.cf_run_id)
    cf.create_run(
        program="run_pipeline.py",
        cmd=(f"run_pipeline.py --left {args.left} --right {args.right} "
             f"--calib {args.calib} --height {args.height}"),
        options={
            "left":      args.left,
            "right":     args.right,
            "calib":     args.calib,
            "height":    args.height,
            "smpl":      args.smpl,
            "from_step": args.from_step,
            "run_dir":   logger.run_dir,
        },
    )

    print(f"\n{'='*60}")
    print(f"  SMPL Measurement Pipeline")
    print(f"  Local run dir: {logger.run_dir}")
    print(f"  CF run_id:     {cf.run_id}")
    print(f"  Start step:    {args.from_step}")
    print(f"{'='*60}")

    # ══════════════════════════════════════════════════════════════════
    # Step 1: Load images
    # ══════════════════════════════════════════════════════════════════
    if args.from_step <= 1:
        from step1_load_images import run as run_step1
        img_left, img_right = run_step1(logger, args.left, args.right)

        cf.log_step(1,
            input_text=(f"left={args.left} ({img_left.shape[1]}x{img_left.shape[0]}px), "
                        f"right={args.right} ({img_right.shape[1]}x{img_right.shape[0]}px)"),
            output_text="images loaded OK — ready for stereo calibration",
            file_path=args.left,          # attach the left input image
            file_mime="image/png",
        )
    else:
        img_left = img_right = None

    # ══════════════════════════════════════════════════════════════════
    # Step 2: Stereo calibration & rectification
    # ══════════════════════════════════════════════════════════════════
    if args.from_step <= 2:
        from step2_stereo_calib import run as run_step2
        rect_left, rect_right, stereo_info, stereo = run_step2(
            logger, calib_path=args.calib, img_left=img_left, img_right=img_right
        )
        baseline = stereo_info.get("baseline_mm", "?")
        img_sz   = stereo_info.get("img_size", "?")
        rms      = stereo_info.get("rms", "?")

        cf.log_step(2,
            input_text=f"calib={args.calib}",
            output_text=(f"baseline={baseline:.1f}mm | imgSize={img_sz} | "
                         f"rms={rms}" if isinstance(baseline, float) else
                         f"baseline={baseline} | imgSize={img_sz}"),
        )
    else:
        rect_left = rect_right = stereo_info = stereo = None

    # ══════════════════════════════════════════════════════════════════
    # Step 3: Pose detection
    # ══════════════════════════════════════════════════════════════════
    if args.from_step <= 3:
        from step3_pose_detection import run as run_step3
        pts_left, pts_right, smpl_indices = run_step3(
            logger, rect_left=rect_left, rect_right=rect_right
        )
        from constants import SMPL_JOINT_NAMES
        joint_names = [SMPL_JOINT_NAMES[i] for i in smpl_indices]

        cf.log_step(3,
            input_text="MediaPipe pose detection on rectified stereo images",
            output_text=(f"matched {len(smpl_indices)} landmarks: "
                         f"{', '.join(joint_names)}"),
            # attach right image as it shows the second camera's pose
            file_path=args.right,
            file_mime="image/png",
        )
    else:
        pts_left = pts_right = smpl_indices = None

    # ══════════════════════════════════════════════════════════════════
    # Step 4: Triangulation
    # ══════════════════════════════════════════════════════════════════
    if args.from_step <= 4:
        from step4_triangulation import run as run_step4
        joints_3d_stereo, smpl_indices = run_step4(
            logger, pts_left=pts_left, pts_right=pts_right,
            smpl_indices=smpl_indices, stereo=stereo
        )
        from constants import SMPL_JOINT_NAMES
        coord_summary = " | ".join(
            f"{SMPL_JOINT_NAMES[smpl_indices[i]]}: "
            f"X={joints_3d_stereo[i][0]:.0f} Y={joints_3d_stereo[i][1]:.0f} Z={joints_3d_stereo[i][2]:.0f}"
            for i in range(min(5, len(joints_3d_stereo)))
        )
        cf.log_step(4,
            input_text=f"triangulate {len(smpl_indices)} joint correspondences",
            output_text=f"triangulated {len(joints_3d_stereo)} 3D joints (mm). "
                        f"First 5: {coord_summary}",
        )
    else:
        joints_3d_stereo = None

    # ══════════════════════════════════════════════════════════════════
    # Step 5: SMPL fitting
    # ══════════════════════════════════════════════════════════════════
    if args.from_step <= 5:
        from step5_smpl_fitting import run as run_step5
        fitted_joints, fitted_verts, smpl_indices = run_step5(
            logger, smpl_path=args.smpl, known_height=args.height,
            joints_3d_stereo=joints_3d_stereo, smpl_indices=smpl_indices
        )
        cf.log_step(5,
            input_text=(f"SMPL fitting: {len(smpl_indices)} joints | "
                        f"height={args.height}cm | model={os.path.basename(args.smpl)}"),
            output_text=(f"fitted {len(fitted_joints)} joints, {len(fitted_verts)} vertices | "
                         f"height_scale applied to {args.height}cm"),
        )
    else:
        fitted_joints = fitted_verts = None

    # ══════════════════════════════════════════════════════════════════
    # Step 6: Measurements
    # ══════════════════════════════════════════════════════════════════
    if args.from_step <= 6:
        from step6_measurements import run as run_step6
        measurements = run_step6(
            logger, fitted_joints=fitted_joints, fitted_verts=fitted_verts
        )
        key_meas = ["Shoulder Width", "Torso (Pelvis to Neck)", "Left Full Arm",
                    "Right Full Arm", "Left Full Leg", "Right Full Leg",
                    "Chest Circumference", "Total Height (chain)"]
        meas_str = " | ".join(
            f"{k}={measurements[k]['cm']:.1f}cm"
            for k in key_meas if k in measurements
        )
        # summary.csv is written by step7 — attach after step 6 run
        cf.log_step(6,
            input_text=f"compute body measurements from {len(fitted_joints)} SMPL joints",
            output_text=meas_str,
        )
    else:
        measurements = None

    # ══════════════════════════════════════════════════════════════════
    # Step 7: Rendering & summary
    # ══════════════════════════════════════════════════════════════════
    if args.from_step <= 7:
        from step7_rendering import run as run_step7
        run_step7(
            logger, known_height=args.height,
            fitted_joints=fitted_joints, fitted_verts=fitted_verts,
            measurements=measurements, stereo_info=stereo_info,
            joints_3d_stereo=joints_3d_stereo, smpl_indices=smpl_indices,
        )
        # Attach annotated mesh PNG
        annotated_img = _step_file(logger, 7, "annotated_mesh.png")
        cf.log_step(7,
            input_text="render SMPL 3D mesh with joints, segment lengths & joint angles",
            output_text=f"annotated_mesh.png + summary.csv written to {logger.run_dir}",
            file_path=annotated_img,
            file_mime="image/png",
        )

    # ══════════════════════════════════════════════════════════════════
    # Summary entry (stepNum=99) — attach summary.csv + flat measurements
    # ══════════════════════════════════════════════════════════════════
    if measurements:
        summary_csv  = os.path.join(logger.run_dir, "summary.csv")
        summary_json = os.path.join(logger.run_dir, "summary.json")
        cf.log_summary(
            measurements=measurements,
            height_cm=args.height,
            csv_path=summary_csv,
            json_path=summary_json,
        )

    # Mark run complete
    cf.complete_run(status="complete")

    print(f"\n{'='*60}")
    print(f"  ✓ Pipeline complete!")
    print(f"  Local results: {logger.run_dir}")
    print(f"  CF run_id:     {cf.run_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
