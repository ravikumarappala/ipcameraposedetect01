"""
Step 3: Pose Detection (MediaPipe)
====================================
Reads: step-2-out/ (rect_left, rect_right)
Writes: step-3-out/ (pts_left, pts_right, smpl_indices, correspondences)
"""
import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger
from constants import SMPL_JOINT_NAMES, MP_TO_SMPL, MP_MIDPOINT_TO_SMPL


# ============================================================
# PoseDetector — fully contained in this step
# ============================================================
class PoseDetector:
    """MediaPipe-based 2D pose detection."""

    def __init__(self, min_confidence=0.5):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        print("  MediaPipe pose detector initialized.")

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        h, w = image.shape[:2]
        landmarks = np.zeros((33, 3))
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = [lm.x * w, lm.y * h, lm.visibility]

        return landmarks

    def get_stereo_correspondences(self, landmarks_left, landmarks_right):
        pts_left = []
        pts_right = []
        mp_indices = []
        smpl_indices = []

        min_vis = 0.5

        # Direct MP->SMPL mappings
        for mp_idx, smpl_idx in MP_TO_SMPL.items():
            vis_L = landmarks_left[mp_idx, 2]
            vis_R = landmarks_right[mp_idx, 2]

            if vis_L >= min_vis and vis_R >= min_vis:
                pts_left.append(landmarks_left[mp_idx, :2])
                pts_right.append(landmarks_right[mp_idx, :2])
                mp_indices.append(mp_idx)
                smpl_indices.append(smpl_idx)

        # Midpoint-derived joints (neck from shoulders, pelvis from hips)
        for (mp1, mp2), smpl_idx in MP_MIDPOINT_TO_SMPL.items():
            vis_L1 = landmarks_left[mp1, 2]
            vis_L2 = landmarks_left[mp2, 2]
            vis_R1 = landmarks_right[mp1, 2]
            vis_R2 = landmarks_right[mp2, 2]

            if vis_L1 >= min_vis and vis_L2 >= min_vis and vis_R1 >= min_vis and vis_R2 >= min_vis:
                mid_left = (landmarks_left[mp1, :2] + landmarks_left[mp2, :2]) / 2.0
                mid_right = (landmarks_right[mp1, :2] + landmarks_right[mp2, :2]) / 2.0
                pts_left.append(mid_left)
                pts_right.append(mid_right)
                mp_indices.append(f"mid({mp1},{mp2})")
                smpl_indices.append(smpl_idx)

        if len(pts_left) == 0:
            return None, None, None, None

        return (
            np.array(pts_left),
            np.array(pts_right),
            mp_indices,
            smpl_indices
        )


def run(logger, rect_left=None, rect_right=None):
    """Detect 2D poses and compute stereo correspondences."""
    print("\n[3/7] Detecting 2D poses with MediaPipe")

    # Load from step 2 if not provided
    if rect_left is None or rect_right is None:
        step2 = logger.load_step_output(2)
        rect_left = step2["rect_left"]
        rect_right = step2["rect_right"]

    detector = PoseDetector()
    landmarks_left = detector.detect(rect_left)
    landmarks_right = detector.detect(rect_right)

    if landmarks_left is None or landmarks_right is None:
        raise RuntimeError("Pose detection failed on one or both images.")

    pts_left, pts_right, mp_indices, smpl_indices = \
        detector.get_stereo_correspondences(landmarks_left, landmarks_right)

    if pts_left is None:
        raise RuntimeError("No stereo correspondences found.")

    print(f"  Matched landmarks: {len(smpl_indices)}")
    smpl_names = [SMPL_JOINT_NAMES[i] for i in smpl_indices]
    print(f"  SMPL joints used: {smpl_names}")

    logger.log_step(3, "in", {
        "rect_left": rect_left,
        "rect_right": rect_right,
    })
    logger.log_step(3, "out", {
        "pts_left": pts_left,
        "pts_right": pts_right,
        "smpl_indices": smpl_indices,
    })

    return pts_left, pts_right, smpl_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Pose detection")
    parser.add_argument("--run-dir", required=True, help="Run directory from step 2")
    args = parser.parse_args()

    logger = StepLogger.from_run_dir(args.run_dir)
    run(logger)
    print(f"  ✓ Step 3 complete. Run dir: {logger.run_dir}")
