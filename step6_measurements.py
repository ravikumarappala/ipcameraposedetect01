"""
Step 6: Body Measurements
===========================
Reads: step-5-out/ (fitted_joints, fitted_verts)
Writes: step-6-out/ (measurements)
"""
import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger
from constants import SMPL_JOINT_NAMES, SEGMENT_MEASUREMENTS, HEIGHT_PATH


# ============================================================
# BodyMeasurements — fully contained in this step
# ============================================================
class BodyMeasurements:
    """Compute body measurements from SMPL joints and mesh."""

    def __init__(self, joints_3d, vertices_3d=None):
        self.joints = joints_3d
        self.vertices = vertices_3d
        self.measurements = {}

    def compute_all(self):
        self._compute_segment_lengths()
        self._compute_height()
        self._compute_arm_span()
        self._compute_inseam()
        if self.vertices is not None:
            self._compute_circumferences()
        return self.measurements

    def _joint_idx(self, name):
        return SMPL_JOINT_NAMES.index(name)

    def _dist(self, j1, j2):
        i1 = self._joint_idx(j1) if isinstance(j1, str) else j1
        i2 = self._joint_idx(j2) if isinstance(j2, str) else j2
        return float(np.linalg.norm(self.joints[i1] - self.joints[i2]))

    def _compute_segment_lengths(self):
        for name, j1, j2 in SEGMENT_MEASUREMENTS:
            length_mm = self._dist(j1, j2)
            self.measurements[name] = {
                'mm': length_mm,
                'cm': length_mm / 10.0,
                'in': length_mm / 25.4,
            }

    def _compute_height(self):
        total = 0.0
        for i in range(len(HEIGHT_PATH) - 1):
            total += self._dist(HEIGHT_PATH[i], HEIGHT_PATH[i+1])
        self.measurements['Total Height (chain)'] = {
            'mm': total, 'cm': total / 10.0, 'in': total / 25.4,
        }
        all_y = self.joints[:, 1]
        vertical_span = float(np.max(all_y) - np.min(all_y))
        self.measurements['Vertical Span'] = {
            'mm': vertical_span, 'cm': vertical_span / 10.0, 'in': vertical_span / 25.4,
        }
        # Vertex-based height: use full vertex Y span, minus toe overshoot
        # Works in any coordinate system (Y-up or Y-down) because we use
        # absolute span and subtract only the toe-to-ankle extension
        if self.vertices is not None:
            # Full vertex span in Y
            full_span = float(np.max(self.vertices[:, 1]) - np.min(self.vertices[:, 1]))
            # Toe overshoot: distance from extreme foot vertex past the ankle
            ankle_y_min = min(self.joints[self._joint_idx("l_ankle"), 1],
                             self.joints[self._joint_idx("r_ankle"), 1])
            ankle_y_max = max(self.joints[self._joint_idx("l_ankle"), 1],
                             self.joints[self._joint_idx("r_ankle"), 1])
            vert_y_min = float(np.min(self.vertices[:, 1]))
            vert_y_max = float(np.max(self.vertices[:, 1]))
            # Foot vertices extend beyond ankle in the "down" direction
            # Determine which end is feet (where ankle is closer to the extreme)
            dist_to_min = abs(ankle_y_min - vert_y_min)
            dist_to_max = abs(ankle_y_max - vert_y_max)
            toe_overshoot = min(dist_to_min, dist_to_max)
            # Also clip the top: use 99th/1st percentile for skull
            vert_y_p1 = float(np.percentile(self.vertices[:, 1], 1))
            vert_y_p99 = float(np.percentile(self.vertices[:, 1], 99))
            clipped_span = abs(vert_y_p99 - vert_y_p1)
            mesh_height = clipped_span - toe_overshoot
            self.measurements['Height from Mesh'] = {
                'mm': mesh_height, 'cm': mesh_height / 10.0, 'in': mesh_height / 25.4,
            }

    def _compute_arm_span(self):
        arm_span = (
            self._dist("l_hand", "l_wrist") +
            self._dist("l_wrist", "l_elbow") +
            self._dist("l_elbow", "l_shoulder") +
            self._dist("l_shoulder", "r_shoulder") +
            self._dist("r_shoulder", "r_elbow") +
            self._dist("r_elbow", "r_wrist") +
            self._dist("r_wrist", "r_hand")
        )
        self.measurements['Arm Span'] = {
            'mm': arm_span, 'cm': arm_span / 10.0, 'in': arm_span / 25.4,
        }

    def _compute_inseam(self):
        left_inseam = self._dist("l_hip", "l_knee") + self._dist("l_knee", "l_ankle")
        right_inseam = self._dist("r_hip", "r_knee") + self._dist("r_knee", "r_ankle")
        avg_inseam = (left_inseam + right_inseam) / 2.0
        self.measurements['Inseam (avg)'] = {
            'mm': avg_inseam, 'cm': avg_inseam / 10.0, 'in': avg_inseam / 25.4,
        }

    def _compute_circumferences(self):
        if self.vertices is None:
            return

        # Chest
        chest_y = self.joints[self._joint_idx("spine2"), 1]
        chest_verts = self.vertices[np.abs(self.vertices[:, 1] - chest_y) < 20]
        if len(chest_verts) > 10:
            c = self._estimate_circumference(chest_verts)
            self.measurements['Chest Circumference'] = {
                'mm': c, 'cm': c / 10.0, 'in': c / 25.4,
            }

        # Waist
        waist_y = (self.joints[self._joint_idx("pelvis"), 1] +
                   self.joints[self._joint_idx("spine1"), 1]) / 2
        waist_verts = self.vertices[np.abs(self.vertices[:, 1] - waist_y) < 20]
        if len(waist_verts) > 10:
            c = self._estimate_circumference(waist_verts)
            self.measurements['Waist Circumference'] = {
                'mm': c, 'cm': c / 10.0, 'in': c / 25.4,
            }

        # Hip
        hip_y = (self.joints[self._joint_idx("l_hip"), 1] +
                 self.joints[self._joint_idx("r_hip"), 1]) / 2
        hip_verts = self.vertices[np.abs(self.vertices[:, 1] - hip_y) < 20]
        if len(hip_verts) > 10:
            c = self._estimate_circumference(hip_verts)
            self.measurements['Hip Circumference'] = {
                'mm': c, 'cm': c / 10.0, 'in': c / 25.4,
            }

    def _estimate_circumference(self, verts_slice):
        pts_2d = verts_slice[:, [0, 2]]
        if len(pts_2d) < 3:
            return 0.0
        pts_2d_f = pts_2d.astype(np.float32)
        hull = cv2.convexHull(pts_2d_f)
        perimeter = cv2.arcLength(hull, closed=True)
        return float(perimeter)


def run(logger, fitted_joints=None, fitted_verts=None):
    """Compute body measurements from SMPL fitted joints/vertices."""
    print("\n[6/7] Computing body measurements")

    # Load from step 5 if not provided
    if fitted_joints is None:
        step5 = logger.load_step_output(5)
        fitted_joints = np.array(step5["fitted_joints"])

    if fitted_verts is None:
        verts_path = os.path.join(logger.run_dir, "step-5-out", "fitted_verts.npy")
        if os.path.exists(verts_path):
            fitted_verts = np.load(verts_path)

    measurer = BodyMeasurements(fitted_joints, fitted_verts)
    measurements = measurer.compute_all()

    # Print measurements
    print(f"\n{'='*60}")
    print("  BODY MEASUREMENTS")
    print(f"{'='*60}")
    for name, vals in measurements.items():
        cm_str = f"{vals['cm']:.1f} cm"
        in_str = f"{vals['in']:.1f} in"
        print(f"  {name:.<42s} {cm_str:>8s}  ({in_str:>7s})")
    print(f"{'='*60}")

    logger.log_step(6, "in", {
        "num_joints": len(fitted_joints),
        "num_vertices": len(fitted_verts) if fitted_verts is not None else 0,
    })
    logger.log_step(6, "out", {
        "measurements": measurements,
    })

    return measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: Body measurements")
    parser.add_argument("--run-dir", required=True, help="Run directory from step 5")
    args = parser.parse_args()

    logger = StepLogger.from_run_dir(args.run_dir)
    run(logger)
    print(f"  ✓ Step 6 complete. Run dir: {logger.run_dir}")
