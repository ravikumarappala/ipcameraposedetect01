"""
Step Logger for SMPL Measurement Pipeline
==========================================
Logs inputs and outputs of each pipeline step to a structured folder:
    run_logs/<run_date>/<timestamp>/
        step-1-in/   step-1-out/
        step-2-in/   step-2-out/
        ...
        step-7-in/   step-7-out/
        summary.csv   (final output in expected format)

Designed for dual use:
  - LOCAL:  Writes files to disk (current)
  - CLOUD:  Calls a cloud function that writes to Firestore + GCS (future)

Usage:
    logger = StepLogger(base_dir="run_logs")
    logger.log_step(1, "in", {"left_path": ..., "right_path": ...})
    logger.log_step(1, "out", {"left_image": img, "right_image": img})
    ...
    logger.write_summary(measurements, stereo_info, fitted_joints, ...)
"""

import os
import json
import csv
import cv2
import numpy as np
from datetime import datetime


class StepLogger:
    """
    Logs pipeline step inputs/outputs to a structured folder.
    Future-ready for cloud function integration.
    """

    STEP_NAMES = {
        1: "load_images",
        2: "stereo_calibration",
        3: "pose_detection_2d",
        4: "triangulation_3d",
        5: "smpl_fitting",
        6: "body_measurements",
        7: "mesh_rendering",
    }

    def __init__(self, base_dir="run_logs", use_cloud=False, cloud_endpoint=None):
        """
        Args:
            base_dir: Root directory for local file logging
            use_cloud: If True, also call cloud function (future)
            cloud_endpoint: URL of the cloud function (future)
        """
        self.use_cloud = use_cloud
        self.cloud_endpoint = cloud_endpoint

        # Create folder structure: run_logs/<date>/<timestamp>/
        now = datetime.now()
        self.run_date = now.strftime("%Y-%m-%d")
        self.run_timestamp = now.strftime("%H%M%S")
        self.run_id = now.strftime("%Y%m%d_%H%M%S")

        self.run_dir = os.path.join(base_dir, self.run_date, self.run_timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

        # Collect all step data for summary
        self._step_data = {}

        print(f"  [StepLogger] Logging to: {self.run_dir}")

    @classmethod
    def from_run_dir(cls, run_dir, use_cloud=False, cloud_endpoint=None):
        """Resume logging to an existing run directory (used by individual step files)."""
        obj = cls.__new__(cls)
        obj.use_cloud = use_cloud
        obj.cloud_endpoint = cloud_endpoint
        obj.run_dir = run_dir
        obj._step_data = {}
        # Parse run_id from path: .../2026-02-28/172751/ -> 20260228_172751
        parts = os.path.normpath(run_dir).split(os.sep)
        obj.run_date = parts[-2] if len(parts) >= 2 else datetime.now().strftime("%Y-%m-%d")
        obj.run_timestamp = parts[-1] if len(parts) >= 1 else datetime.now().strftime("%H%M%S")
        obj.run_id = obj.run_date.replace("-", "") + "_" + obj.run_timestamp
        os.makedirs(run_dir, exist_ok=True)
        print(f"  [StepLogger] Resuming in: {run_dir}")
        return obj

    def _get_step_dir(self, step_num, direction):
        """Get/create directory for step-N-in or step-N-out."""
        dir_name = f"step-{step_num}-{direction}"
        step_dir = os.path.join(self.run_dir, dir_name)
        os.makedirs(step_dir, exist_ok=True)
        return step_dir

    def load_step_output(self, step_num):
        """
        Load previously saved output from step-N-out/.

        Returns:
            dict: key -> value loaded from files (images, npy, json, scalars)
        """
        step_dir = os.path.join(self.run_dir, f"step-{step_num}-out")
        meta_path = os.path.join(step_dir, "_metadata.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No output found for step {step_num} at {step_dir}")

        with open(meta_path) as f:
            metadata = json.load(f)

        result = {}
        for key, info in metadata.get("fields", {}).items():
            ftype = info.get("type", "")

            if ftype == "image":
                img_path = info.get("file", "")
                if os.path.exists(img_path):
                    result[key] = cv2.imread(img_path)
                else:
                    result[key] = None

            elif ftype == "ndarray":
                npy_path = info.get("file_npy", "")
                if os.path.exists(npy_path):
                    result[key] = np.load(npy_path)
                else:
                    result[key] = None

            elif ftype == "dict":
                json_path = info.get("file", "")
                if os.path.exists(json_path):
                    with open(json_path) as jf:
                        result[key] = json.load(jf)
                else:
                    result[key] = info.get("data", {})

            elif ftype == "list":
                json_path = info.get("file", "")
                if os.path.exists(json_path):
                    with open(json_path) as jf:
                        result[key] = json.load(jf)
                else:
                    result[key] = info.get("data", [])

            elif ftype in ("str", "int", "float", "bool"):
                result[key] = info.get("value")

            else:
                result[key] = info.get("value", str(info))

        return result

    def log_step(self, step_num, direction, data):
        """
        Log input or output of a pipeline step.

        Args:
            step_num: int, 1-7
            direction: "in" or "out"
            data: dict with key-value pairs to log.
                  Values can be:
                    - str, int, float, list, dict → saved as JSON
                    - np.ndarray → saved as .npy (or .csv for 2D)
                    - image (np.ndarray with 3 dims) → saved as .png
                    - dict with nested items → flattened into JSON

        Returns:
            dict: payload that was logged (for cloud function compatibility)
        """
        step_dir = self._get_step_dir(step_num, direction)
        step_name = self.STEP_NAMES.get(step_num, f"step_{step_num}")

        # Build payload (JSON-serializable metadata)
        payload = {
            "run_id": self.run_id,
            "step_num": step_num,
            "step_name": step_name,
            "direction": direction,
            "timestamp": datetime.now().isoformat(),
            "fields": {},
        }

        for key, value in data.items():
            file_path = None

            if isinstance(value, np.ndarray):
                if value.ndim == 3 and value.shape[2] in (3, 4):
                    # Image → save as PNG
                    file_path = os.path.join(step_dir, f"{key}.png")
                    cv2.imwrite(file_path, value)
                    payload["fields"][key] = {
                        "type": "image",
                        "shape": list(value.shape),
                        "file": file_path,
                    }
                elif value.ndim <= 2:
                    # Small array → save as CSV; also save .npy
                    npy_path = os.path.join(step_dir, f"{key}.npy")
                    np.save(npy_path, value)
                    csv_path = os.path.join(step_dir, f"{key}.csv")
                    if value.ndim == 2:
                        np.savetxt(csv_path, value, delimiter=",", fmt="%.6f")
                    else:
                        np.savetxt(csv_path, value.reshape(1, -1), delimiter=",", fmt="%.6f")
                    payload["fields"][key] = {
                        "type": "ndarray",
                        "shape": list(value.shape),
                        "file_npy": npy_path,
                        "file_csv": csv_path,
                    }
                else:
                    npy_path = os.path.join(step_dir, f"{key}.npy")
                    np.save(npy_path, value)
                    payload["fields"][key] = {
                        "type": "ndarray",
                        "shape": list(value.shape),
                        "file_npy": npy_path,
                    }

            elif isinstance(value, dict):
                # Dict → save as JSON
                json_path = os.path.join(step_dir, f"{key}.json")
                _save_json(json_path, value)
                payload["fields"][key] = {
                    "type": "dict",
                    "file": json_path,
                    "data": _to_serializable(value),
                }

            elif isinstance(value, (list, tuple)):
                json_path = os.path.join(step_dir, f"{key}.json")
                _save_json(json_path, value)
                payload["fields"][key] = {
                    "type": "list",
                    "length": len(value),
                    "file": json_path,
                    "data": _to_serializable(value),
                }

            elif isinstance(value, (str, int, float, bool)):
                payload["fields"][key] = {
                    "type": type(value).__name__,
                    "value": value,
                }

            else:
                # Fallback: convert to string
                payload["fields"][key] = {
                    "type": "other",
                    "value": str(value),
                }

        # Save step metadata JSON
        meta_path = os.path.join(step_dir, "_metadata.json")
        _save_json(meta_path, payload)

        # Store for summary
        step_key = f"step-{step_num}-{direction}"
        self._step_data[step_key] = payload

        # ---- Cloud function placeholder ----
        if self.use_cloud:
            self._call_cloud_function(payload)

        return payload

    def write_summary(self, measurements, stereo_info, fitted_joints,
                      joints_3d_stereo, smpl_indices, smpl_joint_names,
                      known_height=None):
        """
        Write the final summary CSV in the expected format:
          Section 1: Joint positions (Joint, Left-X, Left-Y, Left-Z, Right-X, Right-Y, Right-Z)
          Section 2: Joint lengths (Joint Length, Left, Right, Difference)
          Section 3: Joint angles (Joint Angles, Left, Right, Difference)
          Section 4: Metadata (RMS, baseline, heights)

        Args:
            measurements: dict from BodyMeasurements.compute_all()
            stereo_info: dict with 'baseline_mm', 'rms' keys
            fitted_joints: (24, 3) SMPL fitted joints in mm
            joints_3d_stereo: (N, 3) raw triangulated joints in mm
            smpl_indices: list of SMPL joint indices that were matched
            smpl_joint_names: list of all 24 SMPL joint names
            known_height: float or None, user-provided height in cm
        """
        csv_path = os.path.join(self.run_dir, "summary.csv")

        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)

            # ---- Section 1: Joint positions ----
            w.writerow(["Joint", "Left-X", "Left-Y", "Left-Z",
                        "Right-X", "Right-Y", "Right-Z"])
            # For each matched joint: raw triangulated = "Left" side,
            # SMPL fitted = "Right" side (for comparison)
            for i, smpl_idx in enumerate(smpl_indices):
                name = smpl_joint_names[smpl_idx]
                raw = joints_3d_stereo[i] if i < len(joints_3d_stereo) else [0, 0, 0]
                fit = fitted_joints[smpl_idx]
                w.writerow([
                    name,
                    f"{raw[0]:.2f}", f"{raw[1]:.2f}", f"{raw[2]:.2f}",
                    f"{fit[0]:.2f}", f"{fit[1]:.2f}", f"{fit[2]:.2f}",
                ])
            # Fill remaining SMPL joints that were not observed
            observed_names = {smpl_joint_names[si] for si in smpl_indices}
            for idx in range(24):
                name = smpl_joint_names[idx]
                if name not in observed_names:
                    fit = fitted_joints[idx]
                    w.writerow([
                        name, "", "", "",
                        f"{fit[0]:.2f}", f"{fit[1]:.2f}", f"{fit[2]:.2f}",
                    ])
            w.writerow([])  # blank separator

            # ---- Section 2: Joint lengths ----
            w.writerow(["Joint Length", "Left", "Right", "Difference"])
            # Pair up left/right segment measurements
            lr_pairs = [
                ("Upper Arm", "Left Upper Arm", "Right Upper Arm"),
                ("Forearm", "Left Forearm", "Right Forearm"),
                ("Full Arm", "Left Full Arm", "Right Full Arm"),
                ("Thigh", "Left Thigh", "Right Thigh"),
                ("Shin", "Left Shin", "Right Shin"),
                ("Full Leg", "Left Full Leg", "Right Full Leg"),
            ]
            for label, left_key, right_key in lr_pairs:
                l_val = measurements.get(left_key, {}).get('cm', 0)
                r_val = measurements.get(right_key, {}).get('cm', 0)
                diff = abs(l_val - r_val)
                w.writerow([label, f"{l_val:.2f}", f"{r_val:.2f}", f"{diff:.2f}"])

            # Single-value measurements
            for key in ["Shoulder Width", "Hip Width", "Torso (Pelvis to Neck)",
                        "Spine (Pelvis to Spine3)", "Neck to Head",
                        "Arm Span", "Inseam (avg)",
                        "Chest Circumference", "Waist Circumference",
                        "Hip Circumference"]:
                if key in measurements:
                    w.writerow([key, f"{measurements[key]['cm']:.2f}", "", ""])

            w.writerow([])  # blank separator

            # ---- Section 3: Joint angles ----
            w.writerow(["Joint Angles", "Left", "Right", "Difference"])
            angle_pairs = [
                ("Elbow", (16, 18, 20), (17, 19, 21)),
                ("Knee", (1, 4, 7), (2, 5, 8)),
                ("Shoulder", (12, 16, 18), (12, 17, 19)),
                ("Hip", (0, 1, 4), (0, 2, 5)),
            ]
            for label, left_jts, right_jts in angle_pairs:
                l_angle = _compute_angle(fitted_joints, *left_jts)
                r_angle = _compute_angle(fitted_joints, *right_jts)
                diff = abs(l_angle - r_angle)
                w.writerow([label, f"{l_angle:.1f}", f"{r_angle:.1f}", f"{diff:.1f}"])

            w.writerow([])  # blank separator

            # ---- Section 4: Metadata ----
            rms_val = stereo_info.get('rms', '')
            baseline_mm = stereo_info.get('baseline_mm', '')
            calculated_height = measurements.get('Total Height (chain)', {}).get('cm', '')
            user_height = known_height if known_height else ''

            w.writerow(["RMS-Difference", rms_val if rms_val else "", "", ""])
            w.writerow(["Length between Cameras",
                        f"{baseline_mm:.2f}" if isinstance(baseline_mm, (int, float)) else "",
                        "", ""])
            w.writerow(["Height of Human-calculated",
                        f"{calculated_height:.2f}" if isinstance(calculated_height, (int, float)) else "",
                        "", ""])
            mesh_height = measurements.get('Height from Mesh', {}).get('cm', '')
            w.writerow(["Height from Mesh",
                        f"{mesh_height:.2f}" if isinstance(mesh_height, (int, float)) else "",
                        "", ""])
            w.writerow(["Height of Human-user-provided",
                        f"{user_height:.1f}" if isinstance(user_height, (int, float)) else "",
                        "", ""])

        print(f"  [StepLogger] Summary saved: {csv_path}")

        # Also save as JSON for cloud function compatibility
        summary_data = {
            "run_id": self.run_id,
            "run_date": self.run_date,
            "run_timestamp": self.run_timestamp,
            "measurements": _to_serializable(measurements),
            "stereo_info": _to_serializable(stereo_info),
            "calculated_height_cm": calculated_height if isinstance(calculated_height, (int, float)) else None,
            "user_provided_height_cm": known_height,
            "num_matched_joints": len(smpl_indices),
            "matched_joints": [smpl_joint_names[i] for i in smpl_indices],
        }
        json_path = os.path.join(self.run_dir, "summary.json")
        _save_json(json_path, summary_data)
        print(f"  [StepLogger] Summary JSON: {json_path}")

        # ---- Cloud function placeholder ----
        if self.use_cloud:
            self._call_cloud_function(summary_data)

        return csv_path

    def _call_cloud_function(self, payload):
        """
        PLACEHOLDER: Call cloud function to write data to Firestore + GCS.

        When the cloud function is ready, this method will:
          1. Serialize the payload to JSON
          2. Upload any referenced files (images, npy) to GCS
          3. POST the payload to the cloud function endpoint
          4. The cloud function will write metadata to Firestore
             and store files in GCS

        Expected cloud function signature:
            POST {self.cloud_endpoint}/log_step
            Body: {
                "run_id": "20260228_171100",
                "step_num": 1,
                "step_name": "load_images",
                "direction": "in",
                "timestamp": "2026-02-28T17:11:00",
                "fields": { ... },
                "files": { "key": "gs://bucket/path" }
            }
            Response: { "status": "ok", "doc_id": "..." }
        """
        # TODO: Implement when cloud function is ready
        #
        # import requests
        # response = requests.post(
        #     f"{self.cloud_endpoint}/log_step",
        #     json=payload,
        #     timeout=30
        # )
        # if response.status_code != 200:
        #     print(f"  [StepLogger] Cloud function error: {response.text}")
        # else:
        #     print(f"  [StepLogger] Cloud function OK: {response.json()}")
        pass


# ============================================================
# Helper functions (module-level)
# ============================================================

def _to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _save_json(path, data):
    """Save data to JSON file with numpy-safe serialization."""
    with open(path, 'w') as f:
        json.dump(_to_serializable(data), f, indent=2, default=str)


def _compute_angle(joints, ja, jb, jc):
    """Compute angle at joint jb (in degrees) between jb->ja and jb->jc."""
    va = joints[ja] - joints[jb]
    vb = joints[jc] - joints[jb]
    cos_a = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
