"""
SMPL-based Stereo Body Measurement System
==========================================
Pipeline:
1. Load stereo image (side-by-side) -> split into left/right
2. Load stereo calibration parameters
3. Detect 2D pose using MediaPipe in both views
4. Triangulate 3D joints using stereo geometry
5. Fit SMPL model to refine 3D joints
6. Extract comprehensive body measurements from SMPL mesh
7. Visualize and save results

Usage:
    python smpl_measure.py --left <left.png> --right <right.png> --calib stereo_params.npz
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import pickle
import time
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# Configuration
# ============================================================
SMPL_MODEL_PATH = "/media/raviappala/edgeextvol2/jetson-inference/3dpose/smpl_models/smpl/SMPL_NEUTRAL.pkl"
DEFAULT_CALIB_PATH = "stereo_params_sidebyside.npz"
OUTPUT_DIR = "smpl_measurements"

# SMPL joint names (24 joints)
SMPL_JOINT_NAMES = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee",
    "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
    "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"
]

# MediaPipe to SMPL joint mapping
# MediaPipe has 33 landmarks; we map as many as possible to SMPL's 24 joints
MP_TO_SMPL = {
    0:  15,  # nose -> head (closest approximation)
    11: 16,  # left_shoulder -> l_shoulder
    12: 17,  # right_shoulder -> r_shoulder
    13: 18,  # left_elbow -> l_elbow
    14: 19,  # right_elbow -> r_elbow
    15: 20,  # left_wrist -> l_wrist
    16: 21,  # right_wrist -> r_wrist
    17: 22,  # left_pinky -> l_hand (closest)
    18: 23,  # right_pinky -> r_hand (closest)
    23: 1,   # left_hip -> l_hip
    24: 2,   # right_hip -> r_hip
    25: 4,   # left_knee -> l_knee
    26: 5,   # right_knee -> r_knee
    27: 7,   # left_ankle -> l_ankle
    28: 8,   # right_ankle -> r_ankle
    29: 10,  # left_heel -> l_foot
    30: 11,  # right_heel -> r_foot
}

# MediaPipe landmark pairs that can be averaged to derive SMPL joints
# (mp_idx1, mp_idx2) -> smpl_idx
MP_MIDPOINT_TO_SMPL = {
    (11, 12): 12,  # midpoint(l_shoulder, r_shoulder) -> neck
    (23, 24): 0,   # midpoint(l_hip, r_hip) -> pelvis
}

# Measurement definitions
SEGMENT_MEASUREMENTS = [
    ("Shoulder Width", "l_shoulder", "r_shoulder"),
    ("Left Upper Arm", "l_shoulder", "l_elbow"),
    ("Left Forearm", "l_elbow", "l_wrist"),
    ("Left Full Arm", "l_shoulder", "l_wrist"),
    ("Right Upper Arm", "r_shoulder", "r_elbow"),
    ("Right Forearm", "r_elbow", "r_wrist"),
    ("Right Full Arm", "r_shoulder", "r_wrist"),
    ("Neck to Head", "neck", "head"),
    ("Torso (Pelvis to Neck)", "pelvis", "neck"),
    ("Spine (Pelvis to Spine3)", "pelvis", "spine3"),
    ("Hip Width", "l_hip", "r_hip"),
    ("Left Thigh", "l_hip", "l_knee"),
    ("Left Shin", "l_knee", "l_ankle"),
    ("Left Full Leg", "l_hip", "l_ankle"),
    ("Right Thigh", "r_hip", "r_knee"),
    ("Right Shin", "r_knee", "r_ankle"),
    ("Right Full Leg", "r_hip", "r_ankle"),
]

# Height path: from ankle up through body
HEIGHT_PATH = ["l_ankle", "l_knee", "l_hip", "pelvis", "spine1", "spine2", "spine3", "neck", "head"]

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12),
    (12, 13), (12, 14),
    (13, 16), (16, 18), (18, 20), (20, 22),
    (14, 17), (17, 19), (19, 21), (21, 23),
    (12, 15),
]


# ============================================================
# SMPL Model (Minimal Implementation)
# ============================================================
class SMPLModel:
    """Minimal SMPL model loader and forward pass using PyTorch."""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        print(f"Loading SMPL model from {model_path}...")

        # Stub to replace chumpy objects during unpickling
        class _ChumpyStub:
            def __init__(self, *args, **kwargs):
                pass
            def __setstate__(self, state):
                if isinstance(state, dict) and 'x' in state:
                    self._data = np.array(state['x'])
                elif isinstance(state, np.ndarray):
                    self._data = state
                else:
                    self._data = None
            def __array__(self):
                if hasattr(self, '_data') and self._data is not None:
                    return self._data
                return np.array([])

        import types
        fake_ch = types.ModuleType('chumpy')
        fake_ch.Ch = _ChumpyStub
        fake_ch.array = _ChumpyStub
        fake_ch_linalg = types.ModuleType('chumpy.linalg')
        fake_ch_utils = types.ModuleType('chumpy.utils')
        sys.modules['chumpy'] = fake_ch
        sys.modules['chumpy.ch'] = fake_ch
        sys.modules['chumpy.linalg'] = fake_ch_linalg
        sys.modules['chumpy.utils'] = fake_ch_utils
        for attr in ['Ch', 'array', 'MatVecMult', 'Select', 'Rodrigues',
                     'dot', 'multiply', 'subtract', 'add']:
            setattr(fake_ch, attr, _ChumpyStub)
            setattr(fake_ch_linalg, attr, _ChumpyStub)

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f, encoding='latin1')

        # Convert chumpy stubs to numpy
        for k, v in model_data.items():
            if isinstance(v, _ChumpyStub):
                model_data[k] = np.array(v)
            elif hasattr(v, 'r'):
                model_data[k] = np.array(v.r)

        # Extract model components
        self.v_template = torch.tensor(
            np.array(model_data['v_template']), dtype=torch.float32, device=device
        )  # (6890, 3)

        self.shapedirs = torch.tensor(
            np.array(model_data['shapedirs'])[:, :, :10], dtype=torch.float32, device=device
        )  # (6890, 3, 10)

        self.posedirs = torch.tensor(
            np.array(model_data['posedirs']), dtype=torch.float32, device=device
        )  # (6890, 3, 207)

        self.J_regressor = torch.tensor(
            np.array(model_data['J_regressor'].todense()), dtype=torch.float32, device=device
        )  # (24, 6890)

        self.weights = torch.tensor(
            np.array(model_data['weights']), dtype=torch.float32, device=device
        )  # (6890, 24)

        self.kintree_table = np.array(model_data['kintree_table']).astype(np.int64)
        self.faces = np.array(model_data['f']).astype(np.int32)

        self.n_joints = 24
        self.n_verts = 6890
        self.n_betas = 10

        self.parent = {
            i: int(self.kintree_table[0, i])
            for i in range(1, self.kintree_table.shape[1])
        }

        print(f"  Vertices: {self.n_verts}, Joints: {self.n_joints}, Faces: {len(self.faces)}")
        print("  SMPL model loaded successfully.")

    def rodrigues(self, r):
        """Convert axis-angle to rotation matrix (batch)."""
        theta = torch.norm(r, dim=1, keepdim=True).unsqueeze(-1)
        r_hat = r / (torch.norm(r, dim=1, keepdim=True) + 1e-8)

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        K = torch.zeros(r.shape[0], 3, 3, device=r.device, dtype=r.dtype)
        K[:, 0, 1] = -r_hat[:, 2]
        K[:, 0, 2] = r_hat[:, 1]
        K[:, 1, 0] = r_hat[:, 2]
        K[:, 1, 2] = -r_hat[:, 0]
        K[:, 2, 0] = -r_hat[:, 1]
        K[:, 2, 1] = r_hat[:, 0]

        I = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)
        R = I + sin * K + (1 - cos) * torch.bmm(K, K)
        return R

    def forward(self, betas=None, pose=None, trans=None):
        """
        Run SMPL forward pass.
        Returns: vertices (6890, 3), joints (24, 3)
        """
        if betas is None:
            betas = torch.zeros(self.n_betas, device=self.device)
        if pose is None:
            pose = torch.zeros(self.n_joints * 3, device=self.device)
        if trans is None:
            trans = torch.zeros(3, device=self.device)

        # 1. Shape blending
        v_shaped = self.v_template + torch.einsum('ijk,k->ij', self.shapedirs, betas)

        # 2. Joint locations from shaped mesh
        J = torch.matmul(self.J_regressor, v_shaped)

        # 3. Pose blend shapes
        pose_params = pose.reshape(-1, 3)
        rot_mats = self.rodrigues(pose_params)

        ident = torch.eye(3, device=self.device).unsqueeze(0)
        pose_feature = (rot_mats[1:] - ident).reshape(-1)
        v_posed = v_shaped + torch.einsum('ijk,k->ij', self.posedirs, pose_feature)

        # 4. Kinematic chain
        G = [self._make_transform(rot_mats[0], J[0])]
        for i in range(1, self.n_joints):
            parent_idx = self.parent[i]
            local_t = self._make_transform(rot_mats[i], J[i] - J[parent_idx])
            G.append(torch.matmul(G[parent_idx], local_t))
        G = torch.stack(G)

        posed_joints = G[:, :3, 3].clone()

        # 5. Remove rest-pose offset for skinning
        J_homo = torch.cat([J, torch.zeros(self.n_joints, 1, device=self.device)], dim=1)
        G_offset = G.clone()
        for i in range(self.n_joints):
            rest_in_world = torch.matmul(G[i], J_homo[i])
            G_offset[i, :3, 3] = G[i, :3, 3] - rest_in_world[:3]

        # 6. Linear Blend Skinning
        G_flat = G_offset.reshape(self.n_joints, 16)
        T = torch.matmul(self.weights, G_flat).reshape(self.n_verts, 4, 4)

        v_homo = torch.cat([v_posed, torch.ones(self.n_verts, 1, device=self.device)], dim=1)
        v_final = torch.einsum('nij,nj->ni', T, v_homo)[:, :3]

        v_final = v_final + trans
        posed_joints = posed_joints + trans

        return v_final, posed_joints

    def _make_transform(self, R, t):
        T = torch.zeros(4, 4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = t
        T[3, 3] = 1.0
        return T

    def fit_to_joints(self, target_joints_3d, joint_indices, n_iters=500, lr=0.01):
        """Fit SMPL parameters to target 3D joints."""
        target = torch.tensor(target_joints_3d, dtype=torch.float32, device=self.device)

        betas = torch.zeros(self.n_betas, device=self.device, requires_grad=True)
        pose = torch.zeros(self.n_joints * 3, device=self.device, requires_grad=True)
        trans = torch.tensor(
            target_joints_3d.mean(axis=0), dtype=torch.float32, device=self.device,
            requires_grad=True
        )

        optimizer = torch.optim.Adam([
            {'params': [trans], 'lr': lr * 10},
            {'params': [betas], 'lr': lr},
            {'params': [pose], 'lr': lr * 0.5},
        ])

        print(f"\n  Fitting SMPL to {len(joint_indices)} joints ({n_iters} iters)...")

        best_loss = float('inf')
        best_params = None

        for i in range(n_iters):
            optimizer.zero_grad()

            verts, joints = self.forward(betas, pose, trans)

            pred_joints = joints[joint_indices]
            loss_joints = torch.mean((pred_joints - target) ** 2)

            # Light regularization
            loss_betas = 0.0001 * torch.sum(betas ** 2)
            loss_pose = 0.0001 * torch.sum(pose ** 2)

            loss = loss_joints + loss_betas + loss_pose

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = (
                    betas.detach().clone(),
                    pose.detach().clone(),
                    trans.detach().clone()
                )

            if (i + 1) % 100 == 0:
                joint_err = torch.sqrt(loss_joints).item()
                print(f"    Iter {i+1}/{n_iters}: loss={loss.item():.6f}, joint_RMSE={joint_err*1000:.1f}mm")

        betas_f, pose_f, trans_f = best_params
        with torch.no_grad():
            verts_final, joints_final = self.forward(betas_f, pose_f, trans_f)

        print(f"  Fitting complete. Final loss: {best_loss:.6f}")

        return (
            betas_f.cpu().numpy(),
            pose_f.cpu().numpy(),
            trans_f.cpu().numpy(),
            verts_final.cpu().numpy(),
            joints_final.cpu().numpy()
        )


# ============================================================
# Stereo Processing
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

        print("  Stereo rectification maps computed.")

    def rectify(self, img_left, img_right):
        rect_left = cv2.remap(img_left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right

    def triangulate_points(self, pts_left, pts_right):
        pts_4d = cv2.triangulatePoints(
            self.P1, self.P2,
            pts_left.T.astype(np.float64),
            pts_right.T.astype(np.float64)
        )
        pts_3d = (pts_4d[:3] / pts_4d[3]).T
        return pts_3d


# ============================================================
# Pose Detection (MediaPipe)
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
                mp_indices.append(f"mid({mp1},{mp2})")  # synthetic label
                smpl_indices.append(smpl_idx)

        if len(pts_left) == 0:
            return None, None, None, None

        return (
            np.array(pts_left),
            np.array(pts_right),
            mp_indices,
            smpl_indices
        )


# ============================================================
# Body Measurements
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


# ============================================================
# Visualization & Rendering
# ============================================================
def visualize_results(img_left, img_right, joints_3d, measurements,
                      landmarks_left=None, landmarks_right=None):
    """Create visualization of results."""
    h, w = img_left.shape[:2]

    vis_left = img_left.copy()
    vis_right = img_right.copy()

    if landmarks_left is not None:
        for mp_idx in MP_TO_SMPL.keys():
            x, y, vis = landmarks_left[mp_idx]
            if vis > 0.5:
                cv2.circle(vis_left, (int(x), int(y)), 4, (0, 255, 0), -1)

    if landmarks_right is not None:
        for mp_idx in MP_TO_SMPL.keys():
            x, y, vis = landmarks_right[mp_idx]
            if vis > 0.5:
                cv2.circle(vis_right, (int(x), int(y)), 4, (0, 255, 0), -1)

    stereo_vis = np.hstack([vis_left, vis_right])

    panel_w = max(w * 2, 600)
    panel_h = max(len(measurements) * 25 + 80, 200)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    cv2.putText(panel, "SMPL Body Measurements", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    cv2.line(panel, (20, 50), (panel_w - 20, 50), (100, 100, 100), 1)

    y_pos = 75
    for name, vals in measurements.items():
        text = f"{name}: {vals['cm']:.1f} cm ({vals['in']:.1f} in)"
        color = (0, 255, 200) if ('Height' in name or 'Span' in name) else (200, 200, 200)
        cv2.putText(panel, text, (30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 25

    scale = panel_w / stereo_vis.shape[1]
    stereo_vis = cv2.resize(stereo_vis, (panel_w, int(stereo_vis.shape[0] * scale)))

    combined = np.vstack([stereo_vis, panel])
    return combined


def create_3d_joint_visualization(joints_3d):
    """Create a simple 3D joint visualization (top and front views)."""
    canvas_size = 400
    canvas = np.zeros((canvas_size * 2, canvas_size * 2, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)

    j = joints_3d.copy()
    j -= j.mean(axis=0)
    max_range = np.max(np.abs(j)) * 1.3
    j = j / max_range * (canvas_size // 2 - 40) + canvas_size // 2

    # Front view (XY)
    cv2.putText(canvas, "Front (XY)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    for i, j_conn in SKELETON_CONNECTIONS:
        if i < len(j) and j_conn < len(j):
            pt1 = (int(j[i, 0]), int(j[i, 1]))
            pt2 = (int(j[j_conn, 0]), int(j[j_conn, 1]))
            cv2.line(canvas, pt1, pt2, (0, 180, 0), 2)
    for idx in range(len(j)):
        cv2.circle(canvas, (int(j[idx, 0]), int(j[idx, 1])), 4, (0, 0, 255), -1)

    # Side view (ZY)
    cv2.putText(canvas, "Side (ZY)", (canvas_size + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    for i, j_conn in SKELETON_CONNECTIONS:
        if i < len(j) and j_conn < len(j):
            pt1 = (int(j[i, 2]) + canvas_size, int(j[i, 1]))
            pt2 = (int(j[j_conn, 2]) + canvas_size, int(j[j_conn, 1]))
            cv2.line(canvas, pt1, pt2, (180, 0, 0), 2)
    for idx in range(len(j)):
        cv2.circle(canvas, (int(j[idx, 2]) + canvas_size, int(j[idx, 1])), 4, (0, 0, 255), -1)

    # Top view (XZ)
    cv2.putText(canvas, "Top (XZ)", (10, canvas_size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    for i, j_conn in SKELETON_CONNECTIONS:
        if i < len(j) and j_conn < len(j):
            pt1 = (int(j[i, 0]), int(j[i, 2]) + canvas_size)
            pt2 = (int(j[j_conn, 0]), int(j[j_conn, 2]) + canvas_size)
            cv2.line(canvas, pt1, pt2, (0, 0, 180), 2)
    for idx in range(len(j)):
        cv2.circle(canvas, (int(j[idx, 0]), int(j[idx, 2]) + canvas_size), 4, (0, 0, 255), -1)

    return canvas


def rotate_for_upright_view(verts_mm, center=None):
    """Rotate vertices so the body stands upright for rendering.
    
    Args:
        verts_mm: (N, 3) array of points
        center: (3,) center to subtract. If None, uses mean of verts_mm.
                Pass the SAME center for both verts and joints to keep them aligned.
    """
    v = verts_mm.copy().astype(np.float32)
    if center is None:
        center = v.mean(axis=0)
    v -= center
    x, y, z = v[:, 0].copy(), v[:, 1].copy(), v[:, 2].copy()
    v[:, 1] = z   # new Y = old Z (forward)
    v[:, 2] = y   # new Z = old Y (up)
    return v


def render_smpl_mesh(verts_mm, faces, img_size=(800, 800)):
    """Render SMPL mesh with pyrender (offscreen)."""
    import trimesh
    import pyrender

    verts = verts_mm.astype(np.float32) / 1000.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])

    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_node)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0.0, -2.0, 1.5]
    scene.add(camera, pose=cam_pose)

    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose1 = np.eye(4)
    light_pose1[:3, 3] = [0.0, -1.0, 2.0]
    scene.add(light1, pose=light_pose1)

    light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    light_pose2 = np.eye(4)
    light_pose2[:3, 3] = [1.0, -1.0, 1.0]
    scene.add(light2, pose=light_pose2)

    r = pyrender.OffscreenRenderer(img_size[0], img_size[1])
    color, _ = r.render(scene)
    r.delete()

    return color


def render_smpl_mesh_matplotlib(verts_mm, faces, img_size=(800, 800)):
    """Fallback renderer using matplotlib (no OpenGL)."""
    v = verts_mm / 1000.0

    fig = plt.figure(figsize=(img_size[0] / 100.0, img_size[1] / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    tris = v[faces]
    mesh = Poly3DCollection(tris, alpha=0.8)
    mesh.set_facecolor((0.7, 0.7, 0.9))
    mesh.set_edgecolor((0.1, 0.1, 0.1))
    ax.add_collection3d(mesh)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2.0
    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=15, azim=90)
    ax.axis("off")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(h, w, 3)
    plt.close(fig)
    return img


def render_smpl_mesh_annotated(verts_mm, faces, joints_mm, measurements, img_size=(1200, 1000)):
    """
    Render 3D SMPL mesh with joints, skeleton, lengths, and angles.
    Tries pyrender first; falls back to matplotlib if blank output detected.
    Annotations (joint names, lengths, angles) overlaid via cv2.
    """
    verts_m = verts_mm.astype(np.float64) / 1000.0
    joints_m = joints_mm.astype(np.float64) / 1000.0

    img = None
    joints_2d = None
    renderer_used = None

    # ============ Try pyrender ============
    try:
        import trimesh
        import pyrender

        scene = pyrender.Scene(bg_color=[255, 255, 255, 255],
                               ambient_light=[0.4, 0.4, 0.4, 1.0])

        # Body mesh (semi-transparent)
        body_mesh = trimesh.Trimesh(vertices=verts_m, faces=faces, process=False)
        body_mesh.fix_normals()
        body_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.65, 0.72, 0.88, 0.55],
            metallicFactor=0.1, roughnessFactor=0.6,
            alphaMode='BLEND'
        )
        scene.add(pyrender.Mesh.from_trimesh(body_mesh, material=body_material, smooth=True))

        # Joint spheres
        joint_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.9, 0.15, 0.15, 1.0],
            metallicFactor=0.3, roughnessFactor=0.4
        )
        for idx in range(len(joints_m)):
            sphere = trimesh.creation.uv_sphere(radius=0.015, count=[8, 8])
            sphere.apply_translation(joints_m[idx])
            scene.add(pyrender.Mesh.from_trimesh(sphere, material=joint_material, smooth=True))

        # Skeleton bones
        bone_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.95, 0.3, 0.3, 1.0],
            metallicFactor=0.2, roughnessFactor=0.5
        )
        for i_conn, j_conn in SKELETON_CONNECTIONS:
            if i_conn < len(joints_m) and j_conn < len(joints_m):
                p1, p2 = joints_m[i_conn], joints_m[j_conn]
                seg = p2 - p1
                length = np.linalg.norm(seg)
                if length < 1e-6:
                    continue
                cyl = trimesh.creation.cylinder(radius=0.006, height=length, sections=8)
                seg_dir = seg / length
                z_axis = np.array([0.0, 0.0, 1.0])
                cr = np.cross(z_axis, seg_dir)
                dt = np.dot(z_axis, seg_dir)
                if np.linalg.norm(cr) < 1e-6:
                    R = np.eye(3) if dt > 0 else np.diag([1.0, -1.0, -1.0])
                else:
                    cr_n = cr / np.linalg.norm(cr)
                    ang = np.arccos(np.clip(dt, -1, 1))
                    K = np.array([[0, -cr_n[2], cr_n[1]], [cr_n[2], 0, -cr_n[0]], [-cr_n[1], cr_n[0], 0]])
                    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = (p1 + p2) / 2.0
                cyl.apply_transform(T)
                scene.add(pyrender.Mesh.from_trimesh(cyl, material=bone_material, smooth=True))

        # Camera
        v_center = verts_m.mean(axis=0)
        v_extent = (verts_m.max(axis=0) - verts_m.min(axis=0)).max()
        yfov = np.pi / 6.0
        cam_dist = v_extent / (2.0 * np.tan(yfov / 2.0)) * 1.4
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=img_size[0] / img_size[1])
        cam_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        cam_t = v_center + np.array([0.0, -cam_dist, 0.0])
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = cam_R
        cam_pose[:3, 3] = cam_t
        scene.add(camera, pose=cam_pose)

        # Lights
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(dl, pose=cam_pose)

        r = pyrender.OffscreenRenderer(img_size[0], img_size[1])
        color_img, _ = r.render(scene)
        r.delete()

        # Check if the render is blank (all white = failed)
        non_white = np.mean(color_img < 250)
        if non_white > 0.05:  # at least 5% non-white pixels
            img = color_img[:, :, :3].copy() if color_img.shape[2] == 4 else color_img.copy()
            # Project joints to 2D
            fy = img_size[1] / (2.0 * np.tan(yfov / 2.0))
            fx = fy
            cx, cy_c = img_size[0] / 2.0, img_size[1] / 2.0
            cam_inv = np.linalg.inv(cam_pose)
            j_cam = (cam_inv[:3, :3] @ joints_m.T + cam_inv[:3, 3:4]).T
            joints_2d = []
            for jc in j_cam:
                px = int(fx * jc[0] / (-jc[2]) + cx) if abs(jc[2]) > 1e-6 else int(cx)
                py = int(fy * (-jc[1]) / (-jc[2]) + cy_c) if abs(jc[2]) > 1e-6 else int(cy_c)
                joints_2d.append((px, py))
            renderer_used = 'pyrender'
            print("    Using pyrender renderer")
        else:
            print("    pyrender produced blank image, falling back to matplotlib")

    except Exception as e:
        print(f"    pyrender error: {e}, falling back to matplotlib")

    # ============ Matplotlib fallback ============
    if img is None:
        from mpl_toolkits.mplot3d import proj3d

        v = verts_m
        j = joints_m

        fig = plt.figure(figsize=(img_size[0] / 100.0, img_size[1] / 100.0), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Lambert-shaded mesh
        tris = v[faces]
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        fn = np.cross(v1 - v0, v2 - v0)
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
        light = np.array([0.2, -0.8, 0.5])
        light = light / np.linalg.norm(light)
        diff = np.abs(np.dot(fn, light))
        intensity = np.clip(0.35 + 0.65 * diff, 0, 1)
        base_c = np.array([0.65, 0.72, 0.88])
        fc = np.outer(intensity, base_c)
        fc = np.clip(fc, 0, 1)
        fc_rgba = np.column_stack([fc, np.ones(len(fc)) * 0.6])  # semi-transparent

        sort_idx = np.argsort(tris[:, :, 1].mean(axis=1))
        mesh_coll = Poly3DCollection(tris[sort_idx])
        mesh_coll.set_facecolor(fc_rgba[sort_idx])
        mesh_coll.set_edgecolor('none')
        ax.add_collection3d(mesh_coll)

        # Skeleton in 3D (drawn on top of mesh)
        for i_c, j_c in SKELETON_CONNECTIONS:
            if i_c < len(j) and j_c < len(j):
                ax.plot([j[i_c,0], j[j_c,0]], [j[i_c,1], j[j_c,1]], [j[i_c,2], j[j_c,2]],
                        '-', color='#FF3333', linewidth=2.5, zorder=10)
        for idx in range(len(j)):
            ax.scatter(j[idx,0], j[idx,1], j[idx,2], c='#CC0000', s=50,
                       zorder=15, depthshade=False, edgecolors='white', linewidths=0.8)

        xv, yv, zv = v[:,0], v[:,1], v[:,2]
        mr = max(xv.max()-xv.min(), yv.max()-yv.min(), zv.max()-zv.min()) / 2.0 * 1.2
        mid = np.array([(xv.max()+xv.min())/2, (yv.max()+yv.min())/2, (zv.max()+zv.min())/2])
        ax.set_xlim(mid[0]-mr, mid[0]+mr)
        ax.set_ylim(mid[1]-mr, mid[1]+mr)
        ax.set_zlim(mid[2]-mr, mid[2]+mr)
        ax.view_init(elev=10, azim=75)
        ax.axis('off')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        fig.canvas.draw()

        # Project joints to 2D
        joints_2d = []
        for idx in range(len(j)):
            x2, y2, _ = proj3d.proj_transform(j[idx,0], j[idx,1], j[idx,2], ax.get_proj())
            xp, yp = ax.transData.transform((x2, y2))
            joints_2d.append((int(round(xp)), int(round(img_size[1] - yp))))

        w_f, h_f = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h_f, w_f, 3).copy()
        plt.close(fig)
        renderer_used = 'matplotlib'
        print("    Using matplotlib renderer")

    # ============ Overlay 2D annotations with cv2 ============
    key_joints = ['head', 'neck', 'l_shoulder', 'r_shoulder',
                  'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                  'pelvis', 'l_hip', 'r_hip',
                  'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    for idx in range(min(len(joints_2d), 24)):
        if SMPL_JOINT_NAMES[idx] in key_joints:
            pt = joints_2d[idx]
            cv2.putText(img, SMPL_JOINT_NAMES[idx],
                        (pt[0] + 12, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)

    seg_labels = [
        ("Shoulder Width", 16, 17), ("Left Upper Arm", 16, 18),
        ("Left Forearm", 18, 20), ("Right Upper Arm", 17, 19),
        ("Right Forearm", 19, 21), ("Left Thigh", 1, 4),
        ("Left Shin", 4, 7), ("Right Thigh", 2, 5),
        ("Right Shin", 5, 8), ("Torso (Pelvis to Neck)", 0, 12),
        ("Hip Width", 1, 2),
    ]
    for seg_name, j1, j2 in seg_labels:
        if seg_name in measurements:
            mx = (joints_2d[j1][0] + joints_2d[j2][0]) // 2
            my = (joints_2d[j1][1] + joints_2d[j2][1]) // 2
            label = f"{measurements[seg_name]['cm']:.1f}cm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (mx+5, my-th-4), (mx+tw+12, my+4), (255,255,255), -1)
            cv2.rectangle(img, (mx+5, my-th-4), (mx+tw+12, my+4), (180, 80, 0), 1)
            cv2.putText(img, label, (mx+8, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 80, 0), 1, cv2.LINE_AA)

    angle_defs = [
        ("L Elbow", 16, 18, 20), ("R Elbow", 17, 19, 21),
        ("L Knee", 1, 4, 7), ("R Knee", 2, 5, 8),
        ("L Shoulder", 12, 16, 18), ("R Shoulder", 12, 17, 19),
    ]
    for angle_name, ja, jb, jc in angle_defs:
        va_v = joints_m[ja] - joints_m[jb]
        vb_v = joints_m[jc] - joints_m[jb]
        cos_a = np.dot(va_v, vb_v) / (np.linalg.norm(va_v) * np.linalg.norm(vb_v) + 1e-8)
        angle_deg = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        pt = joints_2d[jb]
        label = f"{angle_name} {angle_deg:.0f} deg"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.rectangle(img, (pt[0]-tw-14, pt[1]+6), (pt[0]-6, pt[1]+th+12), (230,240,255), -1)
        cv2.rectangle(img, (pt[0]-tw-14, pt[1]+6), (pt[0]-6, pt[1]+th+12), (0,100,200), 1)
        cv2.putText(img, label, (pt[0]-tw-10, pt[1]+th+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 100, 200), 1, cv2.LINE_AA)

    if 'Total Height (chain)' in measurements:
        h_cm = measurements['Total Height (chain)']['cm']
        top_y = min(p[1] for p in joints_2d)
        bot_y = max(p[1] for p in joints_2d)
        rx = max(p[0] for p in joints_2d) + 40
        rx = min(rx, img.shape[1] - 140)
        cv2.line(img, (rx, top_y), (rx, bot_y), (0, 153, 0), 2, cv2.LINE_AA)
        cv2.line(img, (rx-8, top_y), (rx+8, top_y), (0, 153, 0), 2)
        cv2.line(img, (rx-8, bot_y), (rx+8, bot_y), (0, 153, 0), 2)
        label = f"Height: {h_cm:.1f}cm"
        mid_ht = (top_y + bot_y) // 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (rx+6, mid_ht-th-4), (rx+tw+14, mid_ht+6), (232,245,232), -1)
        cv2.rectangle(img, (rx+6, mid_ht-th-4), (rx+tw+14, mid_ht+6), (0,153,0), 1)
        cv2.putText(img, label, (rx+10, mid_ht+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA)

    cv2.putText(img, f"SMPL 3D Mesh with Joints, Lengths & Angles ({renderer_used})",
                (img.shape[1]//2 - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    return img


# ============================================================
# Main Pipeline
# ============================================================
def process_stereo_image(left_path, right_path, calib_path, smpl_path, output_dir, known_height=None):
    """Main processing pipeline."""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(left_path))[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---- Step 1: Load left and right images ----
    print(f"\n[1/7] Loading images")
    print(f"  Left:  {left_path}")
    print(f"  Right: {right_path}")
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    if img_left is None:
        raise FileNotFoundError(f"Cannot read left image: {left_path}")
    if img_right is None:
        raise FileNotFoundError(f"Cannot read right image: {right_path}")
    print(f"  Left size: {img_left.shape[1]}x{img_left.shape[0]}")
    print(f"  Right size: {img_right.shape[1]}x{img_right.shape[0]}")

    # ---- Step 2: Load stereo calibration ----
    print(f"\n[2/7] Loading stereo calibration")
    stereo = StereoProcessor(calib_path)

    # Resize images to match calibration resolution if needed
    calib_w, calib_h = stereo.img_size
    img_h, img_w = img_left.shape[:2]
    if img_w != calib_w or img_h != calib_h:
        print(f"  ⚠️  Image size ({img_w}x{img_h}) != calibration size ({calib_w}x{calib_h})")
        print(f"  Resizing images to match calibration...")
        img_left = cv2.resize(img_left, (calib_w, calib_h), interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_right, (calib_w, calib_h), interpolation=cv2.INTER_LINEAR)
        print(f"  Resized to: {img_left.shape}")

    # Rectify
    rect_left, rect_right = stereo.rectify(img_left, img_right)

    # ---- Step 3: Detect 2D pose ----
    print(f"\n[3/7] Detecting 2D poses with MediaPipe")
    detector = PoseDetector(min_confidence=0.3)

    detect_left = rect_left
    detect_right = rect_right
    scale_factor = 1.0
    min_detect_dim = 480
    if rect_left.shape[0] < min_detect_dim or rect_left.shape[1] < min_detect_dim:
        scale_factor = max(min_detect_dim / rect_left.shape[0], min_detect_dim / rect_left.shape[1])
        new_w = int(rect_left.shape[1] * scale_factor)
        new_h = int(rect_left.shape[0] * scale_factor)
        detect_left = cv2.resize(rect_left, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        detect_right = cv2.resize(rect_right, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"  Upscaled {scale_factor:.1f}x to {new_w}x{new_h} for pose detection")

    landmarks_left = detector.detect(detect_left)
    landmarks_right = detector.detect(detect_right)

    if landmarks_left is None:
        raise RuntimeError("No pose detected in LEFT image.")
    if landmarks_right is None:
        raise RuntimeError("No pose detected in RIGHT image.")

    if scale_factor != 1.0:
        landmarks_left[:, :2] /= scale_factor
        landmarks_right[:, :2] /= scale_factor

    pts_left, pts_right, mp_indices, smpl_indices = \
        detector.get_stereo_correspondences(landmarks_left, landmarks_right)

    if pts_left is None or len(pts_left) < 6:
        raise RuntimeError(f"Not enough matching landmarks (need 6+, got {len(pts_left) if pts_left is not None else 0})")

    print(f"  Matched landmarks: {len(pts_left)}")
    print(f"  SMPL joints used: {[SMPL_JOINT_NAMES[i] for i in smpl_indices]}")

    # ---- Step 4: Triangulate 3D joints ----
    print(f"\n[4/7] Triangulating 3D joints")
    joints_3d_stereo = stereo.triangulate_points(pts_left, pts_right)

    print(f"  Triangulated {len(joints_3d_stereo)} joints")
    for i, (mp_idx, smpl_idx) in enumerate(zip(mp_indices, smpl_indices)):
        j = joints_3d_stereo[i]
        print(f"    {SMPL_JOINT_NAMES[smpl_idx]:>15s}: X={j[0]:7.1f}  Y={j[1]:7.1f}  Z={j[2]:7.1f} mm")

    # ---- Step 5: Fit SMPL model ----
    print(f"\n[5/7] Fitting SMPL model")
    smpl = SMPLModel(smpl_path, device=device)

    # Convert mm→m and flip Y (camera Y-down → SMPL Y-up)
    joints_3d_meters = joints_3d_stereo / 1000.0
    joints_3d_meters[:, 1] = -joints_3d_meters[:, 1]
    print(f"  Converted mm→m and flipped Y (camera Y-down → SMPL Y-up)")

    betas, pose, trans, fitted_verts, fitted_joints = smpl.fit_to_joints(
        joints_3d_meters, smpl_indices,
        n_iters=1000, lr=0.02
    )

    # Convert back to mm
    fitted_verts = fitted_verts * 1000.0
    fitted_joints = fitted_joints * 1000.0

    print(f"\n  SMPL shape parameters (betas): {betas[:5].round(3)}...")
    print(f"  SMPL fitted {len(fitted_joints)} joints, {len(fitted_verts)} vertices")

    # Debug vertex bounds
    print(f"  Vertex bounds:")
    print(f"    X: {fitted_verts[:,0].min():.1f} → {fitted_verts[:,0].max():.1f} mm")
    print(f"    Y: {fitted_verts[:,1].min():.1f} → {fitted_verts[:,1].max():.1f} mm")
    print(f"    Z: {fitted_verts[:,2].min():.1f} → {fitted_verts[:,2].max():.1f} mm")

    # ---- Diagnostic: Compare raw triangulated vs SMPL fitted ----
    print(f"\n  === DIAGNOSTIC: Raw vs Fitted ===")

    raw_y_min = joints_3d_stereo[:, 1].min()
    raw_y_max = joints_3d_stereo[:, 1].max()
    raw_span = raw_y_max - raw_y_min
    print(f"  Raw triangulated Y span: {raw_span:.1f} mm ({raw_span/10:.1f} cm)")

    fit_y_min = fitted_joints[:, 1].min()
    fit_y_max = fitted_joints[:, 1].max()
    fit_span = fit_y_max - fit_y_min
    print(f"  SMPL fitted Y span (24 joints): {fit_span:.1f} mm ({fit_span/10:.1f} cm)")

    # Per-joint comparison
    print(f"\n  Per-joint comparison (observed vs fitted):")
    print(f"  {'Joint':>15s}  {'Raw Y':>8s}  {'Fit Y':>8s}  {'Diff':>8s}")
    total_y_err = 0
    for i, smpl_idx in enumerate(smpl_indices):
        raw_y = joints_3d_stereo[i, 1]
        fit_y = fitted_joints[smpl_idx, 1]
        diff = fit_y - raw_y
        total_y_err += abs(diff)
        print(f"  {SMPL_JOINT_NAMES[smpl_idx]:>15s}  {raw_y:8.1f}  {fit_y:8.1f}  {diff:+8.1f}")
    print(f"  {'Mean |error|':>15s}  {'':>8s}  {'':>8s}  {total_y_err/len(smpl_indices):8.1f}")

    # Full skeleton Y positions
    print(f"\n  Full SMPL skeleton (sorted by Y):")
    joint_y = [(SMPL_JOINT_NAMES[i], fitted_joints[i, 1]) for i in range(24)]
    joint_y.sort(key=lambda x: x[1])
    observed_names = [SMPL_JOINT_NAMES[si] for si in smpl_indices]
    for name, y in joint_y:
        marker = " ◀ observed" if name in observed_names else ""
        print(f"    {name:>15s}: Y = {y:8.1f} mm{marker}")

    # Height chain breakdown
    print(f"\n  Height chain (ankle→head):")
    chain_total = 0
    for i in range(len(HEIGHT_PATH) - 1):
        j1_idx = SMPL_JOINT_NAMES.index(HEIGHT_PATH[i])
        j2_idx = SMPL_JOINT_NAMES.index(HEIGHT_PATH[i+1])
        seg_len = np.linalg.norm(fitted_joints[j1_idx] - fitted_joints[j2_idx])
        chain_total += seg_len
        print(f"    {HEIGHT_PATH[i]:>10s} → {HEIGHT_PATH[i+1]:<10s}: {seg_len/10:6.1f} cm")
    print(f"    {'TOTAL':>23s}: {chain_total/10:6.1f} cm")
    print(f"  ===================================")

    # Scale to known height if provided
    if known_height:
        all_y = fitted_joints[:, 1]
        raw_height_mm = float(np.max(all_y) - np.min(all_y))
        raw_height_cm = raw_height_mm / 10.0
        scale_factor = known_height / raw_height_cm
        print(f"\n  Height scaling:")
        print(f"    Raw fitted height: {raw_height_cm:.1f} cm")
        print(f"    Target height: {known_height:.1f} cm")
        print(f"    Scale factor: {scale_factor:.4f}")
        fitted_joints = fitted_joints * scale_factor
        fitted_verts = fitted_verts * scale_factor

    # ---- Step 6: Compute measurements (before rendering so we can annotate) ----
    print(f"\n[6/7] Computing body measurements")
    measurer = BodyMeasurements(fitted_joints, fitted_verts)
    measurements = measurer.compute_all()

    print(f"\n{'='*60}")
    print(f"  BODY MEASUREMENTS")
    print(f"{'='*60}")
    for name, vals in measurements.items():
        print(f"  {name:.<40s} {vals['cm']:>7.1f} cm  ({vals['in']:>5.1f} in)")
    print(f"{'='*60}")

    # ---- Step 7: Render SMPL mesh with annotations ----
    print(f"\n[7/7] Rendering SMPL mesh with joints, lengths & angles")
    # Use the same center for both so they stay aligned
    shared_center = fitted_verts.mean(axis=0)
    verts_upright = rotate_for_upright_view(fitted_verts, center=shared_center)
    joints_upright = rotate_for_upright_view(fitted_joints, center=shared_center)

    # Annotated mesh render (joints + lengths + angles)
    annotated_img = render_smpl_mesh_annotated(
        verts_upright, smpl.faces, joints_upright, measurements, img_size=(1200, 1000)
    )
    annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
    annotated_path = os.path.join(output_dir, f"{base_name}_smpl_annotated_{timestamp}.png")
    cv2.imwrite(annotated_path, annotated_bgr)
    print(f"  Saved annotated mesh: {annotated_path}")

    # Plain mesh render (for clean view)
    try:
        mesh_img = render_smpl_mesh(verts_upright, smpl.faces, img_size=(800, 800))
        if mesh_img.shape[2] == 4:
            mesh_img_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGBA2BGR)
        else:
            mesh_img_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  pyrender failed ({e}), using matplotlib.")
        mesh_img = render_smpl_mesh_matplotlib(verts_upright, smpl.faces, img_size=(800, 800))
        mesh_img_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)

    mesh_vis_path = os.path.join(output_dir, f"{base_name}_smpl_mesh_{timestamp}.png")
    cv2.imwrite(mesh_vis_path, mesh_img_bgr)
    print(f"  Saved plain mesh: {mesh_vis_path}")

    # ---- Save results ----
    csv_path = os.path.join(output_dir, f"{base_name}_measurements_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Measurement', 'mm', 'cm', 'inches'])
        for name, vals in measurements.items():
            writer.writerow([name, f"{vals['mm']:.1f}", f"{vals['cm']:.1f}", f"{vals['in']:.1f}"])
    print(f"\n  Saved: {csv_path}")

    joints_path = os.path.join(output_dir, f"{base_name}_joints3d_{timestamp}.npy")
    np.save(joints_path, fitted_joints)
    print(f"  Saved: {joints_path}")

    params_path = os.path.join(output_dir, f"{base_name}_smpl_params_{timestamp}.npz")
    np.savez(params_path, betas=betas, pose=pose, trans=trans)
    print(f"  Saved: {params_path}")

    vis = visualize_results(rect_left, rect_right, fitted_joints, measurements,
                           landmarks_left, landmarks_right)
    vis_path = os.path.join(output_dir, f"{base_name}_visualization_{timestamp}.png")
    cv2.imwrite(vis_path, vis)
    print(f"  Saved: {vis_path}")

    joint_vis = create_3d_joint_visualization(fitted_joints)
    jvis_path = os.path.join(output_dir, f"{base_name}_3d_joints_{timestamp}.png")
    cv2.imwrite(jvis_path, joint_vis)
    print(f"  Saved: {jvis_path}")

    mesh_path = os.path.join(output_dir, f"{base_name}_mesh_{timestamp}.obj")
    with open(mesh_path, 'w') as f:
        for v in fitted_verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in smpl.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  Saved: {mesh_path}")

    print(f"\n✓ All results saved to {output_dir}/")
    print(f"✓ Processing complete!")

    return measurements, fitted_joints, fitted_verts


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SMPL-based Stereo Body Measurement System"
    )
    parser.add_argument('--left', required=True,
                       help='Path to left camera image')
    parser.add_argument('--right', required=True,
                       help='Path to right camera image')
    parser.add_argument('--calib', default=DEFAULT_CALIB_PATH,
                       help=f'Stereo calibration file (default: {DEFAULT_CALIB_PATH})')
    parser.add_argument('--smpl', default=SMPL_MODEL_PATH,
                       help='Path to SMPL model .pkl file')
    parser.add_argument('--output', default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--height', type=float, default=None,
                       help='Known height in cm (e.g. 167). Scales all measurements to match.')

    args = parser.parse_args()

    if not os.path.exists(args.left):
        print(f"ERROR: Left image not found: {args.left}")
        sys.exit(1)
    if not os.path.exists(args.right):
        print(f"ERROR: Right image not found: {args.right}")
        sys.exit(1)
    if not os.path.exists(args.calib):
        print(f"ERROR: Calibration file not found: {args.calib}")
        sys.exit(1)
    if not os.path.exists(args.smpl):
        print(f"ERROR: SMPL model not found: {args.smpl}")
        print(f"  Expected at: {args.smpl}")
        sys.exit(1)

    process_stereo_image(args.left, args.right, args.calib, args.smpl, args.output, args.height)


if __name__ == '__main__':
    main()
