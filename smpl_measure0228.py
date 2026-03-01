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
    python smpl_stereo_measure.py --image <stereo_image.png>
    python smpl_stereo_measure.py --image <stereo_image.png> --calib stereo_params.npz
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
# MediaPipe landmark index -> SMPL joint index
MP_TO_SMPL = {
    11: 16,  # left_shoulder -> l_shoulder
    12: 17,  # right_shoulder -> r_shoulder
    13: 18,  # left_elbow -> l_elbow
    14: 19,  # right_elbow -> r_elbow
    15: 20,  # left_wrist -> l_wrist
    16: 21,  # right_wrist -> r_wrist
    23: 1,   # left_hip -> l_hip
    24: 2,   # right_hip -> r_hip
    25: 4,   # left_knee -> l_knee
    26: 5,   # right_knee -> r_knee
    27: 7,   # left_ankle -> l_ankle
    28: 8,   # right_ankle -> r_ankle
}

# Measurement definitions: (name, joint_a, joint_b) or (name, [path of joints])
SEGMENT_MEASUREMENTS = [
    # Upper body
    ("Shoulder Width", "l_shoulder", "r_shoulder"),
    ("Left Upper Arm", "l_shoulder", "l_elbow"),
    ("Left Forearm", "l_elbow", "l_wrist"),
    ("Left Full Arm", "l_shoulder", "l_wrist"),
    ("Right Upper Arm", "r_shoulder", "r_elbow"),
    ("Right Forearm", "r_elbow", "r_wrist"),
    ("Right Full Arm", "r_shoulder", "r_wrist"),
    ("Neck to Head", "neck", "head"),
    # Torso
    ("Torso (Pelvis to Neck)", "pelvis", "neck"),
    ("Spine (Pelvis to Spine3)", "pelvis", "spine3"),
    ("Hip Width", "l_hip", "r_hip"),
    # Lower body
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
    (0, 1), (0, 2), (0, 3),        # pelvis -> hips, spine
    (1, 4), (4, 7), (7, 10),       # left leg
    (2, 5), (5, 8), (8, 11),       # right leg
    (3, 6), (6, 9), (9, 12),       # spine
    (12, 13), (12, 14),            # neck -> collars
    (13, 16), (16, 18), (18, 20), (20, 22),  # left arm
    (14, 17), (17, 19), (19, 21), (21, 23),  # right arm
    (12, 15),                       # neck -> head
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
            """Dummy that absorbs any chumpy constructor args."""
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
        # Create a fake chumpy module so pickle can resolve it
        fake_ch = types.ModuleType('chumpy')
        fake_ch.Ch = _ChumpyStub
        fake_ch.array = _ChumpyStub
        fake_ch_linalg = types.ModuleType('chumpy.linalg')
        fake_ch_utils = types.ModuleType('chumpy.utils')
        sys.modules['chumpy'] = fake_ch
        sys.modules['chumpy.ch'] = fake_ch
        sys.modules['chumpy.linalg'] = fake_ch_linalg
        sys.modules['chumpy.utils'] = fake_ch_utils
        # Point any chumpy class reference to our stub
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
        
        self.kintree_table = np.array(model_data['kintree_table']).astype(np.int64)  # (2, 24)
        
        self.faces = np.array(model_data['f']).astype(np.int32)  # (13776, 3)
        
        self.n_joints = 24
        self.n_verts = 6890
        self.n_betas = 10
        
        # Parent indices for kinematic tree
        self.parent = {
            i: int(self.kintree_table[0, i])
            for i in range(1, self.kintree_table.shape[1])
        }
        
        print(f"  Vertices: {self.n_verts}, Joints: {self.n_joints}, Faces: {len(self.faces)}")
        print("  SMPL model loaded successfully.")
    
    def rodrigues(self, r):
        """Convert axis-angle to rotation matrix (batch)."""
        # r: (N, 3)
        theta = torch.norm(r, dim=1, keepdim=True).unsqueeze(-1)  # (N, 1, 1)
        r_hat = r / (torch.norm(r, dim=1, keepdim=True) + 1e-8)  # (N, 3)
        
        cos = torch.cos(theta)  # (N, 1, 1)
        sin = torch.sin(theta)  # (N, 1, 1)
        
        # Skew-symmetric matrix
        K = torch.zeros(r.shape[0], 3, 3, device=r.device, dtype=r.dtype)
        K[:, 0, 1] = -r_hat[:, 2]
        K[:, 0, 2] = r_hat[:, 1]
        K[:, 1, 0] = r_hat[:, 2]
        K[:, 1, 2] = -r_hat[:, 0]
        K[:, 2, 0] = -r_hat[:, 1]
        K[:, 2, 1] = r_hat[:, 0]
        
        I = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)
        R = I + sin * K + (1 - cos) * torch.bmm(K, K)
        
        return R  # (N, 3, 3)
    
    def forward(self, betas=None, pose=None, trans=None):
        """
        Run SMPL forward pass.
        
        Args:
            betas: (10,) shape parameters
            pose: (72,) pose parameters (24 joints * 3 axis-angle)
            trans: (3,) global translation
            
        Returns:
            vertices: (6890, 3)
            joints: (24, 3)
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
        J = torch.matmul(self.J_regressor, v_shaped)  # (24, 3)
        
        # 3. Pose blend shapes
        pose_params = pose.reshape(-1, 3)  # (24, 3)
        rot_mats = self.rodrigues(pose_params)  # (24, 3, 3)
        
        # Pose blend shapes (exclude global rotation)
        ident = torch.eye(3, device=self.device).unsqueeze(0)
        pose_feature = (rot_mats[1:] - ident).reshape(-1)  # (207,)
        v_posed = v_shaped + torch.einsum('ijk,k->ij', self.posedirs, pose_feature)
        
        # 4. Build kinematic chain - world transforms for each joint
        # Root
        G = [self._make_transform(rot_mats[0], J[0])]
        
        for i in range(1, self.n_joints):
            parent_idx = self.parent[i]
            local_t = self._make_transform(rot_mats[i], J[i] - J[parent_idx])
            G.append(torch.matmul(G[parent_idx], local_t))
        
        G = torch.stack(G)  # (24, 4, 4)
        
        # Get posed joint locations
        posed_joints = G[:, :3, 3].clone()
        
        # 5. Remove rest-pose offset for skinning
        # Each joint transform should move vertices relative to rest pose
        J_homo = torch.cat([J, torch.zeros(self.n_joints, 1, device=self.device)], dim=1)  # (24, 4)
        G_offset = G.clone()
        for i in range(self.n_joints):
            # Subtract the transformed rest joint position
            rest_in_world = torch.matmul(G[i], J_homo[i])  # (4,)
            G_offset[i, :3, 3] = G[i, :3, 3] - rest_in_world[:3]
        
        # 6. Linear Blend Skinning
        # weights: (6890, 24), G_offset: (24, 4, 4)
        # Compute per-vertex transform: T = sum_j(w_j * G_offset_j)
        G_flat = G_offset.reshape(self.n_joints, 16)  # (24, 16)
        T = torch.matmul(self.weights, G_flat).reshape(self.n_verts, 4, 4)  # (6890, 4, 4)
        
        # Apply transforms to posed vertices
        v_homo = torch.cat([v_posed, torch.ones(self.n_verts, 1, device=self.device)], dim=1)  # (6890, 4)
        v_final = torch.einsum('nij,nj->ni', T, v_homo)[:, :3]  # (6890, 3)
        
        # Apply translation
        v_final = v_final + trans
        posed_joints = posed_joints + trans
        
        return v_final, posed_joints
    
    def _make_transform(self, R, t):
        """Create 4x4 transformation matrix."""
        T = torch.zeros(4, 4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = t
        T[3, 3] = 1.0
        return T
    
    def fit_to_joints(self, target_joints_3d, joint_indices, n_iters=500, lr=0.01):
        """
        Fit SMPL parameters to target 3D joints.
        
        Args:
            target_joints_3d: (N, 3) target 3D joint positions in mm
            joint_indices: list of SMPL joint indices corresponding to target joints
            n_iters: optimization iterations
            lr: learning rate
            
        Returns:
            betas, pose, trans, fitted_vertices, fitted_joints
        """
        target = torch.tensor(target_joints_3d, dtype=torch.float32, device=self.device)
        
        # Initialize parameters
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
            
            # Joint matching loss
            pred_joints = joints[joint_indices]
            loss_joints = torch.mean((pred_joints - target) ** 2)
            
            # Regularization (light - allow shape to vary freely)
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
        
        # Get final result
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
        
        # Compute rectification transforms
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.dist1, self.K2, self.dist2,
            self.img_size, self.R, self.T,
            alpha=0
        )
        
        # Compute undistortion maps
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K1, self.dist1, self.R1, self.P1,
            self.img_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.K2, self.dist2, self.R2, self.P2,
            self.img_size, cv2.CV_32FC1
        )
        
        print("  Stereo rectification maps computed.")
    
    def split_image(self, stereo_img):
        """Split side-by-side stereo image into left and right."""
        h, w = stereo_img.shape[:2]
        mid = w // 2
        return stereo_img[:, :mid], stereo_img[:, mid:]
    
    def rectify(self, img_left, img_right):
        """Apply stereo rectification."""
        rect_left = cv2.remap(img_left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(img_right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rect_left, rect_right
    
    def triangulate_points(self, pts_left, pts_right):
        """
        Triangulate 3D points from corresponding 2D points.
        
        Args:
            pts_left: (N, 2) points in left rectified image
            pts_right: (N, 2) points in right rectified image
            
        Returns:
            pts_3d: (N, 3) 3D points in mm
        """
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
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        print("  MediaPipe pose detector initialized.")
    
    def detect(self, image):
        """
        Detect pose landmarks in image.
        
        Returns:
            landmarks: (33, 3) array of (x, y, visibility) or None
        """
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
        """
        Extract matching 2D points for stereo triangulation.
        Only uses landmarks that are visible in both views.
        
        Returns:
            pts_left: (N, 2)
            pts_right: (N, 2)
            mp_indices: list of MediaPipe landmark indices
            smpl_indices: list of corresponding SMPL joint indices
        """
        pts_left = []
        pts_right = []
        mp_indices = []
        smpl_indices = []
        
        min_vis = 0.5
        
        for mp_idx, smpl_idx in MP_TO_SMPL.items():
            vis_L = landmarks_left[mp_idx, 2]
            vis_R = landmarks_right[mp_idx, 2]
            
            if vis_L >= min_vis and vis_R >= min_vis:
                pts_left.append(landmarks_left[mp_idx, :2])
                pts_right.append(landmarks_right[mp_idx, :2])
                mp_indices.append(mp_idx)
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
        """
        Args:
            joints_3d: (24, 3) SMPL joint positions in mm
            vertices_3d: (6890, 3) SMPL mesh vertices in mm (optional)
        """
        self.joints = joints_3d
        self.vertices = vertices_3d
        self.measurements = {}
    
    def compute_all(self):
        """Compute all available measurements."""
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
        """Euclidean distance between two joints (mm)."""
        i1 = self._joint_idx(j1) if isinstance(j1, str) else j1
        i2 = self._joint_idx(j2) if isinstance(j2, str) else j2
        return float(np.linalg.norm(self.joints[i1] - self.joints[i2]))
    
    def _compute_segment_lengths(self):
        """Compute all segment (limb) lengths."""
        for name, j1, j2 in SEGMENT_MEASUREMENTS:
            length_mm = self._dist(j1, j2)
            self.measurements[name] = {
                'mm': length_mm,
                'cm': length_mm / 10.0,
                'in': length_mm / 25.4,
            }
    
    def _compute_height(self):
        """Compute total height along the body chain."""
        total = 0.0
        for i in range(len(HEIGHT_PATH) - 1):
            total += self._dist(HEIGHT_PATH[i], HEIGHT_PATH[i+1])
        
        self.measurements['Total Height (chain)'] = {
            'mm': total,
            'cm': total / 10.0,
            'in': total / 25.4,
        }
        
        # Also compute direct vertical span
        all_y = self.joints[:, 1]  # Y coordinates
        vertical_span = float(np.max(all_y) - np.min(all_y))
        self.measurements['Vertical Span'] = {
            'mm': vertical_span,
            'cm': vertical_span / 10.0,
            'in': vertical_span / 25.4,
        }
    
    def _compute_arm_span(self):
        """Compute full arm span (fingertip to fingertip)."""
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
            'mm': arm_span,
            'cm': arm_span / 10.0,
            'in': arm_span / 25.4,
        }
    
    def _compute_inseam(self):
        """Compute inseam (crotch to ankle)."""
        # Average of left and right
        left_inseam = self._dist("l_hip", "l_knee") + self._dist("l_knee", "l_ankle")
        right_inseam = self._dist("r_hip", "r_knee") + self._dist("r_knee", "r_ankle")
        avg_inseam = (left_inseam + right_inseam) / 2.0
        
        self.measurements['Inseam (avg)'] = {
            'mm': avg_inseam,
            'cm': avg_inseam / 10.0,
            'in': avg_inseam / 25.4,
        }
    
    def _compute_circumferences(self):
        """Compute body circumferences from mesh vertices."""
        if self.vertices is None:
            return
        
        # Approximate circumferences by slicing the mesh at specific heights
        # These vertex index groups are standard SMPL body regions
        
        # Chest: vertices around spine2/spine3 height
        chest_y = self.joints[self._joint_idx("spine2"), 1]
        chest_verts = self.vertices[np.abs(self.vertices[:, 1] - chest_y) < 20]
        if len(chest_verts) > 10:
            circumference = self._estimate_circumference(chest_verts)
            self.measurements['Chest Circumference'] = {
                'mm': circumference,
                'cm': circumference / 10.0,
                'in': circumference / 25.4,
            }
        
        # Waist: vertices around pelvis/spine1 height
        waist_y = (self.joints[self._joint_idx("pelvis"), 1] + 
                   self.joints[self._joint_idx("spine1"), 1]) / 2
        waist_verts = self.vertices[np.abs(self.vertices[:, 1] - waist_y) < 20]
        if len(waist_verts) > 10:
            circumference = self._estimate_circumference(waist_verts)
            self.measurements['Waist Circumference'] = {
                'mm': circumference,
                'cm': circumference / 10.0,
                'in': circumference / 25.4,
            }
        
        # Hip: vertices at hip height
        hip_y = (self.joints[self._joint_idx("l_hip"), 1] + 
                 self.joints[self._joint_idx("r_hip"), 1]) / 2
        hip_verts = self.vertices[np.abs(self.vertices[:, 1] - hip_y) < 20]
        if len(hip_verts) > 10:
            circumference = self._estimate_circumference(hip_verts)
            self.measurements['Hip Circumference'] = {
                'mm': circumference,
                'cm': circumference / 10.0,
                'in': circumference / 25.4,
            }
    
    def _estimate_circumference(self, verts_slice):
        """
        Estimate circumference from a slice of vertices.
        Projects to XZ plane and computes convex hull perimeter.
        """
        # Project onto XZ plane (cross-section)
        pts_2d = verts_slice[:, [0, 2]]  # X and Z
        
        if len(pts_2d) < 3:
            return 0.0
        
        # Compute convex hull
        pts_2d_f = pts_2d.astype(np.float32)
        hull = cv2.convexHull(pts_2d_f)
        
        # Compute perimeter
        perimeter = cv2.arcLength(hull, closed=True)
        return float(perimeter)


# ============================================================
# Visualization
# ============================================================
def visualize_results(img_left, img_right, joints_3d, measurements, landmarks_left=None, landmarks_right=None):
    """Create visualization of results."""
    import mediapipe as mp
    
    h, w = img_left.shape[:2]
    
    # Draw landmarks on images
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
    
    # Side by side stereo view
    stereo_vis = np.hstack([vis_left, vis_right])
    
    # Create measurement text panel
    panel_w = max(w * 2, 600)
    panel_h = max(len(measurements) * 25 + 80, 200)
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # dark background
    
    # Title
    cv2.putText(panel, "SMPL Body Measurements", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    cv2.line(panel, (20, 50), (panel_w - 20, 50), (100, 100, 100), 1)
    
    # Measurements
    y_pos = 75
    for name, vals in measurements.items():
        text = f"{name}: {vals['cm']:.1f} cm ({vals['in']:.1f} in)"
        color = (200, 200, 200)
        if 'Height' in name or 'Span' in name:
            color = (0, 255, 200)  # highlight
        cv2.putText(panel, text, (30, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 25
    
    # Combine
    # Resize stereo view to match panel width
    scale = panel_w / stereo_vis.shape[1]
    stereo_vis = cv2.resize(stereo_vis, (panel_w, int(stereo_vis.shape[0] * scale)))
    
    combined = np.vstack([stereo_vis, panel])
    
    return combined


def create_3d_joint_visualization(joints_3d):
    """Create a simple 3D joint visualization (top and front views)."""
    canvas_size = 400
    canvas = np.zeros((canvas_size * 2, canvas_size * 2, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)
    
    # Normalize joints to canvas
    j = joints_3d.copy()
    j -= j.mean(axis=0)
    max_range = np.max(np.abs(j)) * 1.3
    j = j / max_range * (canvas_size // 2 - 40) + canvas_size // 2
    
    # Front view (XY) - top left
    cv2.putText(canvas, "Front (XY)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    for i, j_conn in SKELETON_CONNECTIONS:
        if i < len(j) and j_conn < len(j):
            pt1 = (int(j[i, 0]), int(j[i, 1]))
            pt2 = (int(j[j_conn, 0]), int(j[j_conn, 1]))
            cv2.line(canvas, pt1, pt2, (0, 180, 0), 2)
    for idx in range(len(j)):
        pt = (int(j[idx, 0]), int(j[idx, 1]))
        cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
    
    # Side view (ZY) - top right
    cv2.putText(canvas, "Side (ZY)", (canvas_size + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    for i, j_conn in SKELETON_CONNECTIONS:
        if i < len(j) and j_conn < len(j):
            pt1 = (int(j[i, 2]) + canvas_size, int(j[i, 1]))
            pt2 = (int(j[j_conn, 2]) + canvas_size, int(j[j_conn, 1]))
            cv2.line(canvas, pt1, pt2, (180, 0, 0), 2)
    for idx in range(len(j)):
        pt = (int(j[idx, 2]) + canvas_size, int(j[idx, 1]))
        cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
    
    # Top view (XZ) - bottom left
    cv2.putText(canvas, "Top (XZ)", (10, canvas_size + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    for i, j_conn in SKELETON_CONNECTIONS:
        if i < len(j) and j_conn < len(j):
            pt1 = (int(j[i, 0]), int(j[i, 2]) + canvas_size)
            pt2 = (int(j[j_conn, 0]), int(j[j_conn, 2]) + canvas_size)
            cv2.line(canvas, pt1, pt2, (0, 0, 180), 2)
    for idx in range(len(j)):
        pt = (int(j[idx, 0]), int(j[idx, 2]) + canvas_size)
        cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
    
    return canvas


# ============================================================
# Main Pipeline
# ============================================================
def process_stereo_image(left_path, right_path, calib_path, smpl_path, output_dir, known_height=None):
    """Main processing pipeline.
    
    Args:
        known_height: If provided, scale results to match this height in cm.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ---- Step 1: Load left and right images ----
    print(f"\n[1/6] Loading images")
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
    print(f"\n[2/6] Loading stereo calibration")
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
    print(f"\n[3/6] Detecting 2D poses with MediaPipe")
    detector = PoseDetector(min_confidence=0.3)
    
    # Upscale images for better pose detection if they are small
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
        raise RuntimeError("No pose detected in LEFT image. Is a person visible?")
    if landmarks_right is None:
        raise RuntimeError("No pose detected in RIGHT image. Is a person visible?")
    
    # Scale landmarks back to calibration resolution
    if scale_factor != 1.0:
        landmarks_left[:, :2] /= scale_factor
        landmarks_right[:, :2] /= scale_factor
        print(f"  Landmarks scaled back to calibration resolution")
    
    # Get corresponding points
    pts_left, pts_right, mp_indices, smpl_indices = \
        detector.get_stereo_correspondences(landmarks_left, landmarks_right)
    
    if pts_left is None or len(pts_left) < 6:
        raise RuntimeError(f"Not enough matching landmarks (need 6+, got {len(pts_left) if pts_left is not None else 0})")
    
    print(f"  Matched landmarks: {len(pts_left)}")
    print(f"  SMPL joints used: {[SMPL_JOINT_NAMES[i] for i in smpl_indices]}")
    
    # ---- Step 4: Triangulate 3D joints ----
    print(f"\n[4/6] Triangulating 3D joints")
    joints_3d_stereo = stereo.triangulate_points(pts_left, pts_right)
    
    print(f"  Triangulated {len(joints_3d_stereo)} joints")
    for i, (mp_idx, smpl_idx) in enumerate(zip(mp_indices, smpl_indices)):
        j = joints_3d_stereo[i]
        print(f"    {SMPL_JOINT_NAMES[smpl_idx]:>15s}: X={j[0]:7.1f}  Y={j[1]:7.1f}  Z={j[2]:7.1f} mm")
    
    # ---- Step 5: Fit SMPL model ----
    print(f"\n[5/6] Fitting SMPL model")
    smpl = SMPLModel(smpl_path, device=device)
    
    # SMPL template is in meters, triangulated joints are in mm -> convert
    joints_3d_meters = joints_3d_stereo / 1000.0
    
    # CRITICAL: Stereo triangulation uses Y-down (camera convention)
    # but SMPL uses Y-up. Flip Y axis to match SMPL.
    joints_3d_meters[:, 1] = -joints_3d_meters[:, 1]
    print(f"  Converted mm→m and flipped Y (camera Y-down → SMPL Y-up)")
    
    betas, pose, trans, fitted_verts, fitted_joints = smpl.fit_to_joints(
        joints_3d_meters, smpl_indices,
        n_iters=1000, lr=0.02
    )
    
    # Convert results back to mm for measurements
    # (keep SMPL Y-up, measurements don't depend on sign)
    fitted_verts = fitted_verts * 1000.0
    fitted_joints = fitted_joints * 1000.0
    
    # ---- Diagnostic: Compare raw triangulated vs SMPL fitted ----
    print(f"\n  === DIAGNOSTIC: Raw vs Fitted ===")
    
    # Raw triangulated vertical span
    raw_y_min = joints_3d_stereo[:, 1].min()
    raw_y_max = joints_3d_stereo[:, 1].max()
    raw_span = raw_y_max - raw_y_min
    print(f"  Raw triangulated Y span: {raw_span:.1f} mm ({raw_span/10:.1f} cm)")
    print(f"    Y range: {raw_y_min:.1f} to {raw_y_max:.1f} mm")
    
    # SMPL fitted vertical span (all 24 joints)
    fit_y_min = fitted_joints[:, 1].min()
    fit_y_max = fitted_joints[:, 1].max()
    fit_span = fit_y_max - fit_y_min
    print(f"  SMPL fitted Y span (24 joints): {fit_span:.1f} mm ({fit_span/10:.1f} cm)")
    print(f"    Y range: {fit_y_min:.1f} to {fit_y_max:.1f} mm")
    
    # Per-joint comparison for the 12 observed joints
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
    
    # Show ALL 24 SMPL joints Y positions to see the height chain
    print(f"\n  Full SMPL skeleton (sorted by Y):")
    joint_y = [(SMPL_JOINT_NAMES[i], fitted_joints[i, 1]) for i in range(24)]
    joint_y.sort(key=lambda x: x[1])
    for name, y in joint_y:
        marker = " ◀ observed" if name in [SMPL_JOINT_NAMES[si] for si in smpl_indices] else ""
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
    scale_factor = 1.0
    if known_height:
        # Compute raw height from fitted joints
        all_y = fitted_joints[:, 1]
        raw_height_mm = float(np.max(all_y) - np.min(all_y))
        raw_height_cm = raw_height_mm / 10.0
        scale_factor = known_height / raw_height_cm
        print(f"  Raw fitted height: {raw_height_cm:.1f} cm")
        print(f"  Target height: {known_height:.1f} cm")
        print(f"  Scale factor: {scale_factor:.4f}")
        fitted_joints = fitted_joints * scale_factor
        fitted_verts = fitted_verts * scale_factor
    
    print(f"\n  SMPL shape parameters (betas): {betas[:5].round(3)}...")
    print(f"  SMPL fitted {len(fitted_joints)} joints, {len(fitted_verts)} vertices")
    
    # ---- Step 6: Compute measurements ----
    print(f"\n[6/6] Computing body measurements")
    measurer = BodyMeasurements(fitted_joints, fitted_verts)
    measurements = measurer.compute_all()
    
    # Print measurements
    print(f"\n{'='*60}")
    print(f"  BODY MEASUREMENTS")
    print(f"{'='*60}")
    for name, vals in measurements.items():
        print(f"  {name:.<40s} {vals['cm']:>7.1f} cm  ({vals['in']:>5.1f} in)")
    print(f"{'='*60}")
    
    # ---- Save results ----
    base_name = os.path.splitext(os.path.basename(left_path))[0]
    
    # Save measurements CSV
    csv_path = os.path.join(output_dir, f"{base_name}_measurements_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Measurement', 'mm', 'cm', 'inches'])
        for name, vals in measurements.items():
            writer.writerow([name, f"{vals['mm']:.1f}", f"{vals['cm']:.1f}", f"{vals['in']:.1f}"])
    print(f"\n  Saved: {csv_path}")
    
    # Save 3D joints
    joints_path = os.path.join(output_dir, f"{base_name}_joints3d_{timestamp}.npy")
    np.save(joints_path, fitted_joints)
    print(f"  Saved: {joints_path}")
    
    # Save SMPL params
    params_path = os.path.join(output_dir, f"{base_name}_smpl_params_{timestamp}.npz")
    np.savez(params_path, betas=betas, pose=pose, trans=trans)
    print(f"  Saved: {params_path}")
    
    # Save visualization
    vis = visualize_results(rect_left, rect_right, fitted_joints, measurements,
                           landmarks_left, landmarks_right)
    vis_path = os.path.join(output_dir, f"{base_name}_visualization_{timestamp}.png")
    cv2.imwrite(vis_path, vis)
    print(f"  Saved: {vis_path}")
    
    # Save 3D joint view
    joint_vis = create_3d_joint_visualization(fitted_joints)
    jvis_path = os.path.join(output_dir, f"{base_name}_3d_joints_{timestamp}.png")
    cv2.imwrite(jvis_path, joint_vis)
    print(f"  Saved: {jvis_path}")
    
    # Save mesh (OBJ format)
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
    
    # Validate inputs
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
