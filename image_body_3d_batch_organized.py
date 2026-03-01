#!/usr/bin/env python3
"""
image_body_3d_modular.py
Modular 3D Biomechanical Analysis from Stereo Images
Supports MediaPipe and HRNet models with clean separation of concerns
"""

import cv2
import glob
from datetime import datetime

def create_dated_folder(base, src):
    d = datetime.now().strftime('%Y-%m-%d')
    p = base + '/' + d
    i = 1
    while os.path.exists(p):
        p = base + '/' + d + '_' + str(i)
        i += 1
    os.makedirs(p, exist_ok=True)
    print('📁 ' + p)
    return p

def copy_files(src, dest, image_nums):
    """Copy NPZ and multiple images to session folder"""
    import shutil
    import glob
    
    # Copy NPZ
    npz = max(glob.glob(src + '/*.npz'), key=os.path.getmtime)
    shutil.copy2(npz, dest)
    print('  ✓ ' + os.path.basename(npz))
    
    # Copy all images
    for num in image_nums:
        if src == 'stereo':
            # stereo: test_image1.png (combined)
            fname = 'test_image' + num + '.png'
            fpath = src + '/' + fname
            if os.path.exists(fpath):
                shutil.copy2(fpath, dest)
                print('  ✓ ' + fname)
            prefix = 'test_image'
        else:
            # ip: test_ss_left1.png, test_ss_right1.png
            for s in ['left','right']:
                fname = 'test_ss_' + s + num + '.png'
                fpath = src + '/' + fname
                if os.path.exists(fpath):
                    shutil.copy2(fpath, dest)
                    print('  ✓ ' + fname)
            prefix = 'test_ss'
    
    return os.path.join(dest, os.path.basename(npz)), prefix

def split_sidebyside_image(combined_img):
    """Split side-by-side stereo image into left and right halves"""
    h, w = combined_img.shape[:2]
    mid = w // 2
    img_left = combined_img[:, :mid]
    img_right = combined_img[:, mid:]
    return img_left, img_right


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import csv
import argparse
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the pose estimation pipeline"""
    SOURCES = {
        'stereo': {'params': 'stereo/stereo_params_sidebyside.npz', 'image_prefix': 'stereo/test', 'output_base': 'stereo'},
        'ip': {'params': 'ip/stereo_params.npz', 'image_prefix': 'ip/test', 'output_base': 'ip'}
    }
    HRNET_PATH = "/media/raviappala/edgeextvol2/jetson-inference/3dpose/HRnet_0.1"
    PARAMS_FILE = "stereo/stereo_params_sidebyside.npz"
    IMAGE_PREFIX = "stereo/test"
    OUTPUT_BASE = "stereo"
    MIN_AXIS_LIMIT = 600
    OUTPUT_DIRS = {'mediapipe': 'biomech_analysis_mediapipe', 'hrnet': 'biomech_analysis_hrnet'}

# ============================================================================
# CALIBRATION MANAGEMENT
# ============================================================================

class StereoCalibration:
    """Handles stereo camera calibration parameters"""
    
    def __init__(self, params_file):
        if not os.path.exists(params_file):
            raise RuntimeError(f"Missing calibration file: {params_file}")
        
        data = np.load(params_file)
        self.K1 = data["K1"]
        self.dist1 = data["dist1"]
        self.K2 = data["K2"]
        self.dist2 = data["dist2"]
        self.P1 = data["P1"]
        self.P2 = data["P2"]
        self.R_stereo = data.get("R", None)
        self.T_stereo = data.get("T", None)
        
        if self.R_stereo is None or self.T_stereo is None:
            raise RuntimeError("stereo_params.npz must contain R and T")
        
        self.T_stereo = self.T_stereo.reshape(3,)
        self.baseline = np.linalg.norm(self.T_stereo)
        
        # Load RMS reprojection error from calibration if available
        self.rms_error = data.get("stereo_rms", 0.0)
        
        # Compute world transform
        self.R_worldcam, self.origin_world, _ = self._build_world_transform()
        
        print("=== CALIBRATION PARAMS ===")
        print(f"Baseline: {self.baseline:.2f} mm")
        if self.rms_error > 0:
            print(f"RMS Reprojection Error: {self.rms_error:.4f} pixels")
    
    def _build_world_transform(self):
        """Build world coordinate system from stereo geometry"""
        C1 = np.zeros(3)
        C2 = self.T_stereo.reshape(3,)
        origin = (C1 + C2) / 2.0
        
        left_up = np.array([0.0, -1.0, 0.0])
        right_up = (self.R_stereo @ left_up)
        avg_up = left_up + right_up
        if np.linalg.norm(avg_up) < 1e-6:
            avg_up = left_up
        avg_up = avg_up / np.linalg.norm(avg_up)
        
        baseline = (C2 - C1)
        baseline_len = np.linalg.norm(baseline)
        if baseline_len < 1e-6:
            raise RuntimeError("Baseline vector too small")
        baseline_dir = baseline / baseline_len
        
        baseline_proj = baseline_dir - np.dot(baseline_dir, avg_up) * avg_up
        if np.linalg.norm(baseline_proj) < 1e-6:
            x_axis = baseline_dir
        else:
            x_axis = baseline_proj / np.linalg.norm(baseline_proj)
        
        x_axis = -x_axis
        z_axis = np.cross(x_axis, avg_up)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        R_wc = np.vstack([x_axis, y_axis, z_axis])
        return R_wc, origin, baseline_len
    
    def triangulate_point(self, ptL, ptR):
        """Triangulate a 3D point from stereo correspondences"""
        ptsL = np.array([[ptL[0]], [ptL[1]]], dtype=np.float64)
        ptsR = np.array([[ptR[0]], [ptR[1]]], dtype=np.float64)
        X_h = cv2.triangulatePoints(self.P1, self.P2, ptsL, ptsR)
        X = (X_h[:3] / X_h[3]).reshape(3,)
        return X
    
    def transform_to_world(self, pts3d_cam):
        """Transform 3D points from camera to world coordinates"""
        pts3d_world = []
        for p_cam in pts3d_cam:
            p_rel = p_cam - self.origin_world
            p_w = self.R_worldcam @ p_rel
            pts3d_world.append(p_w)
        return np.array(pts3d_world)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class PoseModel:
    """Base class for pose estimation models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.landmarks_map = {}
    
    def detect(self, frameL, frameR):
        """Detect 2D keypoints in stereo images"""
        raise NotImplementedError

class MediaPipeModel(PoseModel):
    """MediaPipe pose estimation model"""
    
    def __init__(self):
        super().__init__("mediapipe")
        
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(f"MediaPipe not available: {e}. Install mediapipe to use this model.")
        
        mp_pose = mp.solutions.pose
        self.model = mp_pose.Pose(static_image_mode=True, model_complexity=1, 
                                   min_detection_confidence=0.5)
        
        self.landmarks_map = {
            "nose": 0,
            "Leye_inner": 1, "Leye": 2, "Leye_outer": 3,
            "Reye_inner": 4, "Reye": 5, "Reye_outer": 6,
            "Lear": 7, "Rear": 8,
            "Lmouth": 9, "Rmouth": 10,
            "Lshoulder": 11, "Rshoulder": 12,
            "Lelbow": 13, "Relbow": 14,
            "Lwrist": 15, "Rwrist": 16,
            "Lpinky": 17, "Rpinky": 18,
            "Lindex": 19, "Rindex": 20,
            "Lthumb": 21, "Rthumb": 22,
            "Lhip": 23, "Rhip": 24,
            "Lknee": 25, "Rknee": 26,
            "Lankle": 27, "Rankle": 28,
            "Lheel": 29, "Rheel": 30,
            "Lfoot": 31, "Rfoot": 32
        }
    
    def detect(self, frameL, frameR):
        """Detect 2D keypoints using MediaPipe"""
        resL = self.model.process(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB))
        resR = self.model.process(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB))
        
        if not resL.pose_landmarks or not resR.pose_landmarks:
            return None, None
        
        hL, wL = frameL.shape[:2]
        hR, wR = frameR.shape[:2]
        
        kp_left = {}
        kp_right = {}
        
        for name, idx in self.landmarks_map.items():
            lm_l = resL.pose_landmarks.landmark[idx]
            lm_r = resR.pose_landmarks.landmark[idx]
            kp_left[name] = np.array([lm_l.x * wL, lm_l.y * hL])
            kp_right[name] = np.array([lm_r.x * wR, lm_r.y * hR])
        
        return kp_left, kp_right

class HRNetModel(PoseModel):
    """HRNet pose estimation model"""
    
    def __init__(self):
        super().__init__("hrnet")
        
        # Add HRNet paths
        sys.path.insert(0, Config.HRNET_PATH)
        sys.path.insert(0, os.path.join(Config.HRNET_PATH, 'demo'))
        sys.path.insert(0, os.path.join(Config.HRNET_PATH, 'lib'))
        
        # Import torch and other dependencies here to avoid import errors when not using HRNet
        try:
            import torch
            import torchvision
            import _init_paths
            import models
            from config import cfg, update_config
            from lib.core.inference import get_final_preds
            from lib.utils.transforms import get_affine_transform
        except ImportError as e:
            raise RuntimeError(f"HRNet dependencies not available: {e}. Install torch and torchvision to use HRNet model.")
        
        # Store lib utilities
        self.get_final_preds = get_final_preds
        self.get_affine_transform = get_affine_transform
        
        # Load config (same as working original script)
        config_file = os.path.join(Config.HRNET_PATH, 'demo/inference-config.yaml')
        
        class Args:
            def __init__(self):
                self.cfg = config_file
                self.opts = []
                self.modelDir = ''
                self.logDir = ''
                self.dataDir = ''
        
        args = Args()
        update_config(cfg, args)
        
        self.cfg = cfg
        self.CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Load person detection (match original approach)
        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.box_model.to(self.CTX)
        self.box_model.eval()
        
        # Load HRNet pose model (use W32, not W48)
        self.pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
        model_file = os.path.join(Config.HRNET_PATH, 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth')
        state_dict = torch.load(model_file, map_location=self.CTX)
        self.pose_model.load_state_dict(state_dict)
        self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=cfg.GPUS)
        self.pose_model.to(self.CTX)
        self.pose_model.eval()
        
        self.landmarks_map = {
            "nose": 0,
            "Leye": 1, "Reye": 2,
            "Lear": 3, "Rear": 4,
            "Lshoulder": 5, "Rshoulder": 6,
            "Lelbow": 7, "Relbow": 8,
            "Lwrist": 9, "Rwrist": 10,
            "Lhip": 11, "Rhip": 12,
            "Lknee": 13, "Rknee": 14,
            "Lankle": 15, "Rankle": 16
        }
    
    def detect(self, frameL, frameR):
        """Detect 2D keypoints using HRNet"""
        import torch
        import torchvision.transforms as transforms
        
        def get_person_boxes(model, img_list, threshold=0.7):
            with torch.no_grad():
                preds = model(img_list)
            boxes = []
            for pred in preds:
                keep = pred['scores'] > threshold
                boxes_img = pred['boxes'][keep]
                labels = pred['labels'][keep]
                person_boxes = boxes_img[labels == 1]
                if len(person_boxes) > 0:
                    boxes.append(person_boxes[0].cpu().numpy())
                else:
                    boxes.append(None)
            return boxes
        
        imgL_t = transforms.ToTensor()(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)).to(self.CTX)
        imgR_t = transforms.ToTensor()(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB)).to(self.CTX)
        boxes = get_person_boxes(self.box_model, [imgL_t, imgR_t])
        
        if boxes[0] is None or boxes[1] is None:
            return None, None
        
        kp_left = self._detect_single(frameL, boxes[0])
        kp_right = self._detect_single(frameR, boxes[1])
        
        return kp_left, kp_right
    
    def _detect_single(self, frame, box):
        """Detect keypoints in a single image"""
        import torch
        import torchvision.transforms as transforms
        
        x1, y1, x2, y2 = box
        center = np.array([(x1+x2)/2, (y1+y2)/2])
        scale = max(x2-x1, y2-y1) / 200.0
        
        trans = self.get_affine_transform(center, scale, 0, self.cfg.MODEL.IMAGE_SIZE)
        img_crop = cv2.warpAffine(
            frame, trans, tuple(self.cfg.MODEL.IMAGE_SIZE),
            flags=cv2.INTER_LINEAR
        )
        
        img_t = transforms.ToTensor()(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        img_t = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(img_t)
        
        with torch.no_grad():
            outputs = self.pose_model(img_t.unsqueeze(0).to(self.CTX))
        
        heatmaps = outputs.cpu().numpy()
        preds, maxvals = self.get_final_preds(self.cfg, heatmaps, np.array([center]), np.array([scale]))
        
        kp = {}
        for name, idx in self.landmarks_map.items():
            if idx < preds.shape[1]:
                kp[name] = preds[0, idx, :]
        
        return kp

def load_model(model_type):
    """Factory function to load the specified model"""
    if model_type == 'mediapipe':
        return MediaPipeModel()
    elif model_type == 'hrnet':
        return HRNetModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ============================================================================
# 3D RECONSTRUCTION
# ============================================================================

class Pose3D:
    """Handles 3D pose reconstruction from 2D detections"""
    
    def __init__(self, calibration, landmarks_map):
        self.calib = calibration
        self.lm_map = landmarks_map
    
    def reconstruct(self, kp_left, kp_right, verbose=False):
        """Triangulate 2D keypoints to 3D points"""
        pts3d_cam = []
        sources = []
        
        for name in sorted(self.lm_map.keys()):
            if name in kp_left and name in kp_right:
                X = self.calib.triangulate_point(kp_left[name], kp_right[name])
                pts3d_cam.append(X)
                sources.append(name)
        
        pts3d_cam = np.array(pts3d_cam)
        pts3d_world = self.calib.transform_to_world(pts3d_cam)
        
        if verbose:
            print(f"\n=== DEBUG: Coordinate Transformation ===")
            if 'nose' in sources:
                idx = sources.index('nose')
                print(f"Sample camera coords (nose): {pts3d_cam[idx]}")
                print(f"Sample world coords (nose): {pts3d_world[idx]}")
            print(f"Origin: {self.calib.origin_world}")
            print(f"Baseline: {self.calib.baseline:.2f} mm")
            print("=" * 40)
        
        points = {name: pts3d_world[i] for i, name in enumerate(sources)}
        return points

# ============================================================================
# BIOMECHANICAL MEASUREMENTS
# ============================================================================

class LengthAnalysis:
    """Calculate body measurements and angles"""
    
    @staticmethod
    def dist(p1, p2):
        """3D distance in cm"""
        return np.linalg.norm(p1 - p2) / 10.0
    
    @staticmethod
    def get_angle_deg(v1, v2):
        """Angle between two vectors in degrees"""
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def calculate_segments(self, points):
        """Calculate all segment lengths"""
        m = {}
        dist = self.dist
        
        # Torso
        if all(k in points for k in ["Lshoulder", "Rshoulder"]):
            m["Shoulder Width"] = dist(points["Lshoulder"], points["Rshoulder"])
        if all(k in points for k in ["Lhip", "Rhip"]):
            m["Hip Width"] = dist(points["Lhip"], points["Rhip"])
        if all(k in points for k in ["Lshoulder", "Rshoulder", "Lhip", "Rhip"]):
            m["Torso Length"] = dist(
                (points["Lshoulder"] + points["Rshoulder"]) / 2,
                (points["Lhip"] + points["Rhip"]) / 2
            )
        
        # Arms & Legs
        limb_pairs = [
            ("Lshoulder", "Lelbow", "L Humerus"), ("Rshoulder", "Relbow", "R Humerus"),
            ("Lelbow", "Lwrist", "L Radius"), ("Relbow", "Rwrist", "R Radius"),
            ("Lhip", "Lknee", "L Femur"), ("Rhip", "Rknee", "R Femur"),
            ("Lknee", "Lankle", "L Tibia"), ("Rknee", "Rankle", "R Tibia")
        ]
        for a, b, name in limb_pairs:
            if a in points and b in points:
                m[name] = dist(points[a], points[b])
        
        return m
    
    def calculate_angles(self, points):
        """Calculate all joint angles"""
        angles = {}
        get_angle = self.get_angle_deg
        
        # Shoulders
        if all(k in points for k in ["Lelbow", "Lshoulder", "Lhip", "Rhip"]):
            v1 = points["Lelbow"] - points["Lshoulder"]
            v2 = (points["Lhip"] + points["Rhip"]) / 2 - points["Lshoulder"]
            angles["L Shoulder"] = get_angle(v1, v2)
        
        if all(k in points for k in ["Relbow", "Rshoulder", "Lhip", "Rhip"]):
            v1 = points["Relbow"] - points["Rshoulder"]
            v2 = (points["Lhip"] + points["Rhip"]) / 2 - points["Rshoulder"]
            angles["R Shoulder"] = get_angle(v1, v2)
        
        # Elbows
        if all(k in points for k in ["Lshoulder", "Lelbow", "Lwrist"]):
            angles["L Elbow"] = get_angle(
                points["Lshoulder"] - points["Lelbow"],
                points["Lwrist"] - points["Lelbow"]
            )
        if all(k in points for k in ["Rshoulder", "Relbow", "Rwrist"]):
            angles["R Elbow"] = get_angle(
                points["Rshoulder"] - points["Relbow"],
                points["Rwrist"] - points["Relbow"]
            )
        
        # Hips
        if all(k in points for k in ["Lshoulder", "Rshoulder", "Lhip", "Lknee"]):
            v1 = (points["Lshoulder"] + points["Rshoulder"]) / 2 - points["Lhip"]
            v2 = points["Lknee"] - points["Lhip"]
            angles["L Hip"] = get_angle(v1, v2)
        
        if all(k in points for k in ["Lshoulder", "Rshoulder", "Rhip", "Rknee"]):
            v1 = (points["Lshoulder"] + points["Rshoulder"]) / 2 - points["Rhip"]
            v2 = points["Rknee"] - points["Rhip"]
            angles["R Hip"] = get_angle(v1, v2)
        
        # Knees
        if all(k in points for k in ["Lhip", "Lknee", "Lankle"]):
            angles["L Knee"] = get_angle(
                points["Lhip"] - points["Lknee"],
                points["Lankle"] - points["Lknee"]
            )
        if all(k in points for k in ["Rhip", "Rknee", "Rankle"]):
            angles["R Knee"] = get_angle(
                points["Rhip"] - points["Rknee"],
                points["Rankle"] - points["Rknee"]
            )
        
        # Ankles
        if all(k in points for k in ["Lknee", "Lankle", "Lfoot"]):
            angles["L Ankle"] = get_angle(
                points["Lknee"] - points["Lankle"],
                points["Lfoot"] - points["Lankle"]
            )
        if all(k in points for k in ["Rknee", "Rankle", "Rfoot"]):
            angles["R Ankle"] = get_angle(
                points["Rknee"] - points["Rankle"],
                points["Rfoot"] - points["Rankle"]
            )
        
        return angles
    
    def estimate_height(self, points, segments):
        """Estimate total body height"""
        leg_avg = sum(segments.get(k, 0) for k in ["L Femur", "L Tibia", "R Femur", "R Tibia"]) / 2
        torso = segments.get("Torso Length", 0)
        
        # Estimate head-top
        def estimate_head_top(points):
            required = ["nose", "Leye", "Reye", "Lear", "Rear"]
            if not all(k in points for k in required):
                return points.get("nose", np.array([0, 0, 0])) + np.array([0, 0, 150.0])
            
            eye_avg = (points["Leye"] + points["Reye"]) / 2.0
            ear_top_z = max(points["Lear"][2], points["Rear"][2])
            eye_top_z = max(eye_avg[2], ear_top_z)
            nose_to_eye = np.linalg.norm(points["nose"] - eye_avg)
            
            head_top = eye_avg.copy()
            head_top[2] = eye_top_z + nose_to_eye
            return head_top
        
        shoulder_mid = (points["Lshoulder"] + points["Rshoulder"]) / 2.0 \
            if "Lshoulder" in points and "Rshoulder" in points else np.array([0, 0, 0])
        head_top = estimate_head_top(points)
        neck_head = self.dist(head_top, shoulder_mid)
        
        return leg_avg + torso + neck_head

# ============================================================================
# OUTPUT GENERATION
# ============================================================================

class OutputManager:
    """Manages all output generation (CSV, plots, reports)"""
    
    def __init__(self, output_dir, model_name):
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    
    def annotate_joints(self, frame, keypoints, color=(0, 255, 0)):
        """Mark joints on the image"""
        annotated = frame.copy()
        
        for name, pt in keypoints.items():
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(annotated, (x, y), 5, color, -1)
            cv2.putText(annotated, name, (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        output_path = os.path.join(self.output_dir, f"joints_annotated_{self.timestamp}.png")
        cv2.imwrite(output_path, annotated)
        print(f"✓ Annotated joints saved: {output_path}")
        return output_path
    
    def annotate_measurements(self, frame, keypoints, segments, angles):
        """Draw measurements on the image"""
        annotated = frame.copy()
        
        # Draw skeleton connections
        connections = [
            ("Lshoulder", "Lelbow"), ("Lelbow", "Lwrist"),
            ("Rshoulder", "Relbow"), ("Relbow", "Rwrist"),
            ("Lshoulder", "Rshoulder"),
            ("Lhip", "Rhip"),
            ("Lshoulder", "Lhip"), ("Rshoulder", "Rhip"),
            ("Lhip", "Lknee"), ("Lknee", "Lankle"),
            ("Rhip", "Rknee"), ("Rknee", "Rankle")
        ]
        
        for start, end in connections:
            if start in keypoints and end in keypoints:
                pt1 = tuple(map(int, keypoints[start]))
                pt2 = tuple(map(int, keypoints[end]))
                cv2.line(annotated, pt1, pt2, (255, 255, 255), 2)
        
        # Annotate segment lengths
        segment_annotations = [
            ("L Humerus", "Lelbow", (0, 255, 0)),
            ("R Humerus", "Relbow", (0, 255, 0)),
            ("L Radius", "Lwrist", (0, 200, 0)),
            ("R Radius", "Rwrist", (0, 200, 0)),
            ("L Femur", "Lknee", (255, 0, 0)),
            ("R Femur", "Rknee", (255, 0, 0)),
            ("L Tibia", "Lankle", (200, 0, 0)),
            ("R Tibia", "Rankle", (200, 0, 0))
        ]
        
        for seg_name, joint, color in segment_annotations:
            if seg_name in segments and joint in keypoints:
                x, y = int(keypoints[joint][0]), int(keypoints[joint][1])
                text = f"{seg_name.split()[1]}: {segments[seg_name]:.1f}cm"
                cv2.putText(annotated, text, (x+5, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Annotate joint angles
        angle_annotations = [
            ("L Elbow", "Lelbow", (0, 255, 255)),
            ("R Elbow", "Relbow", (0, 255, 255)),
            ("L Knee", "Lknee", (255, 255, 0)),
            ("R Knee", "Rknee", (255, 255, 0))
        ]
        
        for ang_name, joint, color in angle_annotations:
            if ang_name in angles and joint in keypoints:
                x, y = int(keypoints[joint][0]), int(keypoints[joint][1])
                text = f"{ang_name.split()[1]}: {angles[ang_name]:.0f}°"
                cv2.putText(annotated, text, (x-60, y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        output_path = os.path.join(self.output_dir, f"measurements_annotated_{self.timestamp}.png")
        cv2.imwrite(output_path, annotated)
        print(f"✓ Annotated measurements saved: {output_path}")
        return output_path
    
    def create_3d_plot(self, points_3d, landmarks_map):
        """Create 3D skeleton visualization"""
        # Get points as array
        point_names = sorted(landmarks_map.keys())
        pts3d = np.array([points_3d[name] for name in point_names if name in points_3d])
        
        if len(pts3d) == 0:
            print("Warning: No 3D points to plot")
            return None
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"3D Skeleton ({self.model_name})")
        
        # Plot points (swap axes: Z->X, X->Y, Y->Z for standing view)
        ax.scatter(pts3d[:, 2], pts3d[:, 0], pts3d[:, 1], c='r', s=20)
        
        # Draw skeleton connections
        connections = [
            ("Lshoulder", "Lelbow"), ("Lelbow", "Lwrist"),
            ("Rshoulder", "Relbow"), ("Relbow", "Rwrist"),
            ("Lshoulder", "Rshoulder"),
            ("Lhip", "Rhip"),
            ("Lshoulder", "Lhip"), ("Rshoulder", "Rhip"),
            ("Lhip", "Lknee"), ("Lknee", "Lankle"),
            ("Rhip", "Rknee"), ("Rknee", "Rankle"),
            ("nose", "Lshoulder"), ("nose", "Rshoulder")
        ]
        
        for start, end in connections:
            if start in points_3d and end in points_3d:
                v1, v2 = points_3d[start], points_3d[end]
                ax.plot([v1[2], v2[2]], [v1[0], v2[0]], [v1[1], v2[1]], 
                       color='blue', alpha=0.6)
        
        # Set labels and equal aspect
        ax.set_xlabel("X (Depth)")
        ax.set_ylabel("Y (Width)")
        ax.set_zlabel("Z (Height)")
        
        # Equal scale
        padding = 0.2
        z_min, z_max = np.min(pts3d[:, 2]), np.max(pts3d[:, 2])
        x_min, x_max = np.min(pts3d[:, 0]), np.max(pts3d[:, 0])
        y_min, y_max = np.min(pts3d[:, 1]), np.max(pts3d[:, 1])
        
        max_range = max(z_max-z_min, x_max-x_min, y_max-y_min)
        padded_range = max_range * (1 + 2 * padding)
        
        z_mid, x_mid, y_mid = (z_min+z_max)/2, (x_min+x_max)/2, (y_min+y_max)/2
        half_range = padded_range / 2.0
        
        ax.set_xlim(z_mid - half_range, z_mid + half_range)
        ax.set_ylim(x_mid - half_range, x_mid + half_range)
        ax.set_zlim(y_mid - half_range, y_mid + half_range)
        ax.set_box_aspect([1, 1, 1])
        
        # Set view angle
        ax.view_init(elev=15, azim=-75)
        
        output_path = os.path.join(self.output_dir, f"skeleton_3d_{self.timestamp}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 3D skeleton saved: {output_path}")
        return output_path
    
    def save_csv(self, segments, angles, height_info, camera_info, points_3d=None, scale_info=None, verbose=False):
        """Save measurements to CSV"""
        csv_path = os.path.join(self.output_dir, f"measurements_{self.timestamp}.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Parameters section
            writer.writerow(["PARAMETERS"])
            writer.writerow(["Model Used", self.model_name])
            writer.writerow(["Timestamp", self.timestamp])
            writer.writerow(["RMS Reprojection Error", f"{camera_info['rms']:.2f}"])
            writer.writerow(["Distance Between Cameras", f"{camera_info['distance']:.2f}"])
            writer.writerow(["Angle Between Cameras (deg)", f"{camera_info['angle']:.2f}"])
            
            if height_info['provided']:
                writer.writerow(["Human Height Provided", f"{height_info['provided']:.2f}"])
                if verbose and scale_info:
                    writer.writerow(["Original Calculated Height", f"{scale_info['original']:.2f}"])
                    writer.writerow(["Scale Factor Applied", f"{scale_info['factor']:.4f}"])
                    writer.writerow(["Height Difference", f"{scale_info['difference']:.2f}"])
            else:
                writer.writerow(["Human Height Calculated", f"{height_info['calculated']:.2f}"])
            
            writer.writerow([])
            
            # Joint coordinates - add this section
            writer.writerow(["JOINT COORDINATES (mm)"])
            writer.writerow(["Body Part", "Left X", "Left Y", "Left Z", "Right X", "Right Y", "Right Z"])
            
            # Group bilateral joints
            bilateral_joints = {
                "eye": ("Leye", "Reye"),
                "ear": ("Lear", "Rear"),
                "shoulder": ("Lshoulder", "Rshoulder"),
                "elbow": ("Lelbow", "Relbow"),
                "wrist": ("Lwrist", "Rwrist"),
                "hip": ("Lhip", "Rhip"),
                "knee": ("Lknee", "Rknee"),
                "ankle": ("Lankle", "Rankle"),
                "heel": ("Lheel", "Rheel"),
                "foot": ("Lfoot", "Rfoot")
            }
            
            # Add coordinates
            if points_3d:
                for part_name, (left_key, right_key) in bilateral_joints.items():
                    left_coords = points_3d.get(left_key, np.array([0, 0, 0]))
                    right_coords = points_3d.get(right_key, np.array([0, 0, 0]))
                    writer.writerow([
                        part_name,
                        f"{left_coords[0]:.2f}", f"{left_coords[1]:.2f}", f"{left_coords[2]:.2f}",
                        f"{right_coords[0]:.2f}", f"{right_coords[1]:.2f}", f"{right_coords[2]:.2f}"
                    ])
                
                # Center parts (nose)
                if "nose" in points_3d:
                    nose = points_3d["nose"]
                    writer.writerow(["nose", f"{nose[0]:.2f}", f"{nose[1]:.2f}", f"{nose[2]:.2f}", "", "", ""])
            
            writer.writerow([])
            
            # Segment lengths
            writer.writerow(["SEGMENT LENGTHS (cm)"])
            writer.writerow(["Segment", "Left Length", "Right Length", "Difference"])
            
            # Group segments by body part
            segment_groups = {
                "Humerus": ("L Humerus", "R Humerus"),
                "Radius": ("L Radius", "R Radius"),
                "Femur": ("L Femur", "R Femur"),
                "Tibia": ("L Tibia", "R Tibia")
            }
            
            for name, (left_key, right_key) in segment_groups.items():
                left_val = segments.get(left_key, 0)
                right_val = segments.get(right_key, 0)
                diff = abs(left_val - right_val)
                writer.writerow([name, f"{left_val:.2f}", f"{right_val:.2f}", f"{diff:.2f}"])
            
            # Center measurements
            for key in ["Shoulder Width", "Hip Width", "Torso Length"]:
                if key in segments:
                    writer.writerow([key, f"{segments[key]:.2f}", "", ""])
            
            writer.writerow([])
            
            # Joint angles
            writer.writerow(["JOINT ANGLES (degrees)"])
            writer.writerow(["Joint", "Left Angle", "Right Angle", "Difference"])
            
            angle_groups = {
                "Shoulder": ("L Shoulder", "R Shoulder"),
                "Elbow": ("L Elbow", "R Elbow"),
                "Hip": ("L Hip", "R Hip"),
                "Knee": ("L Knee", "R Knee"),
                "Ankle": ("L Ankle", "R Ankle")
            }
            
            for name, (left_key, right_key) in angle_groups.items():
                left_val = angles.get(left_key, 0)
                right_val = angles.get(right_key, 0)
                diff = abs(left_val - right_val)
                writer.writerow([name, f"{left_val:.2f}", f"{right_val:.2f}", f"{diff:.2f}"])
        
        print(f"✓ CSV saved: {csv_path}")
        return csv_path
    
    def save_session_info(self, command_args, image_nums, camera_info, height_info, results_summary):
        """Save session execution details to text file"""
        info_path = os.path.join(self.output_dir, f"session_info_{self.timestamp}.txt")
        
        with open(info_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SESSION INFORMATION\n")
            f.write("=" * 60 + "\n\n")
            
            # Command executed
            f.write("COMMAND EXECUTED:\n")
            f.write(f"  python3 image_body_3d_batch_organized.py \\\n")
            f.write(f"    --images {','.join(image_nums)} \\\n")
            f.write(f"    --height {command_args.height if command_args.height else 'auto'} \\\n")
            f.write(f"    --model {command_args.model} \\\n")
            f.write(f"    --source {command_args.source}\n\n")
            
            # Images processed
            f.write("IMAGES PROCESSED:\n")
            for num in image_nums:
                f.write(f"  - Image {num}\n")
            f.write(f"  Total: {len(image_nums)} images\n\n")
            
            # Calibration metrics
            f.write("CALIBRATION PARAMETERS:\n")
            f.write(f"  Baseline (distance between cameras): {camera_info['distance']:.2f} cm\n")
            if camera_info['rms'] > 0:
                f.write(f"  RMS Reprojection Error: {camera_info['rms']:.4f} pixels\n")
            f.write(f"  Camera Angle: {camera_info['angle']:.2f} degrees\n\n")
            
            # Results summary
            f.write("RESULTS SUMMARY:\n")
            if height_info['provided']:
                f.write(f"  Target Height: {height_info['provided']:.1f} cm\n")
                f.write(f"  Average Measured Height (raw): {results_summary['mean_height']:.1f} cm\n")
                f.write(f"  Scaling Applied: Yes\n")
            else:
                f.write(f"  Calculated Height: {height_info['calculated']:.1f} cm\n")
            
            f.write(f"  Successful Measurements: {results_summary['success_count']}/{results_summary['total_count']}\n\n")
            
            # Key measurements
            if 'segments' in results_summary:
                f.write("KEY MEASUREMENTS (averaged, cm):\n")
                for seg, val in sorted(results_summary['segments'].items()):
                    f.write(f"  {seg}: {val:.2f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Generated: {self.timestamp}\n")
            f.write("=" * 60 + "\n")
        
        print(f"✓ Session info saved: {info_path}")
        return info_path

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Modular 3D Biomechanical Analysis')
    parser.add_argument('--images', '-i', type=str, required=True, help='Comma-separated image numbers (e.g., 1,2,3,5)')
    parser.add_argument('--height', type=float, default=None, help='True height in cm for scaling')
    parser.add_argument('--model', '-m', type=str, default='mediapipe', choices=['mediapipe', 'hrnet'])
    parser.add_argument('--source', '-s', type=str, default='stereo',
                       choices=list(Config.SOURCES.keys()),
                       help='Camera source: stereo (default, current dir) or ip (ip/ folder)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show scale factor and calculation details')
    parser.add_argument('--debug', '-d', action='store_true', help='Save intermediate outputs (2D keypoints, 3D coords, etc.)')
    args = parser.parse_args()
    
    # Create session
    session = create_dated_folder(args.source, args.source)
    print('\n📋 Copying...')
    Config.PARAMS_FILE, img_pre = image_nums = [n.strip() for n in args.images.split(",")]
    Config.PARAMS_FILE, img_pre = copy_files(args.source, session, image_nums)
    Config.IMAGE_PREFIX = session + '/' + img_pre
    Config.OUTPUT_BASE = session
    print('📷 ' + args.source + '/biomech_analysis/\n')
    
    # Initialize components
    calib = StereoCalibration(Config.PARAMS_FILE)
    model = load_model(args.model)
    pose3d = Pose3D(calib, model.landmarks_map)
    analyzer = LengthAnalysis()
    output = OutputManager(Config.OUTPUT_BASE + "/biomech_analysis", args.model)
    
    # Process multiple images
    image_nums = [n.strip() for n in args.images.split(',')]
    results = []
    
    print(f"\nProcessing {len(image_nums)} images...")
    
    for img_num in image_nums:
        print(f"\n--- Image {img_num} ---")
        
        # Load images
        if args.source == 'stereo':
            combined_path = session + '/' + img_pre + img_num + '.png'
            combined_img = cv2.imread(combined_path)
            if combined_img is None:
                print(f'⚠ Skip: could not read {combined_path}')
                continue
            frameL, frameR = split_sidebyside_image(combined_img)
        else:
            img_left_path = session + '/' + img_pre + '_left' + img_num + '.png'
            img_right_path = session + '/' + img_pre + '_right' + img_num + '.png'
            frameL = cv2.imread(img_left_path)
            frameR = cv2.imread(img_right_path)
            if frameL is None or frameR is None:
                print(f'⚠ Skip: could not read images')
                continue
        
        # Detect 2D
        kp_left, kp_right = model.detect(frameL, frameR)
        if kp_left is None or kp_right is None:
            print("⚠ Skip: pose detection failed")
            continue
        
        # Reconstruct 3D
        points = pose3d.reconstruct(kp_left, kp_right, verbose=False)
        
        # Calculate measurements
        segments = analyzer.calculate_segments(points)
        angles = analyzer.calculate_angles(points)
        measured_height = analyzer.estimate_height(points, segments)
        
        # Store result
        results.append({
            'image_num': img_num,
            'frame': frameL,
            'keypoints_2d': kp_left,
            'points_3d': points,
            'segments': segments,
            'angles': angles,
            'height': measured_height,
            'scale_factor': args.height / measured_height if args.height else 1.0
        })
        print(f"✓ Height: {measured_height:.1f} cm")
    
    if not results:
        print("\nError: No successful measurements")
        return
    
    print(f"\n✓ Processed {len(results)}/{len(image_nums)} images")
    
    # Compute averages
    all_heights = [r['height'] for r in results]
    mean_height = np.mean(all_heights)
    
    # Apply scaling
    if args.height:
        for r in results:
            for k in r['segments']:
                r['segments'][k] *= r['scale_factor']
    
    # Average 3D points
    all_point_names = set()
    for r in results:
        all_point_names.update(r['points_3d'].keys())
    
    mean_points_3d = {}
    for name in all_point_names:
        pts = [r['points_3d'][name] for r in results if name in r['points_3d']]
        if pts:
            mean_points_3d[name] = np.mean(pts, axis=0)
    
    # Average segments
    all_segment_names = set()
    for r in results:
        all_segment_names.update(r['segments'].keys())
    
    segments = {}
    for seg in all_segment_names:
        vals = [r['segments'][seg] for r in results if seg in r['segments']]
        if vals:
            segments[seg] = np.mean(vals)
    
    # Average angles
    all_angle_names = set()
    for r in results:
        all_angle_names.update(r['angles'].keys())
    
    angles = {}
    for ang in all_angle_names:
        vals = [r['angles'][ang] for r in results if ang in r['angles']]
        if vals:
            angles[ang] = np.mean(vals)
    
    points = mean_points_3d
    measured_height = args.height if args.height else mean_height
    
    # Apply scaling
    scale_factor = 1.0
    original_height = measured_height
    scale_info = None
    
    if args.height is not None:
        scale_factor = args.height / measured_height
        
        # Scale all segments
        for key in segments:
            segments[key] *= scale_factor
        
        # Print scaling info
        if args.verbose:
            print(f"\n=== HEIGHT SCALING ===")
            print(f"Calculated Height: {original_height:.2f} cm")
            print(f"Provided Height: {args.height:.2f} cm")
            print(f"Scale Factor: {scale_factor:.4f}")
            print(f"Difference: {abs(args.height - original_height):.2f} cm ({abs(1-scale_factor)*100:.2f}%)")
            print("=" * 40)
        
        scale_info = {
            'original': original_height,
            'factor': scale_factor,
            'difference': abs(args.height - original_height)
        }
        measured_height = args.height
    
    # Calculate camera info from calibration
    camera_info = {
        'rms': calib.rms_error,
        'distance': calib.baseline / 10.0,
        'angle': 0.0  # Placeholder
    }
    
    height_info = {
        'calculated': measured_height,
        'provided': args.height
    }
    
    # Save outputs
    output.save_csv(segments, angles, height_info, camera_info, points, scale_info, args.verbose)
    output.annotate_joints(frameL, kp_left)
    output.annotate_measurements(frameL, kp_left, segments, angles)
    output.create_3d_plot(points, model.landmarks_map)
    
    # Save session info
    results_summary = {
        "mean_height": mean_height,
        "success_count": len(results),
        "total_count": len(image_nums),
        "segments": segments
    }
    output.save_session_info(args, image_nums, camera_info, height_info, results_summary)
    print(f"✓ All outputs in: {output.output_dir}/")

if __name__ == "__main__":
    main()
