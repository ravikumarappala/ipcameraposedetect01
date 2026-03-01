#!/usr/bin/env python3
"""
Full Body 3D Pose Estimation with HRNet
========================================
Based on image_body_3d.py but uses HRNet instead of MediaPipe for pose detection.
Uses same triangulation, metrics calculation, and visualization approach.

HRNet provides 17 COCO keypoints vs MediaPipe's 33 landmarks.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import argparse
from pathlib import Path

# HRNet imports
hrnet_root = "/media/raviappala/edgeextvol2/jetson-inference/3dpose/HRnet_0.1"
sys.path.append(os.path.join(hrnet_root, 'demo'))
import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

# ---------------------------
# Config
# ---------------------------
MIN_AXIS_LIMIT = 600     # units: mm

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CTX = DEVICE

# COCO Keypoint Mapping (17 keypoints)
COCO_KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

# Simplified names for metrics
LM = {
    "nose": 0,
    "Lshoulder": 5, "Rshoulder": 6,
    "Lelbow": 7, "Relbow": 8,
    "Lwrist": 9, "Rwrist": 10,
    "Lhip": 11, "Rhip": 12,
    "Lknee": 13, "Rknee": 14,
    "Lankle": 15, "Rankle": 16,
}

# COCO connections for skeleton visualization
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# COCO person detection categories
COCO_CATEGORIES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ---------------------------
# Load stereo params
# ---------------------------
PARAMS_FILE = "stereo_params_sidebyside.npz"
if not os.path.exists(PARAMS_FILE):
    raise RuntimeError(f"Missing calibration file: {PARAMS_FILE}")

data = np.load(PARAMS_FILE)
K1 = data["K1"]
dist1 = data["dist1"]
K2 = data["K2"]
dist2 = data["dist2"]
P1 = data["P1"]
P2 = data["P2"]
R_stereo = data.get("R", None)
T_stereo = data.get("T", None)
if R_stereo is None or T_stereo is None:
    raise RuntimeError("stereo_params.npz must contain R and T for world alignment")
T_stereo = T_stereo.reshape(3,)

print("=== CALIBRATION PARAMS ===")
print(f"Baseline: {np.linalg.norm(T_stereo):.2f} mm")

# ---------------------------
# HRNet Helper Functions
# ---------------------------
def get_person_boxes(model, input_list, threshold=0.7):
    """Detect person bounding boxes using Faster R-CNN"""
    pred = model(input_list)
    try:
        pred_classes = [COCO_CATEGORIES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    except Exception as e:
        print(f"Error parsing detection boxes: {e}")
        return []

    if not pred_score or max(pred_score) < threshold:
        return []
    
    filtered_indices = [i for i, x in enumerate(pred_score) if x > threshold]
    if not filtered_indices:
        return []
    pred_t = filtered_indices[-1]
    
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)
    return person_boxes

def box_to_center_scale(box, model_w, model_h):
    """Convert bounding box to center and scale"""
    center = np.zeros((2), dtype=np.float32)
    bottom_left = box[0]
    top_right = box[1]
    box_w = top_right[0] - bottom_left[0]
    box_h = top_right[1] - bottom_left[1]
    center[0] = bottom_left[0] + box_w * 0.5
    center[1] = bottom_left[1] + box_h * 0.5

    aspect_ratio = model_w * 1.0 / model_h
    pixel_std = 200

    if box_w > aspect_ratio * box_h:
        box_h = box_w * 1.0 / aspect_ratio
    elif box_w < aspect_ratio * box_h:
        box_w = box_h * aspect_ratio
    scale = np.array([box_w * 1.0 / pixel_std, box_h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    return center, scale

def get_pose_prediction(pose_model, image, center, scale):
    """Get pose keypoints from HRNet"""
    rotation = 0
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image, trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model_input = transform(model_input).unsqueeze(0)
    pose_model.eval()
    with torch.no_grad():
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg, output.clone().cpu().numpy(),
            np.asarray([center]), np.asarray([scale]))
    return preds

def load_hrnet_models():
    """Load HRNet person detection and pose estimation models"""
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Configure HRNet
    args = type('Args', (), {})()
    args.cfg = str(Path(hrnet_root) / 'demo' / 'inference-config.yaml')
    args.opts = []
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = str(Path(hrnet_root))
    update_config(cfg, args)

    # Load person detection model
    try:
        # Try new syntax (torchvision >= 0.13)
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    except ImportError:
        # Fallback to old syntax
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
        box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    # Load pose estimation model
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    if cfg.TEST.MODEL_FILE:
        print(f'Loading HRNet model from {cfg.TEST.MODEL_FILE}')
        if torch.cuda.is_available():
            pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        else:
            pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=False)
    else:
        raise RuntimeError('HRNet model file not specified in config TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()
    
    return box_model, pose_model

def detect_pose_hrnet(image, box_model, pose_model):
    """Detect pose in image using HRNet (returns 17x2 keypoints)"""
    if image is None:
        return None
    
    # Convert BGR to RGB
    image_rgb = image[:, :, [2, 1, 0]]
    img_tensor = torch.from_numpy(image/255.).permute(2,0,1).float().to(CTX)
    input_list = [img_tensor]

    # Detect person
    pred_boxes = get_person_boxes(box_model, input_list, threshold=0.7)
    
    if len(pred_boxes) >= 1:
        # Use first detected person
        box = pred_boxes[0]
        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        image_pose = image_rgb.copy() if cfg.DATASET.COLOR_RGB else image.copy()
        pose_preds = get_pose_prediction(pose_model, image_pose, center, scale)
        
        if len(pose_preds) >= 1:
            return pose_preds[0]  # 17x2 array
    return None

# ---------------------------
# Triangulation util (matching image_arm_3d.py)
# ---------------------------
def triangulate_point(ptL, ptR):
    """Triangulate single 3D point from left and right pixel coordinates"""
    ptsL = np.array([[ptL[0]], [ptL[1]]], dtype=np.float64)
    ptsR = np.array([[ptR[0]], [ptR[1]]], dtype=np.float64)
    X_h = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
    X = (X_h[:3] / X_h[3]).reshape(3,)
    return X

# ---------------------------
# World Transform (matching image_arm_3d.py)
# ---------------------------
def build_world_transform(R, T):
    """Build world coordinate frame from stereo extrinsics"""
    C1 = np.zeros(3)         
    C2 = T.reshape(3,)       
    origin = (C1 + C2) / 2.0  

    left_up = np.array([0.0, -1.0, 0.0])      
    right_up = (R @ left_up)                  
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

R_worldcam, origin_world, baseline_len = build_world_transform(R_stereo, T_stereo)

# ---------------------------
# Main Logic
# ---------------------------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D full body pose estimation with HRNet')
    parser.add_argument('--num', '-n', type=str, default='', 
                        help='Image number suffix (e.g., 1 for test_ss_left1.png)')
    parser.add_argument('--height', type=float, default=None,
                        help='True height in cm (optional - used to scale joints for better accuracy)')
    args = parser.parse_args()
    
    # Build image paths with optional number suffix
    suffix = args.num if args.num else ''
    img_left_path = f"test_ss_left{suffix}.png"
    img_right_path = f"test_ss_right{suffix}.png"

    if not os.path.isfile(img_left_path) or not os.path.isfile(img_right_path):
        print(f"Error: Default files '{img_left_path}' or '{img_right_path}' not found.")
        return

    print(f"\nProcessing:\n  Left: {img_left_path}\n  Right: {img_right_path}")

    frameL = cv2.imread(img_left_path)
    frameR = cv2.imread(img_right_path)
    if frameL is None or frameR is None:
        print("Error: Could not read images")
        return

    hL, wL = frameL.shape[:2]
    hR, wR = frameR.shape[:2]

    # Load HRNet models
    print("Loading HRNet models...")
    box_model, pose_model = load_hrnet_models()

    # Detect pose
    print("Running HRNet pose detection...")
    kp_left = detect_pose_hrnet(frameL, box_model, pose_model)
    kp_right = detect_pose_hrnet(frameR, box_model, pose_model)

    if kp_left is None or kp_right is None:
        print("Error: Could not detect pose in one or both images")
        return

    print(f"Detected {len(kp_left)} keypoints in each image")

    # Triangulate all 17 keypoints (using raw pixels like image_arm_3d.py)
    pts3d_cam = []
    jointsL_px = {}
    for i in range(17):
        # Use raw HRNet pixel coordinates (NO undistort/rectify)
        pL = (float(kp_left[i][0]), float(kp_left[i][1]))
        pR = (float(kp_right[i][0]), float(kp_right[i][1]))
        X = triangulate_point(pL, pR)  # Direct triangulation with P1, P2
        pts3d_cam.append(X)
        # Store pixel coords for annotation
        for name, idx in LM.items():
            if idx == i:
                jointsL_px[name] = (int(pL[0]), int(pL[1]))
    
    pts3d_cam = np.array(pts3d_cam)

    # Transform to world frame
    pts3d_world = []
    for p_cam in pts3d_cam:
        p_rel = p_cam - origin_world
        p_w = R_worldcam @ p_rel
        pts3d_world.append(p_w)
    pts3d_world = np.array(pts3d_world)

    # Dictionary mapping name -> 3D position
    points = {name: pts3d_world[idx] for name, idx in LM.items()}

    # Calculate Metrics
    print("\nCalculating body metrics...")
    m = {}
    def dist(p1, p2): return np.linalg.norm(p1 - p2) / 10.0
    
    def get_angle_deg(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return 0.0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

    # Segment lengths
    m["Shoulder Width"] = dist(points["Lshoulder"], points["Rshoulder"])
    m["Hip Width"] = dist(points["Lhip"], points["Rhip"])
    m["Torso Length"] = dist((points["Lshoulder"]+points["Rshoulder"])/2, (points["Lhip"]+points["Rhip"])/2)
    
    m["L Humerus"] = dist(points["Lshoulder"], points["Lelbow"])
    m["R Humerus"] = dist(points["Rshoulder"], points["Relbow"])
    m["L Radius"] = dist(points["Lelbow"], points["Lwrist"])
    m["R Radius"] = dist(points["Relbow"], points["Rwrist"])
    # Total arm = sum of segments (like image_arm_3d.py)
    m["L Arm Total"] = m["L Humerus"] + m["L Radius"]
    m["R Arm Total"] = m["R Humerus"] + m["R Radius"]
    
    m["L Femur"] = dist(points["Lhip"], points["Lknee"])
    m["R Femur"] = dist(points["Rhip"], points["Rknee"])
    m["L Tibia"] = dist(points["Lknee"], points["Lankle"])
    m["R Tibia"] = dist(points["Rknee"], points["Rankle"])
    # Total leg = sum of segments (consistent with arm approach)
    m["L Leg Total"] = m["L Femur"] + m["L Tibia"]
    m["R Leg Total"] = m["R Femur"] + m["R Tibia"]
    
    # Joint Angles
    m["L Shoulder Angle"] = get_angle_deg(points["Lelbow"]-points["Lshoulder"], (points["Lhip"]+points["Rhip"])/2-points["Lshoulder"])
    m["R Shoulder Angle"] = get_angle_deg(points["Relbow"]-points["Rshoulder"], (points["Lhip"]+points["Rhip"])/2-points["Rshoulder"])
    m["L Elbow Angle"] = get_angle_deg(points["Lshoulder"]-points["Lelbow"], points["Lwrist"]-points["Lelbow"])
    m["R Elbow Angle"] = get_angle_deg(points["Rshoulder"]-points["Relbow"], points["Rwrist"]-points["Relbow"])
    m["L Hip Angle"] = get_angle_deg((points["Lshoulder"]+points["Rshoulder"])/2-points["Lhip"], points["Lknee"]-points["Lhip"])
    m["R Hip Angle"] = get_angle_deg((points["Lshoulder"]+points["Rshoulder"])/2-points["Rhip"], points["Rknee"]-points["Rhip"])
    m["L Knee Angle"] = get_angle_deg(points["Lhip"]-points["Lknee"], points["Lankle"]-points["Lknee"])
    m["R Knee Angle"] = get_angle_deg(points["Rhip"]-points["Rknee"], points["Rankle"]-points["Rknee"])

    # Height estimates (COCO doesn't have heel/foot, so simplified)
    floor_y = (points["Lankle"][1] + points["Rankle"][1]) / 2.0 # Using ankle as proxy for heel/floor
    avg_leg = (m["L Femur"] + m["L Tibia"] + m["R Femur"] + m["R Tibia"]) / 2.0
    neck_head = dist3d(points["nose"], (points["Lshoulder"]+points["Rshoulder"])/2) + 15.0
    m["Height (Proxy)"] = abs(points["nose"][1] - floor_y) / 10.0 + 15.0  
    m["Height (Segments)"] = avg_leg + m["Torso Length"] + neck_head
    
    # Apply height scaling if provided
    scale_factor = 1.0
    if args.height is not None:
        measured_height = m["Height (Segments)"]
        scale_factor = args.height / measured_height
        print(f"\n🎯 HEIGHT SCALING ENABLED:")
        print(f"   Measured height: {measured_height:.1f} cm")
        print(f"   Provided height: {args.height:.1f} cm")
        print(f"   Scale factor: {scale_factor:.4f}")
        print(f"   Scaling all 3D joints...")
        
        # Scale all 3D points
        pts3d_world_scaled = pts3d_world * scale_factor
        points = {name: pts3d_world_scaled[idx] for name, idx in LM.items() if idx < len(pts3d_world_scaled)}
        
        # Recalculate ALL measurements with scaled points
        m["Shoulder Width"] = dist3d(points["Lshoulder"], points["Rshoulder"])
        m["Hip Width"] = dist3d(points["Lhip"], points["Rhip"])
        m["Torso Length"] = dist3d((points["Lshoulder"]+points["Rshoulder"])/2, (points["Lhip"]+points["Rhip"])/2)
        
        m["L Humerus"] = dist3d(points["Lshoulder"], points["Lelbow"])
        m["R Humerus"] = dist3d(points["Rshoulder"], points["Relbow"])
        m["L Radius"] = dist3d(points["Lelbow"], points["Lwrist"])
        m["R Radius"] = dist3d(points["Relbow"], points["Rwrist"])
        m["L Arm Total"] = m["L Humerus"] + m["L Radius"]
        m["R Arm Total"] = m["R Humerus"] + m["R Radius"]
        
        m["L Femur"] = dist3d(points["Lhip"], points["Lknee"])
        m["R Femur"] = dist3d(points["Rhip"], points["Rknee"])
        m["L Tibia"] = dist3d(points["Lknee"], points["Lankle"])
        m["R Tibia"] = dist3d(points["Rknee"], points["Rankle"])
        m["L Leg Total"] = m["L Femur"] + m["L Tibia"]
        m["R Leg Total"] = m["R Femur"] + m["R Tibia"]
        
        # Angles don't change with scaling
        m["Height (Segments)"] = args.height  # Now exact
        m["Height (Proxy)"] = abs(points["nose"][1] - floor_y) / 10.0 + 15.0
        
        # Use scaled points for 3D plot
        pts3d_world = pts3d_world_scaled

    # Print
    print("\n" + "="*40)
    print("RESULTS (HRNet)")
    print("="*40)
    for name, val in m.items():
        unit = "deg" if "Angle" in name else "cm"
        print(f"{name:<22}: {val:>8.2f} {unit}")
    print("="*40)

    # Save outputs
    out_dir = "whole_body_measurement_sidebyside_hrnet"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    
    # Annotated 2D image with ALL measurements
    dispL = frameL.copy()
    
    # Draw skeleton
    for conn in COCO_CONNECTIONS:
        if conn[0] in jointsL_px.values() or conn[1] in jointsL_px.values():
            # Find pixel positions
            pt1, pt2 = None, None
            for name, idx in LM.items():
                if idx == conn[0]:
                    pt1 = jointsL_px.get(name)
                if idx == conn[1]:
                    pt2 = jointsL_px.get(name)
            if pt1 and pt2:
                cv2.line(dispL, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for name, pt in jointsL_px.items():
        cv2.circle(dispL, pt, 5, (0, 0, 255), -1)
    
    # Comprehensive annotations - all segments and angles
    annotations = [
        # Arms - segments
        ("L Hum: {:.1f}".format(m["L Humerus"]), jointsL_px["Lelbow"], (0, 255, 0)),
        ("R Hum: {:.1f}".format(m["R Humerus"]), jointsL_px["Relbow"], (0, 255, 0)),
        ("L Rad: {:.1f}".format(m["L Radius"]), jointsL_px["Lwrist"], (0, 200, 0)),
        ("R Rad: {:.1f}".format(m["R Radius"]), jointsL_px["Rwrist"], (0, 200, 0)),
        ("L Arm: {:.1f}cm".format(m["L Arm Total"]), (jointsL_px["Lwrist"][0]+20, jointsL_px["Lwrist"][1]+20), (0, 255, 0)),
        ("R Arm: {:.1f}cm".format(m["R Arm Total"]), (jointsL_px["Rwrist"][0]+20, jointsL_px["Rwrist"][1]+20), (0, 255, 0)),
        # Legs - segments
        ("L Fem: {:.1f}".format(m["L Femur"]), jointsL_px["Lknee"], (255, 0, 0)),
        ("R Fem: {:.1f}".format(m["R Femur"]), jointsL_px["Rknee"], (255, 0, 0)),
        ("L Tib: {:.1f}".format(m["L Tibia"]), jointsL_px["Lankle"], (200, 0, 0)),
        ("R Tib: {:.1f}".format(m["R Tibia"]), jointsL_px["Rankle"], (200, 0, 0)),
        ("L Leg: {:.1f}cm".format(m["L Leg Total"]), (jointsL_px["Lankle"][0]+20, jointsL_px["Lankle"][1]+20), (255, 0, 0)),
        ("R Leg: {:.1f}cm".format(m["R Leg Total"]), (jointsL_px["Rankle"][0]+20, jointsL_px["Rankle"][1]+20), (255, 0, 0)),
        # Joint angles
        ("L Shldr: {:.0f}°".format(m["L Shoulder Angle"]), (jointsL_px["Lshoulder"][0]-60, jointsL_px["Lshoulder"][1]), (0, 255, 255)),
        ("R Shldr: {:.0f}°".format(m["R Shoulder Angle"]), (jointsL_px["Rshoulder"][0]+20, jointsL_px["Rshoulder"][1]), (0, 255, 255)),
        ("L Elb: {:.0f}°".format(m["L Elbow Angle"]), (jointsL_px["Lelbow"][0]-60, jointsL_px["Lelbow"][1]), (0, 255, 255)),
        ("R Elb: {:.0f}°".format(m["R Elbow Angle"]), (jointsL_px["Relbow"][0]+20, jointsL_px["Relbow"][1]), (0, 255, 255)),
        ("L Hip: {:.0f}°".format(m["L Hip Angle"]), (jointsL_px["Lhip"][0]-60, jointsL_px["Lhip"][1]), (255, 255, 0)),
        ("R Hip: {:.0f}°".format(m["R Hip Angle"]), (jointsL_px["Rhip"][0]+20, jointsL_px["Rhip"][1]), (255, 255, 0)),
        ("L Knee: {:.0f}°".format(m["L Knee Angle"]), (jointsL_px["Lknee"][0]-60, jointsL_px["Lknee"][1]), (255, 255, 0)),
        ("R Knee: {:.0f}°".format(m["R Knee Angle"]), (jointsL_px["Rknee"][0]+20, jointsL_px["Rknee"][1]), (255, 255, 0)),
        # Torso
        ("Torso: {:.1f}cm".format(m["Torso Length"]), (jointsL_px["Lshoulder"][0], jointsL_px["Lshoulder"][1]+100), (255, 255, 255)),
        ("Height: {:.1f}cm".format(m["Height (Segments)"]), (50, 50), (255, 255, 255)),
    ]
    for text, pos, color in annotations:
        cv2.putText(dispL, text, (pos[0]+5, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    pose_2d_path = os.path.join(out_dir, f"body_2d_annotated_{ts}.png")
    cv2.imwrite(pose_2d_path, dispL)
    print(f"\n2D annotated image saved: {pose_2d_path}")

    # Create 3D plot with CORRECTED axes for standing pose
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Full Body 3D Skeleton (Standing) - HRNet")
    
    # SWAP AXES: Plot Z (depth) as X, X (width) as Y, Y (height) as Z to show standing
    ax.set_xlabel("X (Depth)")
    ax.set_ylabel("Y (Width)")
    ax.set_zlabel("Z (Height)")

    # Plot skeleton with SWAPPED axes: (Z, X, Y) -> (depth, width, height)
    ax.scatter(pts3d_world[:,2], pts3d_world[:,0], pts3d_world[:,1], c='r', s=20)
    for conn in COCO_CONNECTIONS:
        v1, v2 = pts3d_world[conn[0]], pts3d_world[conn[1]]
        # Plot as (Z, X, Y) to show standing
        ax.plot([v1[2], v2[2]], [v1[0], v2[0]], [v1[1], v2[1]], color='blue', alpha=0.6)

    # Equal scale with swapped axes
    padding = 0.2
    z_min, z_max = np.min(pts3d_world[:,2]), np.max(pts3d_world[:,2])  # depth -> X
    x_min, x_max = np.min(pts3d_world[:,0]), np.max(pts3d_world[:,0])  # width -> Y
    y_min, y_max = np.min(pts3d_world[:,1]), np.max(pts3d_world[:,1])  # height -> Z
    
    z_range = z_max-z_min
    x_range = x_max-x_min
    y_range = y_max-y_min
    max_range = max(x_range, y_range, z_range)
    padded_range = max_range * (1 + 2 * padding)
    
    z_mid, x_mid, y_mid = (z_min+z_max)/2, (x_min+x_max)/2, (y_min+y_max)/2
    half_range = padded_range / 2.0
    
    ax.set_xlim(z_mid - half_range, z_mid + half_range)
    ax.set_ylim(x_mid - half_range, x_mid + half_range)
    ax.set_zlim(y_mid - half_range, y_mid + half_range)
    ax.set_box_aspect([1,1,1])
    
    # Set view angle for 3D perspective (3/4 view from front-right)
    ax.view_init(elev=15, azim=45)
    
    plot_3d_path = os.path.join(out_dir, f"body_3d_standing_{ts}.png")
    plt.savefig(plot_3d_path, dpi=150, bbox_inches='tight')
    print(f"3D standing plot saved: {plot_3d_path}")
    plt.close()

    # Comprehensive CSV with ALL measurements
    csv_path = os.path.join(out_dir, f"body_measurements_complete_{ts}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Measurement', 'Left', 'Right', 'Difference', 'Unit'])
        
        # Segment lengths
        writer.writerow(['Segment', 'Humerus (Upper Arm)', f"{m['L Humerus']:.2f}", f"{m['R Humerus']:.2f}", f"{abs(m['L Humerus']-m['R Humerus']):.2f}", 'cm'])
        writer.writerow(['Segment', 'Radius (Forearm)', f"{m['L Radius']:.2f}", f"{m['R Radius']:.2f}", f"{abs(m['L Radius']-m['R Radius']):.2f}", 'cm'])
        writer.writerow(['Segment', 'Arm Total', f"{m['L Arm Total']:.2f}", f"{m['R Arm Total']:.2f}", f"{abs(m['L Arm Total']-m['R Arm Total']):.2f}", 'cm'])
        writer.writerow(['Segment', 'Femur (Thigh)', f"{m['L Femur']:.2f}", f"{m['R Femur']:.2f}", f"{abs(m['L Femur']-m['R Femur']):.2f}", 'cm'])
        writer.writerow(['Segment', 'Tibia (Shin)', f"{m['L Tibia']:.2f}", f"{m['R Tibia']:.2f}", f"{abs(m['L Tibia']-m['R Tibia']):.2f}", 'cm'])
        writer.writerow(['Segment', 'Leg Total', f"{m['L Leg Total']:.2f}", f"{m['R Leg Total']:.2f}", f"{abs(m['L Leg Total']-m['R Leg Total']):.2f}", 'cm'])
        
        # Joint angles
        writer.writerow(['Angle', 'Shoulder', f"{m['L Shoulder Angle']:.1f}", f"{m['R Shoulder Angle']:.1f}", f"{abs(m['L Shoulder Angle']-m['R Shoulder Angle']):.1f}", 'deg'])
        writer.writerow(['Angle', 'Elbow', f"{m['L Elbow Angle']:.1f}", f"{m['R Elbow Angle']:.1f}", f"{abs(m['L Elbow Angle']-m['R Elbow Angle']):.1f}", 'deg'])
        writer.writerow(['Angle', 'Hip', f"{m['L Hip Angle']:.1f}", f"{m['R Hip Angle']:.1f}", f"{abs(m['L Hip Angle']-m['R Hip Angle']):.1f}", 'deg'])
        writer.writerow(['Angle', 'Knee', f"{m['L Knee Angle']:.1f}", f"{m['R Knee Angle']:.1f}", f"{abs(m['L Knee Angle']-m['R Knee Angle']):.1f}", 'deg'])
        
        # Body dimensions
        writer.writerow(['Body', 'Shoulder Width', f"{m['Shoulder Width']:.2f}", '-', '-', 'cm'])
        writer.writerow(['Body', 'Hip Width', f"{m['Hip Width']:.2f}", '-', '-', 'cm'])
        writer.writerow(['Body', 'Torso Length', f"{m['Torso Length']:.2f}", '-', '-', 'cm'])
        writer.writerow(['Body', 'Height (Proxy)', f"{m['Height (Proxy)']:.2f}", '-', '-', 'cm'])
        writer.writerow(['Body', 'Height (Segments)', f"{m['Height (Segments)']:.2f}", '-', '-', 'cm'])
    
    print(f"CSV comparison saved: {csv_path}")

    # Text report
    out_path = os.path.join(out_dir, f"body_3d_report_{ts}.txt")
    with open(out_path, 'w') as f:
        f.write(f"Left: {img_left_path}\nRight: {img_right_path}\n")
        f.write(f"Model: HRNet\nTimestamp: {ts}\n" + "="*40 + "\n")
        for name, val in m.items():
            unit = "deg" if "Angle" in name else "cm"
            f.write(f"{name}: {val:.2f} {unit}\n")
    print(f"Text report saved: {out_path}")
    print(f"\n✓ All outputs saved to: {out_dir}/")

if __name__ == "__main__":
    main()
