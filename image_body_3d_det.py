# image_body_3d_det.py
# Detailed Biomechanical Analysis with Triangle-based Measurements
# Supports both MediaPipe and HRNet models
# Calculates all segment lengths and joint angles with height-based scaling

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import csv
import argparse
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
MIN_AXIS_LIMIT = 600  # units: mm
HRNET_PATH = "/media/raviappala/edgeextvol2/jetson-inference/3dpose/HRnet_0.1"

# ---------------------------
# Load stereo params
# ---------------------------
PARAMS_FILE = "stereo_params_sidebyside.npz"
if not os.path.exists(PARAMS_FILE):
    raise RuntimeError(f"Missing calibration file: {PARAMS_FILE}")

data = np.load(PARAMS_FILE)
K1, dist1 = data["K1"], data["dist1"]
K2, dist2 = data["K2"], data["dist2"]
P1, P2 = data["P1"], data["P2"]
R_stereo = data.get("R", None)
T_stereo = data.get("T", None)
if R_stereo is None or T_stereo is None:
    raise RuntimeError("stereo_params.npz must contain R and T")
T_stereo = T_stereo.reshape(3,)

print("=== CALIBRATION PARAMS ===")
print(f"Baseline: {np.linalg.norm(T_stereo):.2f} mm")

# ---------------------------
# Model Loading Functions
# ---------------------------
def load_mediapipe():
    """Load MediaPipe pose model"""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    
    LM = {
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
    return pose, LM, "mediapipe"

def load_hrnet():
    """Load HRNet pose model"""
    sys.path.insert(0, os.path.join(HRNET_PATH, 'demo'))
    import torch
    import torchvision
    import _init_paths
    import models
    from config import cfg, update_config
    
    # Load config
    config_file = os.path.join(HRNET_PATH, 'demo/inference-config.yaml')
    update_config(cfg, config_file)
    
    CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load person detection
    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    box_model.to(CTX)
    box_model.eval()
    
    # Load HRNet pose model
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    model_file = os.path.join(HRNET_PATH, 'models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth')
    state_dict = torch.load(model_file, map_location=CTX)
    pose_model.load_state_dict(state_dict)
    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()
    
    # COCO 17 keypoint mapping to body landmarks
    LM = {
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
    
    return (box_model, pose_model, cfg, CTX), LM, "hrnet"

def detect_pose_mediapipe(model, frameL, frameR, LM):
    """Detect pose using MediaPipe"""
    import mediapipe as mp
    pose = model
    
    resL = pose.process(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB))
    resR = pose.process(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB))
    
    if not resL.pose_landmarks or not resR.pose_landmarks:
        return None, None
    
    hL, wL = frameL.shape[:2]
    hR, wR = frameR.shape[:2]
    
    kp_left = {}
    kp_right = {}
    
    for name, idx in LM.items():
        lm_l = resL.pose_landmarks.landmark[idx]
        lm_r = resR.pose_landmarks.landmark[idx]
        kp_left[name] = np.array([lm_l.x * wL, lm_l.y * hL])
        kp_right[name] = np.array([lm_r.x * wR, lm_r.y * hR])
    
    return kp_left, kp_right

def detect_pose_hrnet(models, frameL, frameR, LM):
    """Detect pose using HRNet"""
    from lib.core.inference import get_final_preds
    from lib.utils.transforms import get_affine_transform
    import torch
    import torchvision.transforms as transforms
    
    box_model, pose_model, cfg, CTX = models
    
    def get_person_boxes(model, img_list, threshold=0.7):
        with torch.no_grad():
            pred = model(img_list)
        boxes = []
        for p in pred:
            for i, score in enumerate(p['scores']):
                if score > threshold and p['labels'][i] == 1:
                    boxes.append(p['boxes'][i].cpu().numpy())
        return boxes
    
    def box_to_center_scale(box, model_image_width, model_image_height):
        x, y, x2, y2 = box
        w, h = x2 - x, y2 - y
        center = np.array([x + w/2, y + h/2])
        aspect_ratio = model_image_width / model_image_height
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w / 200.0, h / 200.0])
        return center, scale
    
    def get_pose_estimation(model, image, center, scale, cfg):
        trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
        input_img = cv2.warpAffine(image, trans, tuple(cfg.MODEL.IMAGE_SIZE), flags=cv2.INTER_LINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(input_img).unsqueeze(0).to(CTX)
        with torch.no_grad():
            output = model(input_tensor)
        preds, _ = get_final_preds(cfg, output.cpu().numpy(), np.array([center]), np.array([scale]))
        return preds[0]
    
    # Process both images
    transform = transforms.Compose([transforms.ToTensor()])
    imgL_tensor = transform(frameL).to(CTX)
    imgR_tensor = transform(frameR).to(CTX)
    
    # Detect persons
    boxesL = get_person_boxes(box_model, [imgL_tensor], threshold=0.7)
    boxesR = get_person_boxes(box_model, [imgR_tensor], threshold=0.7)
    
    if len(boxesL) == 0 or len(boxesR) == 0:
        return None, None
    
    # Get poses
    centerL, scaleL = box_to_center_scale(boxesL[0], cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
    centerR, scaleR = box_to_center_scale(boxesR[0], cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
    
    poseL = get_pose_estimation(pose_model, frameL, centerL, scaleL, cfg)
    poseR = get_pose_estimation(pose_model, frameR, centerR, scaleR, cfg)
    
    kp_left = {name: poseL[idx] for name, idx in LM.items()}
    kp_right = {name: poseR[idx] for name, idx in LM.items()}
    
    return kp_left, kp_right

# ---------------------------
# Triangulation & Transform
# ---------------------------
def triangulate_point(ptL, ptR):
    ptsL = np.array([[ptL[0]], [ptL[1]]], dtype=np.float64)
    ptsR = np.array([[ptR[0]], [ptR[1]]], dtype=np.float64)
    X_h = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
    X = (X_h[:3] / X_h[3]).reshape(3,)
    return X

def build_world_transform(R, T):
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
        raise RuntimeError("Baseline too small")
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
# Measurement Functions
# ---------------------------
def dist(p1, p2):
    """3D distance in cm"""
    return np.linalg.norm(p1 - p2) / 10.0

def get_angle_deg(v1, v2):
    """Angle between two vectors in degrees"""
    v1_n = v1 / (np.linalg.norm(v1) + 1e-9)
    v2_n = v2 / (np.linalg.norm(v2) + 1e-9)
    dot = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    return np.arccos(dot) * 180.0 / np.pi

# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description='Detailed 3D biomechanical analysis')
    parser.add_argument('--num', '-n', type=str, default='',
                        help='Image number suffix')
    parser.add_argument('--height', type=float, default=None,
                        help='True height in cm (for scaling)')
    parser.add_argument('--model', '-m', type=str, default='mediapipe', choices=['mediapipe', 'hrnet'],
                        help='Pose detection model to use')
    args = parser.parse_args()
    
    # Build paths
    suffix = args.num if args.num else ''
    img_left_path = f"test_ss_left{suffix}.png"
    img_right_path = f"test_ss_right{suffix}.png"
    
    if not os.path.isfile(img_left_path) or not os.path.isfile(img_right_path):
        print(f"Error: Files not found: {img_left_path}, {img_right_path}")
        return
    
    print(f"\nProcessing:\n  Left: {img_left_path}\n  Right: {img_right_path}")
    print(f"  Model: {args.model.upper()}")
    
    frameL = cv2.imread(img_left_path)
    frameR = cv2.imread(img_right_path)
    
    # Load model
    print(f"\nLoading {args.model.upper()} model...")
    if args.model == 'mediapipe':
        model, LM, model_name = load_mediapipe()
        kp_left, kp_right = detect_pose_mediapipe(model, frameL, frameR, LM)
    else:  # hrnet
        model, LM, model_name = load_hrnet()
        kp_left, kp_right = detect_pose_hrnet(model, frameL, frameR, LM)
    
    if kp_left is None or kp_right is None:
        print("Error: Pose not detected in one or both images")
        return
    
    print("Running pose detection...")
    print(f"Detected {len(kp_left)} keypoints")
    
    # Triangulate all points
    pts3d_cam = []
    sources = []
    for name in sorted(LM.keys()):
        if name in kp_left and name in kp_right:
            X = triangulate_point(kp_left[name], kp_right[name])
            pts3d_cam.append(X)
            sources.append(name)
    
    pts3d_cam = np.array(pts3d_cam)
    
    # Transform to world coordinates
    pts3d_world = ((pts3d_cam - origin_world) @ R_worldcam.T)
    points = {name: pts3d_world[i] for i, name in enumerate(sources)}
    
    # Calculate basic measurements
    m = {}
    
    # Basic segments (existing measurements)
    if "Lshoulder" in points and "Rshoulder" in points:
        m["Shoulder Width"] = dist(points["Lshoulder"], points["Rshoulder"])
    if "Lhip" in points and "Rhip" in points:
        m["Hip Width"] = dist(points["Lhip"], points["Rhip"])
    if all(k in points for k in ["Lshoulder", "Rshoulder", "Lhip", "Rhip"]):
        m["Torso Length"] = dist((points["Lshoulder"]+points["Rshoulder"])/2,
                                  (points["Lhip"]+points["Rhip"])/2)
    
    # Arms
    if all(k in points for k in ["Lshoulder", "Lelbow"]):
        m["L Humerus"] = dist(points["Lshoulder"], points["Lelbow"])
    if all(k in points for k in ["Rshoulder", "Relbow"]):
        m["R Humerus"] = dist(points["Rshoulder"], points["Relbow"])
    if all(k in points for k in ["Lelbow", "Lwrist"]):
        m["L Radius"] = dist(points["Lelbow"], points["Lwrist"])
    if all(k in points for k in ["Relbow", "Rwrist"]):
        m["R Radius"] = dist(points["Relbow"], points["Rwrist"])
    
    # Legs
    if all(k in points for k in ["Lhip", "Lknee"]):
        m["L Femur"] = dist(points["Lhip"], points["Lknee"])
    if all(k in points for k in ["Rhip", "Rknee"]):
        m["R Femur"] = dist(points["Rhip"], points["Rknee"])
    if all(k in points for k in ["Lknee", "Lankle"]):
        m["L Tibia"] = dist(points["Lknee"], points["Lankle"])
    if all(k in points for k in ["Rknee", "Rankle"]):
        m["R Tibia"] = dist(points["Rknee"], points["Rankle"])
    
    # Triangle measurements (from the diagram)
    # Triangle 1 & 2: Head/Shoulders
    if all(k in points for k in ["nose", "Lshoulder", "Rshoulder"]):
        m["Head to L Shoulder"] = dist(points["nose"], points["Lshoulder"])
        m["Head to R Shoulder"] = dist(points["nose"], points["Rshoulder"])
    
    # Triangle 3: Torso/Hips
    if all(k in points for k in ["Lshoulder", "Lhip", "Rhip"]):
        m["L Shoulder to L Hip"] = dist(points["Lshoulder"], points["Lhip"])
        m["L Shoulder to R Hip"] = dist(points["Lshoulder"], points["Rhip"])
    if all(k in points for k in ["Rshoulder", "Lhip", "Rhip"]):
        m["R Shoulder to R Hip"] = dist(points["Rshoulder"], points["Rhip"])
        m["R Shoulder to L Hip"] = dist(points["Rshoulder"], points["Lhip"])
    
    # Triangle 4: Knees
    if all(k in points for k in ["Lhip", "Rhip", "Lknee", "Rknee"]):
        m["L Hip to R Knee"] = dist(points["Lhip"], points["Rknee"])
        m["R Hip to L Knee"] = dist(points["Rhip"], points["Lknee"])
    
    # Triangle 5 & 6: Feet
    if all(k in points for k in ["Lknee", "Lankle"]):
        m["L Knee to Ankle"] = m.get("L Tibia", dist(points["Lknee"], points["Lankle"]))
    if all(k in points for k in ["Rknee", "Rankle"]):
        m["R Knee to Ankle"] = m.get("R Tibia", dist(points["Rknee"], points["Rankle"]))
    
    # Additional MediaPipe-specific segments
    if "Lheel" in points and "Lankle" in points:
        m["L Ankle to Heel"] = dist(points["Lankle"], points["Lheel"])
    if "Rheel" in points and "Rankle" in points:
        m["R Ankle to Heel"] = dist(points["Rankle"], points["Rheel"])
    if "Lheel" in points and "Lfoot" in points:
        m["L Heel to Toe"] = dist(points["Lheel"], points["Lfoot"])
    if "Rheel" in points and "Rfoot" in points:
        m["R Heel to Toe"] = dist(points["Rheel"], points["Rfoot"])
    
    # Hand segments (MediaPipe only)
    if "Lwrist" in points and "Lindex" in points:
        m["L Hand Length"] = dist(points["Lwrist"], points["Lindex"])
    if "Rwrist" in points and "Rindex" in points:
        m["R Hand Length"] = dist(points["Rwrist"], points["Rindex"])
    
    # Calculate ALL joint angles
    angles = {}
    
    # Shoulder angles
    if all(k in points for k in ["Lshoulder", "Lelbow", "Lhip"]):
        v1 = points["Lelbow"] - points["Lshoulder"]
        v2 = points["Lhip"] - points["Lshoulder"]
        angles["L Shoulder"] = get_angle_deg(v1, v2)
    if all(k in points for k in ["Rshoulder", "Relbow", "Rhip"]):
        v1 = points["Relbow"] - points["Rshoulder"]
        v2 = points["Rhip"] - points["Rshoulder"]
        angles["R Shoulder"] = get_angle_deg(v1, v2)
    
    # Elbow angles
    if all(k in points for k in ["Lshoulder", "Lelbow", "Lwrist"]):
        v1 = points["Lshoulder"] - points["Lelbow"]
        v2 = points["Lwrist"] - points["Lelbow"]
        angles["L Elbow"] = get_angle_deg(v1, v2)
    if all(k in points for k in ["Rshoulder", "Relbow", "Rwrist"]):
        v1 = points["Rshoulder"] - points["Relbow"]
        v2 = points["Rwrist"] - points["Relbow"]
        angles["R Elbow"] = get_angle_deg(v1, v2)
    
    # Hip angles
    if all(k in points for k in ["Lshoulder", "Lhip", "Lknee"]):
        v1 = points["Lshoulder"] - points["Lhip"]
        v2 = points["Lknee"] - points["Lhip"]
        angles["L Hip"] = get_angle_deg(v1, v2)
    if all(k in points for k in ["Rshoulder", "Rhip", "Rknee"]):
        v1 = points["Rshoulder"] - points["Rhip"]
        v2 = points["Rknee"] - points["Rhip"]
        angles["R Hip"] = get_angle_deg(v1, v2)
    
    # Knee angles
    if all(k in points for k in ["Lhip", "Lknee", "Lankle"]):
        v1 = points["Lhip"] - points["Lknee"]
        v2 = points["Lankle"] - points["Lknee"]
        angles["L Knee"] = get_angle_deg(v1, v2)
    if all(k in points for k in ["Rhip", "Rknee", "Rankle"]):
        v1 = points["Rhip"] - points["Rknee"]
        v2 = points["Rankle"] - points["Rknee"]
        angles["R Knee"] = get_angle_deg(v1, v2)
    
    # Ankle angles (MediaPipe only)
    if all(k in points for k in ["Lknee", "Lankle", "Lheel"]):
        v1 = points["Lknee"] - points["Lankle"]
        v2 = points["Lheel"] - points["Lankle"]
        angles["L Ankle"] = get_angle_deg(v1, v2)
    if all(k in points for k in ["Rknee", "Rankle", "Rheel"]):
        v1 = points["Rknee"] - points["Rankle"]
        v2 = points["Rheel"] - points["Rankle"]
        angles["R Ankle"] = get_angle_deg(v1, v2)
    
    # Height estimation
    if "Lankle" in points and "Rankle" in points:
        floor_y = (points["Lankle"][1] + points["Rankle"][1]) / 2.0
    elif "Lheel" in points and "Rheel" in points:
        floor_y = (points["Lheel"][1] + points["Rheel"][1]) / 2.0
    else:
        floor_y = 0
    
    leg_avg = (m.get("L Femur", 0) + m.get("L Tibia", 0) + m.get("R Femur", 0) + m.get("R Tibia", 0)) / 2.0
    if "nose" in points and all(k in points for k in ["Lshoulder", "Rshoulder"]):
        neck_head = dist(points["nose"], (points["Lshoulder"]+points["Rshoulder"])/2) + 15.0
    else:
        neck_head = 20.0
    
    measured_height = leg_avg + m.get("Torso Length", 0) + neck_head
    m["Height (Segments)"] = measured_height
    
    # Apply height scaling if provided
    if args.height is not None:
        scale_factor = args.height / measured_height
        print(f"\n🎯 HEIGHT SCALING ENABLED:")
        print(f"   Measured height: {measured_height:.1f} cm")
        print(f"   Provided height: {args.height:.1f} cm")
        print(f"   Scale factor: {scale_factor:.4f}")
        
        # Scale all 3D points
        pts3d_world = pts3d_world * scale_factor
        points = {name: pts3d_world[i] for i, name in enumerate(sources)}
        
        # Recalculate all measurements
        for key in list(m.keys()):
            if key != "Height (Segments)":
                m[key] = m[key] * scale_factor
        
        m["Height (Segments)"] = args.height
    
    # Print results
    print("\n" + "="*60)
    print(f"DETAILED BIOMECHANICAL ANALYSIS ({model_name.upper()})")
    print("="*60)
    
    print("\n--- SEGMENT LENGTHS (cm) ---")
    for key in sorted(m.keys()):
        print(f"{key:.<30} {m[key]:>8.2f} cm")
    
    print("\n--- JOINT ANGLES (degrees) ---")
    for key in sorted(angles.keys()):
        print(f"{key:.<30} {angles[key]:>8.2f}°")
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = f"biomech_analysis_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = f"{output_dir}/measurements_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Measurement Type", "Name", "Value", "Unit"])
        for key, val in sorted(m.items()):
            writer.writerow(["Length", key, f"{val:.2f}", "cm"])
        for key, val in sorted(angles.items()):
            writer.writerow(["Angle", key, f"{val:.2f}", "degrees"])
    
    print(f"\n✓ CSV saved: {csv_path}")
    print(f"✓ All outputs in: {output_dir}/")

if __name__ == "__main__":
    main()
