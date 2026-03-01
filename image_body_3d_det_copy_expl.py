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
    # Add HRNet paths to sys.path
    sys.path.insert(0, HRNET_PATH)
    sys.path.insert(0, os.path.join(HRNET_PATH, 'demo'))
    sys.path.insert(0, os.path.join(HRNET_PATH, 'lib'))
    
    import torch
    import torchvision
    import _init_paths
    import models
    from config import cfg, update_config
    from lib.core.inference import get_final_preds
    from lib.utils.transforms import get_affine_transform
    
    # Load config
    config_file = os.path.join(HRNET_PATH, 'demo/inference-config.yaml')
    # Create args object with all attributes expected by update_config
    class Args:
        def __init__(self):
            self.cfg = config_file
            self.opts = []  # Empty list for additional options
            self.modelDir = ''  # Empty string means use default from config
            self.logDir = ''    # Empty string means use default from config
            self.dataDir = ''   # Empty string means use default from config
    args = Args()
    update_config(cfg, args)
    
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
    
    # Return models and lib utilities
    return (box_model, pose_model, cfg, CTX, get_final_preds, get_affine_transform), LM, "hrnet"

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
    import torch
    import torchvision.transforms as transforms
    
    # Unpack models and utilities
    box_model, pose_model, cfg, CTX, get_final_preds, get_affine_transform = models
    
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
    parser = argparse.ArgumentParser(description='3D Biomechanical Analysis')
    parser.add_argument('--num', '-n', type=str, default='', help='Image number suffix')
    parser.add_argument('--height', type=float, default=None, help='True height in cm for scaling')
    parser.add_argument('--model', '-m', type=str, default='mediapipe', choices=['mediapipe', 'hrnet'])
    parser.add_argument('--verbose', '-v', action='store_true', help='Show scale factor and calculation details')
    args = parser.parse_args()

    # ------------------------- Load Calibration -----------------------------
    PARAMS_FILE = "stereo_params_sidebyside.npz"
    if not os.path.exists(PARAMS_FILE):
        raise RuntimeError(f"Missing calibration file: {PARAMS_FILE}")

    data = np.load(PARAMS_FILE)
    K1, dist1 = data["K1"], data["dist1"]
    K2, dist2 = data["K2"], data["dist2"]
    P1, P2 = data["P1"], data["P2"]
    R_stereo, T_stereo = data.get("R", None), data.get("T", None)
    if R_stereo is None or T_stereo is None:
        raise RuntimeError("stereo_params_sidebyside.npz must contain R and T")
    T_stereo = T_stereo.reshape(3,)

    # Build world transform
    R_worldcam, origin_world, baseline_len = build_world_transform(R_stereo, T_stereo)
    cam_left_pos = -origin_world
    cam_right_pos = T_stereo - origin_world

    print(f"=== CALIBRATION PARAMS ===\nBaseline: {baseline_len:.2f} mm")

    # ------------------------- Load Images ---------------------------------
    suffix = args.num if args.num else ''
    img_left_path = f"test_ss_left{suffix}.png"
    img_right_path = f"test_ss_right{suffix}.png"

    if not os.path.isfile(img_left_path) or not os.path.isfile(img_right_path):
        print(f"Error: Files not found: {img_left_path}, {img_right_path}")
        return

    frameL = cv2.imread(img_left_path)
    frameR = cv2.imread(img_right_path)

    # ------------------------- Load Pose Model -----------------------------
    if args.model == 'mediapipe':
        model, LM, model_name = load_mediapipe()
        kp_left, kp_right = detect_pose_mediapipe(model, frameL, frameR, LM)
    else:
        model, LM, model_name = load_hrnet()
        kp_left, kp_right = detect_pose_hrnet(model, frameL, frameR, LM)

    if kp_left is None or kp_right is None:
        print("Error: Pose not detected in one or both images")
        return

    # ------------------------- Triangulate Points --------------------------
    pts3d_cam = []
    sources = []
    for name in sorted(LM.keys()):
        if name in kp_left and name in kp_right:
            X = triangulate_point(kp_left[name], kp_right[name])
            pts3d_cam.append(X)
            sources.append(name)
    pts3d_cam = np.array(pts3d_cam)
    
    # Transform to world frame (matching image_body_3d.py approach)
    pts3d_world = []
    for p_cam in pts3d_cam:
        p_rel = p_cam - origin_world
        p_w = R_worldcam @ p_rel  # Correct: multiply on LEFT without transpose
        pts3d_world.append(p_w)
    pts3d_world = np.array(pts3d_world)
    
    # Debug: Print sample coordinates
    if args.verbose:
        print(f"\n=== DEBUG: Coordinate Transformation ===")
        print(f"Sample camera coords (nose): {pts3d_cam[sources.index('nose')] if 'nose' in sources else 'N/A'}")
        print(f"Sample world coords (nose): {pts3d_world[sources.index('nose')] if 'nose' in sources else 'N/A'}")
        print(f"Origin: {origin_world}")
        print(f"Baseline: {np.linalg.norm(T_stereo):.2f} mm")
        print("=" * 40)
    
    points = {name: pts3d_world[i] for i, name in enumerate(sources)}

    # ------------------------- Calculate Segment Lengths -------------------
    m = {}
    # Example: Torso
    if all(k in points for k in ["Lshoulder", "Rshoulder"]):
        m["Shoulder Width"] = dist(points["Lshoulder"], points["Rshoulder"])
    if all(k in points for k in ["Lhip", "Rhip"]):
        m["Hip Width"] = dist(points["Lhip"], points["Rhip"])
    if all(k in points for k in ["Lshoulder","Rshoulder","Lhip","Rhip"]):
        m["Torso Length"] = dist((points["Lshoulder"]+points["Rshoulder"])/2,
                                 (points["Lhip"]+points["Rhip"])/2)

    # Arms & Legs
    limb_pairs = [
        ("Lshoulder","Lelbow","L Humerus"), ("Rshoulder","Relbow","R Humerus"),
        ("Lelbow","Lwrist","L Radius"), ("Relbow","Rwrist","R Radius"),
        ("Lhip","Lknee","L Femur"), ("Rhip","Rknee","R Femur"),
        ("Lknee","Lankle","L Tibia"), ("Rknee","Rankle","R Tibia")
    ]
    for a,b,name in limb_pairs:
        if a in points and b in points:
            m[name] = dist(points[a], points[b])

    # MediaPipe-specific foot/hand points
    extra_pairs = [
        ("Lankle","Lheel","L Ankle to Heel"), ("Rankle","Rheel","R Ankle to Heel"),
        ("Lheel","Lfoot","L Heel to Toe"), ("Rheel","Rfoot","R Heel to Toe"),
        ("Lwrist","Lindex","L Hand Length"), ("Rwrist","Rindex","R Hand Length")
    ]
    for a,b,name in extra_pairs:
        if a in points and b in points:
            m[name] = dist(points[a], points[b])

    # ------------------------- Calculate Joint Angles ---------------------
    angles = {}
    angle_triplets = [
        ("Lshoulder","Lelbow","Lhip","L Shoulder"),
        ("Rshoulder","Relbow","Rhip","R Shoulder"),
        ("Lshoulder","Lelbow","Lwrist","L Elbow"),
        ("Rshoulder","Relbow","Rwrist","R Elbow"),
        ("Lshoulder","Lhip","Lknee","L Hip"),
        ("Rshoulder","Rhip","Rknee","R Hip"),
        ("Lhip","Lknee","Lankle","L Knee"),
        ("Rhip","Rknee","Rankle","R Knee"),
        ("Lknee","Lankle","Lheel","L Ankle"),
        ("Rknee","Rankle","Rheel","R Ankle")
    ]
    for a,b,c,name in angle_triplets:
        if all(k in points for k in [a,b,c]):
            v1 = points[a] - points[b]
            v2 = points[c] - points[b]
            angles[name] = get_angle_deg(v1,v2)

    # ------------------------- Height Estimation --------------------------
    leg_avg = sum(m.get(k,0) for k in ["L Femur","L Tibia","R Femur","R Tibia"]) / 2
    torso = m.get("Torso Length",0)
    # neck_head = dist(points["nose"], (points["Lshoulder"]+points["Rshoulder"])/2) + 15.0 \
    #     if all(k in points for k in ["Lshoulder","Rshoulder","nose"]) else 20.0
    def estimate_head_top(points):
        """
        Estimate 3D head-top from nose, eyes, and ears
        points: dict of 3D points from triangulation
        returns: 3D coordinate of head-top
        """
        required = ["nose", "Leye", "Reye", "Lear", "Rear"]
        if not all(k in points for k in required):
            # fallback: 15 cm above nose
            return points.get("nose", np.array([0, 0, 0])) + np.array([0, 0, 150.0])  # 150mm = 15cm
        
        # Average eyes
        eye_avg = (points["Leye"] + points["Reye"]) / 2.0
        # Highest ear point in Z (assuming Z is vertical)
        ear_top_z = max(points["Lear"][2], points["Rear"][2])
        eye_top_z = max(eye_avg[2], ear_top_z)
        
        # Distance from nose to eye_avg (in mm, same as point coordinates)
        nose_to_eye = np.linalg.norm(points["nose"] - eye_avg)
        
        # Head-top estimate: extend above eye/ear by nose-to-eye distance
        head_top = eye_avg.copy()
        head_top[2] = eye_top_z + nose_to_eye  # vertical coordinate (all in mm)
        
        return head_top
    
    shoulder_mid = (points["Lshoulder"] + points["Rshoulder"]) / 2.0 if "Lshoulder" in points and "Rshoulder" in points else np.array([0,0,0])
    head_top = estimate_head_top(points)
    # Convert mm to cm using dist() function
    neck_head = dist(head_top, shoulder_mid)

    # --- Step 4: Final estimated height ---
    measured_height = leg_avg + torso + neck_head

    # Apply user height scaling
    scale_factor = 1.0
    original_measured_height = measured_height
    
    if args.height is not None:
        scale_factor = args.height / measured_height
        pts3d_world *= scale_factor
        points = {name: pts3d_world[i] for i,name in enumerate(sources)}
        for k in m.keys():
            m[k] *= scale_factor
        measured_height = args.height
        
        # Print scale factor details if verbose mode
        if args.verbose:
            print(f"\n=== HEIGHT SCALING ===")
            print(f"Calculated Height: {original_measured_height:.2f} cm")
            print(f"Provided Height: {args.height:.2f} cm")
            print(f"Scale Factor: {scale_factor:.4f}")
            print(f"Difference: {abs(args.height - original_measured_height):.2f} cm ({abs(1-scale_factor)*100:.2f}%)")
            print("=" * 40)

    # ------------------------- RMS Calculation ---------------------------
    # Calculate 2D pixel distance RMS between left and right keypoints
    rms = np.sqrt(np.mean([np.linalg.norm(kp_left[name] - kp_right[name])**2 
                           for name in kp_left if name in kp_right]))

    # Camera distance & angle
    cam_distance = np.linalg.norm(cam_left_pos - cam_right_pos)
    cam_vector = cam_right_pos - cam_left_pos
    cam_angle = np.degrees(np.arctan2(cam_vector[1], cam_vector[0]))

    # ------------------------- Save CSV -----------------------------------
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = f"biomech_analysis_{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/measurements_{timestamp}.csv"
    with open(csv_path,'w',newline='') as f:
        writer = csv.writer(f)

        # Table 1: Joints 3D coords - Organized by body part
        writer.writerow(["--- JOINT COORDINATES ---"])
        writer.writerow(["Body Part","Left X","Left Y","Left Z","Right X","Right Y","Right Z"])
        
        # Extract body part groups
        body_parts = {}
        for name in sources:
            # Determine if this is a left, right, or center part
            if name.startswith('L'):
                part_name = name[1:]  # Remove 'L' prefix
                if part_name not in body_parts:
                    body_parts[part_name] = {'left': None, 'right': None}
                body_parts[part_name]['left'] = pts3d_world[sources.index(name)]
            elif name.startswith('R'):
                part_name = name[1:]  # Remove 'R' prefix
                if part_name not in body_parts:
                    body_parts[part_name] = {'left': None, 'right': None}
                body_parts[part_name]['right'] = pts3d_world[sources.index(name)]
            else:
                # Center parts like nose
                if name not in body_parts:
                    body_parts[name] = {'left': None, 'right': None}
                body_parts[name]['left'] = pts3d_world[sources.index(name)]
        
        # Write organized data
        for part_name in sorted(body_parts.keys()):
            left_pt = body_parts[part_name]['left']
            right_pt = body_parts[part_name]['right']
            
            row = [part_name]
            
            # Add left coordinates
            if left_pt is not None:
                row.extend([f"{left_pt[0]:.2f}", f"{left_pt[1]:.2f}", f"{left_pt[2]:.2f}"])
            else:
                row.extend(["", "", ""])
            
            # Add right coordinates
            if right_pt is not None:
                row.extend([f"{right_pt[0]:.2f}", f"{right_pt[1]:.2f}", f"{right_pt[2]:.2f}"])
            else:
                row.extend(["", "", ""])
            
            writer.writerow(row)

        # Table 2: Segment lengths - Organized by body part
        writer.writerow(["\n--- SEGMENT LENGTHS (cm) ---"])
        writer.writerow(["Segment","Left Length","Right Length","Difference"])
        
        # Group segments by body part
        segment_groups = {}
        for name, value in m.items():
            # Check if this is a left/right segment
            if name.startswith('L '):
                part_name = name[2:]  # Remove 'L ' prefix
                if part_name not in segment_groups:
                    segment_groups[part_name] = {'left': None, 'right': None}
                segment_groups[part_name]['left'] = value
            elif name.startswith('R '):
                part_name = name[2:]  # Remove 'R ' prefix
                if part_name not in segment_groups:
                    segment_groups[part_name] = {'left': None, 'right': None}
                segment_groups[part_name]['right'] = value
            else:
                # Center/bilateral measurements like "Shoulder Width", "Hip Width", "Torso Length"
                if name not in segment_groups:
                    segment_groups[name] = {'left': None, 'right': None}
                segment_groups[name]['left'] = value
        
        # Write organized data
        for part_name in sorted(segment_groups.keys()):
            left_val = segment_groups[part_name]['left']
            right_val = segment_groups[part_name]['right']
            
            row = [part_name]
            
            # Add left length
            if left_val is not None:
                row.append(f"{left_val:.2f}")
            else:
                row.append("")
            
            # Add right length
            if right_val is not None:
                row.append(f"{right_val:.2f}")
            else:
                row.append("")
            
            # Calculate difference
            if left_val is not None and right_val is not None:
                diff = abs(left_val - right_val)
                row.append(f"{diff:.2f}")
            else:
                row.append("")
            
            writer.writerow(row)

        # Table 3: Angles - Organized by joint
        writer.writerow(["\n--- JOINT ANGLES (degrees) ---"])
        writer.writerow(["Joint","Left Angle","Right Angle","Difference"])
        
        # Group angles by joint
        angle_groups = {}
        for name, value in angles.items():
            # Check if this is a left/right angle
            if name.startswith('L '):
                joint_name = name[2:]  # Remove 'L ' prefix
                if joint_name not in angle_groups:
                    angle_groups[joint_name] = {'left': None, 'right': None}
                angle_groups[joint_name]['left'] = value
            elif name.startswith('R '):
                joint_name = name[2:]  # Remove 'R ' prefix
                if joint_name not in angle_groups:
                    angle_groups[joint_name] = {'left': None, 'right': None}
                angle_groups[joint_name]['right'] = value
            else:
                # Center joints (if any)
                if name not in angle_groups:
                    angle_groups[name] = {'left': None, 'right': None}
                angle_groups[name]['left'] = value
        
        # Write organized data
        for joint_name in sorted(angle_groups.keys()):
            left_angle = angle_groups[joint_name]['left']
            right_angle = angle_groups[joint_name]['right']
            
            row = [joint_name]
            
            # Add left angle
            if left_angle is not None:
                row.append(f"{left_angle:.2f}")
            else:
                row.append("")
            
            # Add right angle
            if right_angle is not None:
                row.append(f"{right_angle:.2f}")
            else:
                row.append("")
            
            # Calculate difference
            if left_angle is not None and right_angle is not None:
                diff = abs(left_angle - right_angle)
                row.append(f"{diff:.2f}")
            else:
                row.append("")
            
            writer.writerow(row)

        # Table 4: Parameters
        writer.writerow(["\n--- PARAMETERS ---"])
        writer.writerow(["Human Height Calculated", f"{measured_height:.2f}"])
        writer.writerow(["RMS Left-Right Keypoints", f"{rms:.2f}"])
        writer.writerow(["Distance Between Cameras", f"{cam_distance:.2f}"])
        writer.writerow(["Angle Between Cameras (deg)", f"{cam_angle:.2f}"])
        if args.height:
            writer.writerow(["Human Height Provided", f"{args.height:.2f}"])
            if args.verbose:
                writer.writerow(["Original Calculated Height", f"{original_measured_height:.2f}"])
                writer.writerow(["Scale Factor Applied", f"{scale_factor:.4f}"])
                writer.writerow(["Height Difference", f"{abs(args.height - original_measured_height):.2f}"])

    print(f"✓ CSV saved: {csv_path}")
    print(f"✓ All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()

