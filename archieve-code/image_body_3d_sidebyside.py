# image_body_3d_sidebyside.py
# 3D full body pose estimation from a stereo pair of static images
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import sys
import ssl
from datetime import datetime

# Bypass SSL verification for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# ---------------------------
# Config
# ---------------------------
WIDTH = 1280
HEIGHT = 720
MIN_AXIS_LIMIT = 600     # units: mm (from calibration)

# Test image paths
IMG_LEFT = "test_ss_left.png"
IMG_RIGHT = "test_ss_right.png"

# ---------------------------
# Load stereo params (from side-by-side calibration)
# ---------------------------
PARAMS_FILE = "stereo_params_sidebyside.npz"
if not os.path.exists(PARAMS_FILE):
    raise RuntimeError(f"Missing calibration file: {PARAMS_FILE}")

data = np.load(PARAMS_FILE)
K1 = data["K1"]
dist1 = data["dist1"]
K2 = data["K2"]
dist2 = data["dist2"]
R_stereo = data["R"]
T_stereo = data["T"]
img_size_calib = data.get("img_size", [2880, 1620]) # [width, height]
calib_w = img_size_calib[0]
calib_h = img_size_calib[1]

print(f"Loaded calibration for {calib_w}x{calib_h} resolution")

# Re-calculate rectification matrices to ensure consistency
# This is crucial for accurate triangulation
print("Calculating rectification matrices...")
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, dist1, K2, dist2, (calib_w, calib_h), R_stereo, T_stereo
)

# ---------------------------
# MediaPipe Pose init
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# All 33 landmarks for skeleton connection
BODY_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Landmark names for specific metrics
LM = {
    "nose": 0,
    "Lshoulder": 11, "Rshoulder": 12,
    "Lelbow": 13, "Relbow": 14,
    "Lwrist": 15, "Rwrist": 16,
    "Lhip": 23, "Rhip": 24,
    "Lknee": 25, "Rknee": 26,
    "Lankle": 27, "Rankle": 28,
    "Lheel": 29, "Rheel": 30,
    "Lfoot": 31, "Rfoot": 32
}

# ---------------------------
# Utilities
# ---------------------------
def undistort_rectify_px(pt, K, dist, R_rect, P_rect):
    """Map a point to rectified coordinates using calibration resolution."""
    pts = np.array(pt, dtype=np.float32).reshape(1,1,2)
    # R and P are essential to align the two cameras' image planes
    und = cv2.undistortPoints(pts, K, dist, R=R_rect, P=P_rect)
    u, v = und[0,0]
    return float(u), float(v)

def triangulate_landmarks(ptsL, ptsR):
    """Triangulate points from left and right rectified coords into 3D."""
    ptsL_arr = np.array(ptsL, dtype=np.float64).T
    ptsR_arr = np.array(ptsR, dtype=np.float64).T
    X_h = cv2.triangulatePoints(P1, P2, ptsL_arr, ptsR_arr)
    X = (X_h[:3] / X_h[3]).T
    return X # (33, 3)

def get_angle(v1, v2):
    """Compute internal angle in degrees between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 0.0
    cos = np.dot(v1, v2) / (n1 * n2)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def build_world_transform(R, T):
    """Build transform from left-cam to human-friendly world frame."""
    C1 = np.zeros(3)
    C2 = T.reshape(3,)
    origin = (C1 + C2) / 2.0
    
    left_up = np.array([0.0, -1.0, 0.0]) # -Y is up in camera coords
    right_up = R @ left_up
    avg_up = (left_up + right_up) / 2.0
    y_axis = avg_up / np.linalg.norm(avg_up)
    
    baseline = C2 - C1
    x_axis = baseline / np.linalg.norm(baseline)
    
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    
    # Correct Y to be orthogonal to X and Z
    y_axis = np.cross(z_axis, x_axis)
    
    R_wc = np.vstack([x_axis, y_axis, z_axis])
    return R_wc, origin

# ---------------------------
# Execution Logic
# ---------------------------
def main():
    if not os.path.exists(IMG_LEFT) or not os.path.exists(IMG_RIGHT):
        print(f"Error: Could not find {IMG_LEFT} or {IMG_RIGHT}")
        return

    # Load images
    frameL = cv2.imread(IMG_LEFT)
    frameR = cv2.imread(IMG_RIGHT)
    
    # Process with MediaPipe
    # WE DO NOT FLIP IMAGES for triangulation. Flipping breaks stereo disparity.
    print("Detecting landmarks with MediaPipe (High Complexity)...")
    resL = pose.process(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB))
    resR = pose.process(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB))

    if not resL.pose_landmarks or not resR.pose_landmarks:
        print("Error: Could not detect full body in one or both images.")
        return

    # Extract all 33 landmarks for triangulation
    lmL = resL.pose_landmarks.landmark
    lmR = resR.pose_landmarks.landmark
    
    ptsL_rect = []
    ptsR_rect = []
    
    for i in range(33):
        # Scale normalized coordinates to CALIBRATION resolution (2880x1620)
        uL = lmL[i].x * calib_w
        vL = lmL[i].y * calib_h
        ptsL_rect.append(undistort_rectify_px((uL, vL), K1, dist1, R1, P1))
        
        uR = lmR[i].x * calib_w
        vR = lmR[i].y * calib_h
        ptsR_rect.append(undistort_rectify_px((uR, vR), K2, dist2, R2, P2))

    # Triangulation
    pts3d_cam = triangulate_landmarks(ptsL_rect, ptsR_rect) # mm

    # Transform to world frame (midpoint origin, Y-up)
    R_wc, origin_w = build_world_transform(R_stereo, T_stereo)
    pts3d_world = np.array([R_wc @ (p - origin_w) for p in pts3d_cam])

    # Dictionary mapping name -> 3D position
    points = {name: pts3d_world[idx] for name, idx in LM.items()}

    # Calculate Body Metrics
    print("\nCalculating body metrics...")
    metrics = {}
    
    # Bone segments (mm to cm)
    segments = [
        ("Shoulder Width", points["Lshoulder"], points["Rshoulder"]),
        ("Hip Width", points["Lhip"], points["Rhip"]),
        ("Torso Length", (points["Lshoulder"]+points["Rshoulder"])/2, (points["Lhip"]+points["Rhip"])/2),
        ("L Humerus", points["Lshoulder"], points["Lelbow"]),
        ("R Humerus", points["Rshoulder"], points["Relbow"]),
        ("L Radius", points["Lelbow"], points["Lwrist"]),
        ("R Radius", points["Relbow"], points["Rwrist"]),
        ("L Femur", points["Lhip"], points["Lknee"]),
        ("R Femur", points["Rhip"], points["Rknee"]),
        ("L Tibia", points["Lknee"], points["Lankle"]),
        ("R Tibia", points["Rknee"], points["Rankle"]),
        ("L Foot", points["Lankle"], points["Lfoot"]),
        ("R Foot", points["Rankle"], points["Rfoot"])
    ]
    
    for name, p1, p2 in segments:
        metrics[name] = np.linalg.norm(p1 - p2) / 10.0 # to cm

    # Joint Angles
    angles = [
        ("L Elbow", points["Lshoulder"] - points["Lelbow"], points["Lwrist"] - points["Lelbow"]),
        ("R Elbow", points["Rshoulder"] - points["Relbow"], points["Rwrist"] - points["Relbow"]),
        ("L Knee", points["Lhip"] - points["Lknee"], points["Lankle"] - points["Lknee"]),
        ("R Knee", points["Rhip"] - points["Rknee"], points["Rankle"] - points["Rknee"]),
    ]
    for name, v1, v2 in angles:
        metrics[f"{name} Angle"] = get_angle(v1, v2)

    # Height Calculations
    # 1. Height Proxy (Nose to average heel)
    floor_y = (points["Lheel"][1] + points["Rheel"][1]) / 2.0
    metrics["Height (Proxy)"] = abs(points["nose"][1] - floor_y) / 10.0 + 15.0 # +15cm for head top
    
    # 2. Segmented Height (More robust to posture)
    avg_leg = (metrics["L Femur"] + metrics["L Tibia"] + metrics["R Femur"] + metrics["R Tibia"]) / 2.0
    torso = metrics["Torso Length"]
    neck_head = np.linalg.norm(points["nose"] - (points["Lshoulder"]+points["Rshoulder"])/2) / 10.0 + 15.0
    metrics["Height (Segments)"] = avg_leg + torso + neck_head

    # Print Results
    print("\n" + "="*60)
    print(f"3D BODY RESULTS (Units: cm)")
    print("="*60)
    for name, val in metrics.items():
        if "Angle" in name:
            print(f"{name:<20}: {val:>8.1f} deg")
        else:
            print(f"{name:<20}: {val:>8.2f} cm")
    print("="*60)

    # ---------------------------
    # Visualizations
    # ---------------------------
    # 1. 2D Landmarks on Left Image
    dispL = frameL.copy()
    mp_drawing.draw_landmarks(dispL, resL.pose_landmarks, BODY_CONNECTIONS)
    cv2.imshow("Left Pose (2D)", cv2.resize(dispL, (1280, 720)))

    # 2. 3D Skeleton Plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Height)")
    ax.set_zlabel("Z (Depth)")
    ax.set_title("Full Body 3D Skeleton")

    # Draw landmarks and connections
    ax.scatter(pts3d_world[:,0], pts3d_world[:,1], pts3d_world[:,2], c='r', s=20)
    for pair in BODY_CONNECTIONS:
        v1, v2 = pts3d_world[pair[0]], pts3d_world[pair[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], color='blue', alpha=0.6)

    # Set roughly equal scale
    max_range = np.ptp(pts3d_world, axis=0).max() / 2.0
    mids = (pts3d_world.max(axis=0) + pts3d_world.min(axis=0)) * 0.5
    ax.set_xlim(mids[0] - max_range, mids[0] + max_range)
    ax.set_ylim(mids[1] - max_range, mids[1] + max_range)
    ax.set_zlim(mids[2] - max_range, mids[2] + max_range)
    ax.set_box_aspect([1,1,1])

    # Save summary
    out_dir = "whole_body_measurement_sidebyside"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    out_path = os.path.join(out_dir, f"body_3d_report_{ts}.txt")
    with open(out_path, 'w') as f:
        for name, val in metrics.items():
            unit = "deg" if "Angle" in name else "cm"
            f.write(f"{name}: {val:.2f} {unit}\n")
    print(f"\nReport saved to: {out_path}")

    print("\n[INFO] Press any key on the image window to close.")
    plt.show(block=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
