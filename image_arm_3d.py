# image_arm_3d.py
# Process two static images instead of realtime camera feeds
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
MIN_AXIS_LIMIT = 600     # minimum half-size (units same as calibration units) for plot cube

# ---------------------------
# Load stereo params
# ---------------------------
data = np.load("stereo_params_sidebyside.npz")
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
print(f"K1:\n{K1}")
print(f"K2:\n{K2}")
print(f"P1:\n{P1}")
print(f"P2:\n{P2}")
print(f"T_stereo: {T_stereo}")
print(f"Baseline magnitude: {np.linalg.norm(T_stereo):.2f} mm")

# ---------------------------
# MediaPipe Pose init
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# Indices (MediaPipe)
# User is measuring RIGHT arm
LANDMARKS = {"shoulder": 12, "elbow": 14, "wrist": 16}


# ---------------------------
# Triangulation util
# ---------------------------
def undistort_px(pt, K, dist):
    pts = np.array(pt, dtype=np.float32).reshape(1,1,2)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    u, v = und[0,0]
    return float(u), float(v)

def triangulate_point(ptL, ptR):
    ptsL = np.array([[ptL[0]], [ptL[1]]], dtype=np.float64)
    ptsR = np.array([[ptR[0]], [ptR[1]]], dtype=np.float64)
    X_h = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
    X = (X_h[:3] / X_h[3]).reshape(3,)
    return X

# ---------------------------
# Build robust world-frame from R, T, and camera ups
# ---------------------------
def build_world_transform(R, T):
    """
    Build transform that maps points from left-camera coordinates to a human-friendly world frame.
    Returns (R_wc, origin) where:
      p_world = R_wc @ (p_cam1 - origin)
    """
    # camera centers in cam1 coords:
    C1 = np.zeros(3)         # left camera at origin in cam1 coords
    C2 = T.reshape(3,)       # right camera center in cam1 coords

    origin = (C1 + C2) / 2.0  # midpoint between cameras

    # compute 'up' directions in left-cam coords:
    left_up = np.array([0.0, -1.0, 0.0])      # camera Y points down -> -Y is up
    right_up = (R @ left_up)                  # right cam up expressed in left-cam frame

    # average up vector and normalize
    avg_up = left_up + right_up
    if np.linalg.norm(avg_up) < 1e-6:
        avg_up = left_up
    avg_up = avg_up / np.linalg.norm(avg_up)  # world Y (up)

    # baseline direction (from left to right) in cam1 coords
    baseline = (C2 - C1)
    baseline_len = np.linalg.norm(baseline)
    if baseline_len < 1e-6:
        raise RuntimeError("Baseline vector too small")
    baseline_dir = baseline / baseline_len

    # project baseline onto plane orthogonal to avg_up to get horizontal baseline direction
    baseline_proj = baseline_dir - np.dot(baseline_dir, avg_up) * avg_up
    if np.linalg.norm(baseline_proj) < 1e-6:
        x_axis = baseline_dir
    else:
        x_axis = baseline_proj / np.linalg.norm(baseline_proj)
    
    # Flip X axis to match visual left-to-right direction in images
    # (camera baseline points left-to-right, but we want right-to-left to match image view)
    x_axis = -x_axis

    # z = cross(x, y) to form right-handed frame (forward)
    z_axis = np.cross(x_axis, avg_up)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # re-orthogonalize y = cross(z, x)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Build rotation matrix R_wc that maps cam1 coords -> world coords:
    # p_world = R_wc @ (p_cam1 - origin)
    R_wc = np.vstack([x_axis, y_axis, z_axis])  # 3x3

    return R_wc, origin, baseline_len

# Use user's R_stereo, T_stereo to build world frame
R = R_stereo
T = T_stereo
R_worldcam, origin_world, baseline_len = build_world_transform(R, T)

# determine axis limits from baseline (units same as calibration units)
axis_half = int(max(MIN_AXIS_LIMIT, baseline_len * 1.2))

# ---------------------------
# Main: Process images from current directory
# ---------------------------
def main():
    # List available image files
    current_dir = '.'
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_files = []
    for file in os.listdir(current_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {current_dir}")
        return
    
    print("\nAvailable images in current directory:")
    for i, file in enumerate(image_files, 1):
        print(f"  {i}. {file}")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D arm pose estimation from stereo images')
    parser.add_argument('--num', '-n', type=str, default='', 
                        help='Image number suffix (e.g., 1 for test_ss_left1.png)')
    args = parser.parse_args()
    
    # Build image paths with optional number suffix
    suffix = args.num if args.num else ''
    img_left_path = f"test_ss_left{suffix}.png"
    img_right_path = f"test_ss_right{suffix}.png"
    
    # Check if files exist
    if not os.path.isfile(img_left_path):
        print(f"  Error: File '{img_left_path}' not found.")
        return
    if not os.path.isfile(img_right_path):
        print(f"  Error: File '{img_right_path}' not found.")
        return
    
    print(f"\nProcessing:")
    print(f"  Left image:  {img_left_path}")
    print(f"  Right image: {img_right_path}")
    
    # Load images at original resolution (use raw images, no undistortion)
    frameL = cv2.imread(img_left_path)
    frameR = cv2.imread(img_right_path)
    
    if frameL is None or frameR is None:
        print("Error: Could not read one or both images")
        return
    
    # Get original resolution
    hL, wL = frameL.shape[:2]
    hR, wR = frameR.shape[:2]
    print(f"Loaded original image sizes: Left {wL}x{hL}, Right {wR}x{hR}")
    
    # NO flip - keep images in original frame for P1/P2 calibration
    
    # Run pose detection
    print("Running pose detection...")
    resL = pose.process(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB))
    resR = pose.process(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB))
    
    jointsL = {}
    jointsR = {}
    valid = False
    
    if resL.pose_landmarks and resR.pose_landmarks:
        hL, wL = frameL.shape[:2]
        hR, wR = frameR.shape[:2]
        lmL = resL.pose_landmarks.landmark
        lmR = resR.pose_landmarks.landmark
        
        # Gather right-arm pixel coords (if visible)
        try:
            jL = {name: (int(lmL[idx].x * wL), int(lmL[idx].y * hL)) for name, idx in LANDMARKS.items()}
            jR = {name: (int(lmR[idx].x * wR), int(lmR[idx].y * hR)) for name, idx in LANDMARKS.items()}
            jointsL = jL
            jointsR = jR
            valid = True
        except Exception as e:
            print(f"Error extracting joints: {e}")
            valid = False
    else:
        print("Warning: Could not detect pose in one or both images")
        valid = False
    
    if not valid:
        print("Failed to extract arm landmarks from images")
        return
    
    # Draw 2D joints for debugging
    frameL_display = frameL.copy()
    frameR_display = frameR.copy()
    
    if jointsL:
        for name, p in jointsL.items():
            cv2.circle(frameL_display, p, 6, (0,255,0), -1)
            cv2.putText(frameL_display, name, (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    if jointsR:
        for name, p in jointsR.items():
            cv2.circle(frameR_display, p, 6, (0,255,0), -1)
            cv2.putText(frameR_display, name, (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.imshow("Left Image", frameL_display)
    cv2.imshow("Right Image", frameR_display)
    cv2.waitKey(0)
    
    # NOTE: P1 and P2 were calibrated for DISTORTED pixel coordinates
    # MediaPipe joints are already in distorted image space, so use them directly
    # DO NOT undistort before triangulation
    print(f"Left joints (distorted pixels): {jointsL}")
    print(f"Right joints (distorted pixels): {jointsR}")
    
    # Triangulate each joint into 3D (in left-camera coords)
    pts3d_cam = {}
    for k in ("shoulder","elbow","wrist"):
        try:
            X = triangulate_point(jointsL[k], jointsR[k])  # 3-vector in cam1 coords
            pts3d_cam[k] = X
        except Exception as e:
            print(f"Error triangulating {k}: {e}")
            pts3d_cam[k] = None
    
    if not all(pts3d_cam[k] is not None for k in pts3d_cam):
        print("Failed to triangulate all joints")
        return
    
    print("\n=== RAW TRIANGULATED (camera frame) ===")
    print(f"Shoulder (cam): {np.array(pts3d_cam['shoulder']).reshape(3,)}")
    print(f"Elbow (cam):    {np.array(pts3d_cam['elbow']).reshape(3,)}")
    print(f"Wrist (cam):    {np.array(pts3d_cam['wrist']).reshape(3,)}")
    
    # Transform to world frame
    pts3d_world = {}
    for k, p_cam in pts3d_cam.items():
        p_cam = np.array(p_cam).reshape(3,)
        p_rel = p_cam - origin_world            # translate to midpoint origin
        p_w = R_worldcam @ p_rel                # rotate into world axes
        pts3d_world[k] = p_w
    
    S = pts3d_world["shoulder"]
    E = pts3d_world["elbow"]
    W = pts3d_world["wrist"]
    
    # Calculate distances: shoulder-to-elbow and elbow-to-wrist
    distance1 = float(np.linalg.norm(E - S))  # shoulder to elbow
    distance2 = float(np.linalg.norm(W - E))  # elbow to wrist
    
    # Calculate elbow internal angle between upper-arm and forearm
    v1 = S - E  # elbow->shoulder
    v2 = W - E  # elbow->wrist
    dot = float(np.dot(v1, v2))
    cross_norm = float(np.linalg.norm(np.cross(v1, v2)))
    angle_rad = np.arctan2(cross_norm, dot)
    angle_deg = float(np.degrees(angle_rad))
    
    # Convert distances from mm to cm
    d1_cm = distance1 / 10.0
    d2_cm = distance2 / 10.0
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Shoulder position (world): {S}")
    print(f"Elbow position (world):    {E}")
    print(f"Wrist position (world):    {W}")
    print("-"*60)
    print(f"Shoulder-to-Elbow distance: {d1_cm:.2f} cm ({distance1:.2f} mm)")
    print(f"Elbow-to-Wrist distance:    {d2_cm:.2f} cm ({distance2:.2f} mm)")
    print(f"Total arm length:           {(d1_cm + d2_cm):.2f} cm ({distance1 + distance2:.2f} mm)")
    print(f"Elbow angle:                {angle_deg:.2f}°")
    print("="*60 + "\n")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X (baseline)")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z (forward)")
    ax.set_title("3D Right-Arm (shoulder->elbow->wrist) - world aligned")
    
    # Plot the arm
    Xs = [float(S[0]), float(E[0]), float(W[0])]
    Ys = [float(S[1]), float(E[1]), float(W[1])]
    Zs = [float(S[2]), float(E[2]), float(W[2])]
    
    ax.plot(Xs, Ys, Zs, marker="o", markersize=8, linewidth=2, color='blue')
    ax.text(Xs[0], Ys[0], Zs[0], 'Shoulder', color='red', fontsize=10)
    ax.text(Xs[1], Ys[1], Zs[1], 'Elbow', color='green', fontsize=10)
    ax.text(Xs[2], Ys[2], Zs[2], 'Wrist', color='blue', fontsize=10)
    
    # Auto-scale axes with EQUAL aspect ratio (same physical scale for all dimensions)
    padding = 0.2
    x_min, x_max = min(Xs), max(Xs)
    y_min, y_max = min(Ys), max(Ys)
    z_min, z_max = min(Zs), max(Zs)
    
    x_range = x_max - x_min if x_max > x_min else 100
    y_range = y_max - y_min if y_max > y_min else 100
    z_range = z_max - z_min if z_max > z_min else 100
    
    # Find maximum range to use for all axes (equal aspect ratio)
    max_range = max(x_range, y_range, z_range)
    padded_range = max_range * (1 + 2 * padding)
    
    # Center each axis around its midpoint
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    z_mid = (z_min + z_max) / 2.0
    half_range = padded_range / 2.0
    
    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)
    ax.set_box_aspect([1,1,1])
    
    print(f"3D plot axis ranges (equal aspect ratio):")
    print(f"  X: [{x_mid - half_range:.1f}, {x_mid + half_range:.1f}]")
    print(f"  Y: [{y_mid - half_range:.1f}, {y_mid + half_range:.1f}]")
    print(f"  Z: [{z_mid - half_range:.1f}, {z_mid + half_range:.1f}]")
    
    plt.show()
    
    # Save results to file
    output_dir = 'arm_length_measurement'
    os.makedirs(output_dir, exist_ok=True)
    ts_file = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')[:-3]
    output_file = os.path.join(output_dir, f'arm_lengths_image_{ts_file}.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("IMAGE ARM MEASUREMENT RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Left image:  {img_left_path}\n")
        f.write(f"Right image: {img_right_path}\n")
        f.write(f"Timestamp:   {ts_file}\n")
        f.write("-"*60 + "\n")
        f.write(f"Shoulder position (world): {S}\n")
        f.write(f"Elbow position (world):    {E}\n")
        f.write(f"Wrist position (world):    {W}\n")
        f.write("-"*60 + "\n")
        f.write(f"Shoulder-to-Elbow distance: {d1_cm:.2f} cm ({distance1:.2f} mm)\n")
        f.write(f"Elbow-to-Wrist distance:    {d2_cm:.2f} cm ({distance2:.2f} mm)\n")
        f.write(f"Total arm length:           {(d1_cm + d2_cm):.2f} cm ({distance1 + distance2:.2f} mm)\n")
        f.write(f"Elbow angle:                {angle_deg:.2f}°\n")
        f.write("="*60 + "\n")
    
    print(f"Results saved to: {output_file}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
