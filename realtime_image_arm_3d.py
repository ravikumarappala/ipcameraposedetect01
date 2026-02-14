# image_arm_3d.py
# Process two static images instead of realtime camera feeds
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
WIDTH = 1280
HEIGHT = 720
MIN_AXIS_LIMIT = 600     # minimum half-size (units same as calibration units) for plot cube

# ---------------------------
# Load stereo params
# ---------------------------
data = np.load("stereo_params_sidebyside1.npz")
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

# ---------------------------
# MediaPipe Pose init
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# Indices (MediaPipe)
# Left arm landmarks
LANDMARKS =  {"shoulder": 11, "elbow": 13, "wrist": 15}
# Right arm landmarks (alternative)
LANDMARKS_RIGHT = {"shoulder": 12, "elbow": 14, "wrist": 16}


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
    # Use specified test images by default
    img_left_path = "test_ss_left.png"
    img_right_path = "test_ss_right.png"
    
    if not os.path.exists(img_left_path) or not os.path.exists(img_right_path):
        print(f"Error: One or both default images not found: {img_left_path}, {img_right_path}")
        # List available image files
        current_dir = '.'
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(current_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if len(image_files) == 0:
            print("No image files found.")
            return
            
        print("\nAvailable images:")
        for i, file in enumerate(image_files, 1):
            print(f"  {i}. {file}")
            
        img_left_path = input("Enter LEFT image filename: ").strip()
        img_right_path = input("Enter RIGHT image filename: ").strip()

    print(f"\nProcessing:")
    print(f"  Left image:  {img_left_path}")
    print(f"  Right image: {img_right_path}")
    
    # Load and process images
    frameL = cv2.imread(img_left_path)
    frameR = cv2.imread(img_right_path)
    
    if frameL is None or frameR is None:
        print("Error: Could not read images")
        return
    
    frameL = cv2.resize(frameL, (WIDTH, HEIGHT))
    frameR = cv2.resize(frameR, (WIDTH, HEIGHT))
    frameL = cv2.flip(frameL, 1)
    frameR = cv2.flip(frameR, 1)
    
    print("Running pose detection...")
    resL = pose.process(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB))
    resR = pose.process(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB))
    
    valid = False
    if resL.pose_landmarks and resR.pose_landmarks:
        try:
            hL, wL = frameL.shape[:2]
            hR, wR = frameR.shape[:2]
            jointsL = {name: (int(resL.pose_landmarks.landmark[idx].x * wL), int(resL.pose_landmarks.landmark[idx].y * hL)) for name, idx in LANDMARKS.items()}
            jointsR = {name: (int(resR.pose_landmarks.landmark[idx].x * wR), int(resR.pose_landmarks.landmark[idx].y * hR)) for name, idx in LANDMARKS.items()}
            valid = True
        except Exception as e:
            print(f"Error extracting joints: {e}")
    
    if not valid:
        print("Failed to detect pose in one or both images.")
        return

    # Undistort pixel coordinates
    undL = {k: undistort_px(v, K1, dist1) for k,v in jointsL.items()}
    undR = {k: undistort_px(v, K2, dist2) for k,v in jointsR.items()}
    
    # Triangulate each joint into 3D (in left-camera coords)
    pts3d_cam = {}
    for k in ("shoulder","elbow","wrist"):
        try:
            X = triangulate_point(undL[k], undR[k])  # 3-vector in cam1 coords
            pts3d_cam[k] = X
        except Exception as e:
            print(f"Error triangulating {k}: {e}")
            pts3d_cam[k] = None
    
    if not all(pts3d_cam[k] is not None for k in pts3d_cam):
        print("Failed to triangulate all joints")
        return
    
    # Transform to world frame
    pts3d_world = {}
    for k, p_cam in pts3d_cam.items():
        p_cam = np.array(p_cam).reshape(3,)
        p_rel = p_cam - origin_world            # translate to midpoint origin
        p_w = R_worldcam @ p_rel                # rotate into world axes
        pts3d_world[k] = p_w
    
    S, E, W = pts3d_world["shoulder"], pts3d_world["elbow"], pts3d_world["wrist"]
    d1 = np.linalg.norm(E - S)
    d2 = np.linalg.norm(W - E)
    
    v1, v2 = S - E, W - E
    angle_rad = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    angle_deg = np.degrees(angle_rad)
    
    # Print RESULTS
    print("\n" + "="*60)
    print("ARM MEASUREMENT RESULTS")
    print("="*60)
    print(f"Shoulder position (world): {S}")
    print(f"Elbow position (world):    {E}")
    print(f"Wrist position (world):    {W}")
    print("-"*60)
    print(f"Shoulder-to-Elbow: {d1/10.0:.2f} cm ({d1:.2f} mm)")
    print(f"Elbow-to-Wrist:    {d2/10.0:.2f} cm ({d2:.2f} mm)")
    print(f"Total Arm Length:  {(d1+d2)/10.0:.2f} cm ({(d1+d2):.2f} mm)")
    print(f"Elbow Angle:       {angle_deg:.2f}°")
    print("="*60)

    # Save results to file
    output_dir = 'arm_length_measurement'
    os.makedirs(output_dir, exist_ok=True)
    ts_file = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')[:-3]
    output_file = os.path.join(output_dir, f'arm_lengths_image_{ts_file}.txt')
    with open(output_file, 'w') as f:
        f.write(f"Left image: {img_left_path}\nRight image: {img_right_path}\n\n")
        f.write(f"Shoulder-to-Elbow: {d1/10.0:.2f} cm\n")
        f.write(f"Elbow-to-Wrist:    {d2/10.0:.2f} cm\n")
        f.write(f"Total Arm Length:  {(d1+d2)/10.0:.2f} cm\n")
        f.write(f"Elbow Angle:       {angle_deg:.2f} deg\n")
    print(f"Results saved to: {output_file}")

    # Visualization
    frameL_disp = frameL.copy()
    frameR_disp = frameR.copy()
    for name, p in jointsL.items(): 
        cv2.circle(frameL_disp, p, 6, (0,255,0), -1)
        cv2.putText(frameL_disp, name, (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    for name, p in jointsR.items(): 
        cv2.circle(frameR_disp, p, 6, (0,255,0), -1)
        cv2.putText(frameR_disp, name, (p[0]+10, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    cv2.imshow("Left (L)", frameL_disp)
    cv2.imshow("Right (R)", frameR_disp)
    
    # 3D visualization
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    Xs, Ys, Zs = [S[0], E[0], W[0]], [S[1], E[1], W[1]], [S[2], E[2], W[2]]
    ax.plot(Xs, Ys, Zs, marker="o", markersize=8, linewidth=2, color='blue')
    ax.set_xlim(-1000, 0); ax.set_ylim(0, 2600); ax.set_zlim(1500, 2500)
    ax.set_box_aspect([1,1,1])
    plt.title("3D Arm Visualization")
    
    print("\n[INFO] Press any key on the image window to close.")
    cv2.waitKey(0)
    plt.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
