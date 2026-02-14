# image_body_3d.py
# 3D full body pose estimation from a stereo pair of static images
# Derived strictly from image_arm_3d.py patterns
import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import sys
import csv
import argparse
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
MIN_AXIS_LIMIT = 600     # units: mm

# ---------------------------
# Load stereo params (matching image_arm_3d.py)
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
# MediaPipe Pose init
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
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
# Triangulation util (matching image_arm_3d.py)
# ---------------------------
def triangulate_point(ptL, ptR):
    ptsL = np.array([[ptL[0]], [ptL[1]]], dtype=np.float64)
    ptsR = np.array([[ptR[0]], [ptR[1]]], dtype=np.float64)
    X_h = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
    X = (X_h[:3] / X_h[3]).reshape(3,)
    return X

# ---------------------------
# World Transform (matching image_arm_3d.py)
# ---------------------------
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
    parser = argparse.ArgumentParser(description='3D full body pose estimation from stereo images')
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

    # MediaPipe
    print("Running pose detection...")
    resL = pose.process(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB))
    resR = pose.process(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB))

    if not resL.pose_landmarks or not resR.pose_landmarks:
        print("Error: Could not detect pose in one or both images")
        return

    lmL = resL.pose_landmarks.landmark
    lmR = resR.pose_landmarks.landmark

    # Triangulate all 33 landmarks (using raw pixels like image_arm_3d.py)
    pts3d_cam = []
    jointsL_px = {}
    for i in range(33):
        # Use raw MediaPipe pixel coordinates (NO undistort/rectify)
        pL = (int(lmL[i].x * wL), int(lmL[i].y * hL))
        pR = (int(lmR[i].x * wR), int(lmR[i].y * hR))
        X = triangulate_point(pL, pR)  # Direct triangulation with P1, P2
        pts3d_cam.append(X)
        # Store pixel coords for annotation
        for name, idx in LM.items():
            if idx == i:
                jointsL_px[name] = pL
    
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
    m["L Ankle Angle"] = get_angle_deg(points["Lknee"]-points["Lankle"], points["Lfoot"]-points["Lankle"])
    m["R Ankle Angle"] = get_angle_deg(points["Rknee"]-points["Rankle"], points["Rfoot"]-points["Rankle"])

    # Height
    floor_y = (points["Lheel"][1] + points["Rheel"][1]) / 2.0
    m["Height (Proxy)"] = abs(points["nose"][1] - floor_y) / 10.0 + 15.0
    avg_leg = (m["L Femur"] + m["L Tibia"] + m["R Femur"] + m["R Tibia"]) / 2.0
    neck_head = dist(points["nose"], (points["Lshoulder"]+points["Rshoulder"])/2) + 15.0
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
        points = {name: pts3d_world_scaled[idx] for name, idx in LM.items()}
        
        # Recalculate ALL measurements with scaled points
        m["Shoulder Width"] = dist(points["Lshoulder"], points["Rshoulder"])
        m["Hip Width"] = dist(points["Lhip"], points["Rhip"])
        m["Torso Length"] = dist((points["Lshoulder"]+points["Rshoulder"])/2, (points["Lhip"]+points["Rhip"])/2)
        
        m["L Humerus"] = dist(points["Lshoulder"], points["Lelbow"])
        m["R Humerus"] = dist(points["Rshoulder"], points["Relbow"])
        m["L Radius"] = dist(points["Lelbow"], points["Lwrist"])
        m["R Radius"] = dist(points["Relbow"], points["Rwrist"])
        m["L Arm Total"] = m["L Humerus"] + m["L Radius"]
        m["R Arm Total"] = m["R Humerus"] + m["R Radius"]
        
        m["L Femur"] = dist(points["Lhip"], points["Lknee"])
        m["R Femur"] = dist(points["Rhip"], points["Rknee"])
        m["L Tibia"] = dist(points["Lknee"], points["Lankle"])
        m["R Tibia"] = dist(points["Rknee"], points["Rankle"])
        m["L Leg Total"] = m["L Femur"] + m["L Tibia"]
        m["R Leg Total"] = m["R Femur"] + m["R Tibia"]
        
        # Angles don't change with scaling
        m["Height (Segments)"] = args.height  # Now exact
        m["Height (Proxy)"] = abs(points["nose"][1] - floor_y) / 10.0 + 15.0
        
        # Use scaled points for 3D plot
        pts3d_world = pts3d_world_scaled

    # Print
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    for name, val in m.items():
        unit = "deg" if "Angle" in name else "cm"
        print(f"{name:<22}: {val:>8.2f} {unit}")
    print("="*40)

    # Save outputs
    out_dir = "whole_body_measurement_sidebyside"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    
    # Annotated 2D image with ALL measurements
    dispL = frameL.copy()
    mp_drawing.draw_landmarks(dispL, resL.pose_landmarks, BODY_CONNECTIONS)
    
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
        ("L Ank: {:.0f}°".format(m["L Ankle Angle"]), (jointsL_px["Lankle"][0]-60, jointsL_px["Lankle"][1]), (255, 200, 0)),
        ("R Ank: {:.0f}°".format(m["R Ankle Angle"]), (jointsL_px["Rankle"][0]+20, jointsL_px["Rankle"][1]), (255, 200, 0)),
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
    ax.set_title("Full Body 3D Skeleton (Standing)")
    
    # SWAP AXES: Plot Z (depth) as X, X (width) as Y, Y (height) as Z to show standing
    # This creates: X=depth, Y=width, Z=height (standard standing orientation)
    ax.set_xlabel("X (Depth)")
    ax.set_ylabel("Y (Width)")
    ax.set_zlabel("Z (Height)")

    # Plot skeleton with SWAPPED axes: (Z, X, Y) -> (depth, width, height)
    ax.scatter(pts3d_world[:,2], pts3d_world[:,0], pts3d_world[:,1], c='r', s=20)
    for pair in BODY_CONNECTIONS:
        v1, v2 = pts3d_world[pair[0]], pts3d_world[pair[1]]
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
    
    # Set view angle to show standing from front-side
    ax.view_init(elev=15, azim=-75)
    
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
        writer.writerow(['Angle', 'Ankle', f"{m['L Ankle Angle']:.1f}", f"{m['R Ankle Angle']:.1f}", f"{abs(m['L Ankle Angle']-m['R Ankle Angle']):.1f}", 'deg'])
        
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
        f.write(f"Timestamp: {ts}\n" + "="*40 + "\n")
        for name, val in m.items():
            unit = "deg" if "Angle" in name else "cm"
            f.write(f"{name}: {val:.2f} {unit}\n")
    print(f"Text report saved: {out_path}")
    print(f"\n✓ All outputs saved to: {out_dir}/")

if __name__ == "__main__":
    main()
