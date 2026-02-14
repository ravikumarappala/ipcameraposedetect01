# realtime_body_3d_sidebyside.py
# Real-time 3D body pose estimation for side-by-side camera setup
# Both cameras are on the same wall, ~4 feet apart, facing the same direction

import time
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import csv
import threading
import sys
import argparse

# ---------------------------
# Parse arguments
# ---------------------------
parser = argparse.ArgumentParser(description='3D Body Pose Estimation for Side-by-Side Cameras')
parser.add_argument('--test-images', action='store_true', 
                    help='Use static test images (test_ss_left.png, test_ss_right.png) instead of live camera')
parser.add_argument('--left-image', type=str, default='test_ss_left.png',
                    help='Path to left test image (default: test_ss_left.png)')
parser.add_argument('--right-image', type=str, default='test_ss_right.png',
                    help='Path to right test image (default: test_ss_right.png)')
args = parser.parse_args()

import matplotlib
if args.test_images:
    matplotlib.use('Agg') # Non-interactive backend for headless execution

import time
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
# Use RTSP cameras for side-by-side setup
USE_RTSP = True  # Set to False to use USB webcams
LEFT_RTSP = "rtsp://admin:Test12345@172.16.1.4:554/11"
RIGHT_RTSP = "rtsp://admin:Test12345@172.16.1.3:554/11"

# For USB cameras (if USE_RTSP = False)
LEFT_CAM_IDX = 1
RIGHT_CAM_IDX = 2

WIDTH = 1280
HEIGHT = 720
FPS = 30
PLOT_UPDATE_RATE = 0.06  # secs between plot updates (~16 Hz)
SMOOTH_ALPHA = 0.6       # smoothing for 3D points (0..1), larger -> smoother
MIN_AXIS_LIMIT = 600     # minimum half-size (units same as calibration units) for plot cube

# Flip settings (adjust as needed)
FLIP_LEFT_HORIZONTAL = False
FLIP_LEFT_VERTICAL = False
FLIP_RIGHT_HORIZONTAL = False
FLIP_RIGHT_VERTICAL = False

# ---------------------------
# Load stereo params
# ---------------------------
print("Loading stereo calibration parameters...")
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
    raise RuntimeError("stereo_params_sidebyside.npz must contain R and T for world alignment")
T_stereo = T_stereo.reshape(3,)

baseline_mm = data.get("baseline_mm", np.linalg.norm(T_stereo))
print(f"Baseline: {baseline_mm:.2f} mm ({baseline_mm/304.8:.2f} feet)")

# ---------------------------
# RTSP Camera class (for streaming cameras)
# ---------------------------
class RTSPCamera:
    def __init__(self, url, name="camera"):
        self.url = url
        self.name = name
        self.cap = None
        self.frame = None
        self.timestamp = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def open(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(30):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.frame = frame.copy()
                    self.timestamp = time.time()
                return True
            time.sleep(0.1)
        self.cap.release()
        self.cap = None
        return False
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._grab_frames, daemon=True)
        self.thread.start()
        
    def _grab_frames(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.time()
            else:
                time.sleep(0.01)
                
    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None
    
    def release(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

# ---------------------------
# Helper: camera setup
# ---------------------------
def setup_rtsp_camera(url, name):
    cam = RTSPCamera(url, name)
    if cam.open():
        cam.start()
        time.sleep(0.5)
        return cam
    return None

def setup_usb_camera(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    # Try to force manual exposure/gain
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        if cam_index == 1:
            cap.set(cv2.CAP_PROP_EXPOSURE, -9)
        else:
            cap.set(cv2.CAP_PROP_EXPOSURE, -8)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_GAIN, 0)
    except Exception:
        pass
    
    # Warm-up frames
    for _ in range(8):
        cap.read()
        time.sleep(0.02)
    return cap

# ---------------------------
# MediaPipe Pose init
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# Full body landmarks
LANDMARKS = {
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "nose": 0,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "left_ankle": 27,
    "left_heel": 29,
    "left_foot_index": 31,
    "right_knee": 26,
    "right_ankle": 28,
    "right_heel": 30,
    "right_foot_index": 32,
}

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
# Build world-frame transform
# ---------------------------
def build_world_transform(R, T):
    """
    Build transform that maps points from left-camera coordinates to a human-friendly world frame.
    For side-by-side setup, cameras should have minimal rotation, mostly horizontal translation.
    """
    # camera centers in cam1 coords:
    C1 = np.zeros(3)         # left camera at origin
    C2 = T.reshape(3,)       # right camera center

    origin = (C1 + C2) / 2.0  # midpoint between cameras

    # compute 'up' directions in left-cam coords:
    left_up = np.array([0.0, -1.0, 0.0])      # camera Y points down -> -Y is up
    right_up = (R @ left_up)                  # right cam up expressed in left-cam frame

    # average up vector and normalize
    avg_up = left_up + right_up
    if np.linalg.norm(avg_up) < 1e-6:
        avg_up = left_up
    avg_up = avg_up / np.linalg.norm(avg_up)  # world Y (up)

    # baseline direction (from left to right)
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

    # Build rotation matrix R_wc
    R_wc = np.vstack([x_axis, y_axis, z_axis])  # 3x3

    return R_wc, origin, baseline_len

# Build world frame
R = R_stereo
T = T_stereo
R_worldcam, origin_world, baseline_len = build_world_transform(R, T)

# determine axis limits from baseline
axis_half = int(max(MIN_AXIS_LIMIT, baseline_len * 1.2))

# ---------------------------
# Visualization setup (Matplotlib interactive)
# ---------------------------
if not args.test_images:
    plt.ion()
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X (baseline)")
ax.set_ylabel("Y (up)")
ax.set_zlabel("Z (forward)")
ax.set_title("Live 3D Body Pose - Side-by-Side Cameras")

# Autoscale will be used, but we set initial limits
ax.set_xlim(-axis_half, axis_half)
ax.set_ylim(-axis_half, axis_half)
ax.set_zlim(-axis_half, axis_half)
ax.set_box_aspect([2,2,2])

# Keep smoothed 3D points
smoothed = {name: None for name in LANDMARKS.keys()}
last_plot_time = 0.0

# Helper to label the middle of a line segment
def label_line(ax, p1, p2, label, color='k', fontsize=8):
    try:
        p1 = np.array(p1).reshape(3,)
        p2 = np.array(p2).reshape(3,)
        mid = (p1 + p2) / 2.0
        ax.text(float(mid[0]), float(mid[1]), float(mid[2]), label, color=color, fontsize=fontsize)
    except Exception:
        pass

# Helper: compute angle (degrees) between two vectors
def angle_deg(v1, v2):
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

from datetime import datetime

# Prepare CSV output file for per-frame whole-body metrics
output_dir = 'whole_body_measurement_sidebyside'
os.makedirs(output_dir, exist_ok=True)
ts_file = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')[:-3]
output_file = os.path.join(output_dir, f'whole_body_measurements_{ts_file}.tsv')
f_out = None
csv_writer = None
try:
    f_out = open(output_file, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f_out, delimiter='\t')
    header = [
        'timestamp',
        'shoulder_dist',
        'torso_length',
        'hip_dist',
        'left_shoulder_elbow',
        'left_elbow_wrist',
        'right_shoulder_elbow',
        'right_elbow_wrist',
        'left_hip_knee',
        'left_knee_ankle',
        'left_ankle_foot',
        'right_hip_knee',
        'right_knee_ankle',
        'right_ankle_foot',
        'left_elbow_angle_deg',
        'right_elbow_angle_deg',
        'left_knee_angle_deg',
        'right_knee_angle_deg',
        'left_shoulder_angle_deg',
        'right_shoulder_angle_deg',
        'spine_angle_deg',
        'head_sh_mid_dist',
        'total_height_proxy',
    ]
    csv_writer.writerow(header)
    f_out.flush()
except Exception as e:
    print(f"Warning: could not open {output_file} for writing: {e}")
    f_out = None
    csv_writer = None

# ---------------------------
# Camera start or load test images
# ---------------------------
capL = None
capR = None
test_frameL = None
test_frameR = None

if args.test_images:
    # Load static test images
    print(f"Loading test images...")
    print(f"  Left: {args.left_image}")
    print(f"  Right: {args.right_image}")
    
    test_frameL = cv2.imread(args.left_image)
    test_frameR = cv2.imread(args.right_image)
    
    if test_frameL is None:
        raise RuntimeError(f"Could not load left test image: {args.left_image}")
    if test_frameR is None:
        raise RuntimeError(f"Could not load right test image: {args.right_image}")
    
    print(f"Test images loaded successfully!")
    print(f"  Left image size: {test_frameL.shape[1]}x{test_frameL.shape[0]}")
    print(f"  Right image size: {test_frameR.shape[1]}x{test_frameR.shape[0]}")
    print("Processing static images. Press 'q' to quit.")
else:
    # Use live cameras
    print("Opening cameras...")
    if USE_RTSP:
        capL = setup_rtsp_camera(LEFT_RTSP, "Left")
        capR = setup_rtsp_camera(RIGHT_RTSP, "Right")
    else:
        capL = setup_usb_camera(LEFT_CAM_IDX)
        capR = setup_usb_camera(RIGHT_CAM_IDX)

    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Unable to open one or both cameras. Check settings and connections.")

    print("Cameras ready. Starting real-time 3D pose estimation...")
    print("Press 'q' in any camera window to quit.")

# ---------------------------
# Main loop
# ---------------------------
try:
    print("Processing static images. Press 'q' to quit.")
    frame_count = 0
    max_test_frames = 5  # Exit test mode after this many frames if no valid pose
    while True:
        frame_count += 1
        if args.test_images and frame_count > max_test_frames:
            print(f"No valid pose detected after {max_test_frames} attempts. Exiting...")
            break
        t0 = time.time()
        
        # Get frames from either test images or live cameras
        if args.test_images:
            # Use static test images
            frameL = test_frameL.copy()
            frameR = test_frameR.copy()
            retL, retR = True, True
        else:
            # Read from live cameras
            retL, frameL = capL.read()
            retR, frameR = capR.read()
        
        if not retL or not retR:
            print("Camera read error, exiting.")
            break

        # Apply flips if needed
        if FLIP_LEFT_HORIZONTAL:
            frameL = cv2.flip(frameL, 1)
        if FLIP_LEFT_VERTICAL:
            frameL = cv2.flip(frameL, 0)
        if FLIP_RIGHT_HORIZONTAL:
            frameR = cv2.flip(frameR, 1)
        if FLIP_RIGHT_VERTICAL:
            frameR = cv2.flip(frameR, 0)

        # run pose detection on both frames
        resL = pose.process(cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB))
        resR = pose.process(cv2.cvtColor(frameR, cv2.COLOR_BGR2RGB))

        jointsL = {}
        jointsR = {}
        valid = False

        # Debug output for pose detection
        print(f"Frame {frame_count}: Left pose detected: {resL.pose_landmarks is not None}, Right pose detected: {resR.pose_landmarks is not None}")

        if resL.pose_landmarks and resR.pose_landmarks:
            hL, wL = frameL.shape[:2]
            hR, wR = frameR.shape[:2]
            lmL = resL.pose_landmarks.landmark
            lmR = resR.pose_landmarks.landmark

            # gather pixel coords
            print("gather pixel coords.")
            try:
                jL = {name: (int(lmL[idx].x * wL), int(lmL[idx].y * hL)) for name, idx in LANDMARKS.items()}
                jR = {name: (int(lmR[idx].x * wR), int(lmR[idx].y * hR)) for name, idx in LANDMARKS.items()}
                jointsL = jL
                jointsR = jR
                valid = True
            except Exception:
                valid = False

        # draw 2D joints for debugging
        if jointsL:
            for p in jointsL.values():
                cv2.circle(frameL, p, 6, (0,255,0), -1)
        if jointsR:
            for p in jointsR.values():
                cv2.circle(frameR, p, 6, (0,255,0), -1)

        if not args.test_images:
            cv2.imshow("Left (L)", frameL)
            cv2.imshow("Right (R)", frameR)

        if valid:
            # undistort pixel coordinates
            undL = {k: undistort_px(v, K1, dist1) for k,v in jointsL.items()}
            undR = {k: undistort_px(v, K2, dist2) for k,v in jointsR.items()}

            # triangulate each landmark into 3D (in left-camera coords)
            pts3d_cam = {}
            for k in LANDMARKS.keys():
                try:
                    X = triangulate_point(undL[k], undR[k])
                    if smoothed[k] is None:
                        smoothed[k] = X
                    else:
                        smoothed[k] = SMOOTH_ALPHA * smoothed[k] + (1 - SMOOTH_ALPHA) * X
                    pts3d_cam[k] = smoothed[k]
                except Exception:
                    pts3d_cam[k] = None

            # update 3D plot at lower rate
            now = time.time()
            valid_count = sum(1 for k in pts3d_cam if pts3d_cam[k] is not None)
            print(f"  Valid 3D points: {valid_count}/{len(pts3d_cam)}")
            if now - last_plot_time > PLOT_UPDATE_RATE and valid_count >= 10:  # Require at least 10 landmarks
                # transform to world frame
                pts3d_world = {}
                for k, p_cam in pts3d_cam.items():
                    p_cam = np.array(p_cam).reshape(3,)
                    p_rel = p_cam - origin_world
                    p_w = R_worldcam @ p_rel
                    pts3d_world[k] = p_w

                # Get required points
                Ls = pts3d_world.get("left_shoulder")
                Rs = pts3d_world.get("right_shoulder")
                Le = pts3d_world.get("left_elbow")
                Re = pts3d_world.get("right_elbow")
                Lw = pts3d_world.get("left_wrist")
                Rw = pts3d_world.get("right_wrist")
                nose = pts3d_world.get("nose")
                Lhip = pts3d_world.get("left_hip")
                Rhip = pts3d_world.get("right_hip")
                Lk = pts3d_world.get("left_knee")
                La = pts3d_world.get("left_ankle")
                Lf = pts3d_world.get("left_foot_index")
                Rk = pts3d_world.get("right_knee")
                Ra = pts3d_world.get("right_ankle")
                Rf = pts3d_world.get("right_foot_index")

                # compute shoulder midpoint
                shoulder_mid = None
                if Ls is not None and Rs is not None:
                    shoulder_mid = (Ls + Rs) / 2.0

                # Compute metrics if all required points exist
                required = [Ls, Rs, Le, Re, Lw, Rw, nose, Lhip, Rhip, Lk, La, Lf, Rk, Ra, Rf]
                if all(p is not None for p in required):
                    shoulder_dist = float(np.linalg.norm(Rs - Ls))
                    left_sh_el = float(np.linalg.norm(Le - Ls))
                    left_el_wr = float(np.linalg.norm(Lw - Le))
                    right_sh_el = float(np.linalg.norm(Re - Rs))
                    right_el_wr = float(np.linalg.norm(Rw - Re))
                    
                    shoulder_mid = (Ls + Rs) / 2.0
                    head_sh_mid = float(np.linalg.norm(nose - shoulder_mid))
                    
                    left_hip_knee = float(np.linalg.norm(Lk - Lhip))
                    left_knee_ankle = float(np.linalg.norm(La - Lk))
                    left_ankle_foot = float(np.linalg.norm(Lf - La))
                    
                    right_hip_knee = float(np.linalg.norm(Rk - Rhip))
                    right_knee_ankle = float(np.linalg.norm(Ra - Rk))
                    right_ankle_foot = float(np.linalg.norm(Rf - Ra))
                    
                    hip_mid = (Lhip + Rhip) / 2.0
                    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
                    
                    left_elbow_angle = angle_deg(Ls - Le, Lw - Le)
                    right_elbow_angle = angle_deg(Rs - Re, Rw - Re)
                    left_knee_angle = angle_deg(Lhip - Lk, La - Lk)
                    right_knee_angle = angle_deg(Rhip - Rk, Ra - Rk)
                    left_shoulder_angle = angle_deg(Le - Ls, Rs - Ls)
                    right_shoulder_angle = angle_deg(Re - Rs, Ls - Rs)
                    
                    torso_vec = shoulder_mid - hip_mid
                    up_axis = np.array([0.0, 1.0, 0.0])
                    spine_angle = angle_deg(torso_vec, up_axis)
                    
                    foot_min_y = min(float(Lf[1]), float(Rf[1]))
                    total_height_proxy = abs(float(nose[1]) - foot_min_y)

                    # write to CSV
                    if csv_writer is not None and f_out is not None:
                        try:
                            vals = [
                                shoulder_dist, torso_length, float(np.linalg.norm(Rhip - Lhip)),
                                left_sh_el, left_el_wr, right_sh_el, right_el_wr,
                                left_hip_knee, left_knee_ankle, left_ankle_foot,
                                right_hip_knee, right_knee_ankle, right_ankle_foot,
                                left_elbow_angle, right_elbow_angle,
                                left_knee_angle, right_knee_angle,
                                left_shoulder_angle, right_shoulder_angle,
                                spine_angle, head_sh_mid, total_height_proxy,
                            ]
                            vals_int = [int(round(v)) for v in vals]
                            ts = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
                            row = [ts] + vals_int
                            csv_writer.writerow(row)
                            f_out.flush()
                        except Exception:
                            pass

                # Prepare arrays for plotting
                plot_joints = [
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
                    "left_foot_index", "right_foot_index", "nose"
                ]
                Xs = [float(pts3d_world[j][0]) for j in plot_joints if j in pts3d_world]
                Ys = [float(pts3d_world[j][1]) for j in plot_joints if j in pts3d_world]
                Zs = [float(pts3d_world[j][2]) for j in plot_joints if j in pts3d_world]

                # Clear previous lines and text
                for ln in ax.lines[:]:
                    try:
                        ln.remove()
                    except Exception:
                        pass
                for txt in ax.texts[:]:
                    try:
                        txt.remove()
                    except Exception:
                        pass

                # Draw skeleton lines
                if Ls is not None and Rs is not None:
                    ax.plot([float(Ls[0]), float(Rs[0])], [float(Ls[1]), float(Rs[1])], [float(Ls[2]), float(Rs[2])], color='k', lw=2)
                    label_line(ax, Ls, Rs, 'shoulder', color='k')

                # Left arm
                if Ls is not None and Le is not None:
                    ax.plot([float(Ls[0]), float(Le[0])], [float(Ls[1]), float(Le[1])], [float(Ls[2]), float(Le[2])], color='b', lw=2)
                if Le is not None and Lw is not None:
                    ax.plot([float(Le[0]), float(Lw[0])], [float(Le[1]), float(Lw[1])], [float(Le[2]), float(Lw[2])], color='b', lw=2)

                # Left leg
                if Lhip is not None and Lk is not None:
                    ax.plot([float(Lhip[0]), float(Lk[0])], [float(Lhip[1]), float(Lk[1])], [float(Lhip[2]), float(Lk[2])], color='b', lw=2)
                if Lk is not None and La is not None:
                    ax.plot([float(Lk[0]), float(La[0])], [float(Lk[1]), float(La[1])], [float(Lk[2]), float(La[2])], color='b', lw=2)
                if La is not None and Lf is not None:
                    ax.plot([float(La[0]), float(Lf[0])], [float(La[1]), float(Lf[1])], [float(La[2]), float(Lf[2])], color='b', lw=2)

                # Right arm
                if Rs is not None and Re is not None:
                    ax.plot([float(Rs[0]), float(Re[0])], [float(Rs[1]), float(Re[1])], [float(Rs[2]), float(Re[2])], color='r', lw=2)
                if Re is not None and Rw is not None:
                    ax.plot([float(Re[0]), float(Rw[0])], [float(Re[1]), float(Rw[1])], [float(Re[2]), float(Rw[2])], color='r', lw=2)

                # Right leg
                if Rhip is not None and Rk is not None:
                    ax.plot([float(Rhip[0]), float(Rk[0])], [float(Rhip[1]), float(Rk[1])], [float(Rhip[2]), float(Rk[2])], color='r', lw=2)
                if Rk is not None and Ra is not None:
                    ax.plot([float(Rk[0]), float(Ra[0])], [float(Rk[1]), float(Ra[1])], [float(Rk[2]), float(Ra[2])], color='r', lw=2)
                if Ra is not None and Rf is not None:
                    ax.plot([float(Ra[0]), float(Rf[0])], [float(Ra[1]), float(Rf[1])], [float(Ra[2]), float(Rf[2])], color='r', lw=2)

                # Torso and head
                if nose is not None and shoulder_mid is not None:
                    ax.plot([float(nose[0]), float(shoulder_mid[0])], [float(nose[1]), float(shoulder_mid[1])], [float(nose[2]), float(shoulder_mid[2])], color='m', lw=1)
                if Lhip is not None and Rhip is not None:
                    hip_mid = (Lhip + Rhip) / 2.0
                    ax.plot([float(Lhip[0]), float(Rhip[0])], [float(Lhip[1]), float(Rhip[1])], [float(Lhip[2]), float(Rhip[2])], color='k', lw=1)
                    if shoulder_mid is not None:
                        ax.plot([float(shoulder_mid[0]), float(hip_mid[0])], [float(shoulder_mid[1]), float(hip_mid[1])], [float(shoulder_mid[2]), float(hip_mid[2])], color='k', lw=1)
                        label_line(ax, shoulder_mid, hip_mid, 'torso', color='k')

                # Add joint labels
                for j in plot_joints:
                    if j in pts3d_world:
                        p = pts3d_world[j]
                        ax.text(float(p[0]), float(p[1]), float(p[2]), j[:3], color='green', fontsize=6)

                # Autoscale axes
                try:
                    all_x = np.array(Xs)
                    all_y = np.array(Ys)
                    all_z = np.array(Zs)
                    pad = 200
                    if all_x.size > 0 and all_y.size > 0 and all_z.size > 0:
                        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
                        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
                        ax.set_zlim(all_z.min() - pad, all_z.max() + pad)
                except Exception:
                    pass

                last_plot_time = now
                
                # If in test mode, save results and exit
                if args.test_images:
                    print("Saving visualization results...")
                    cv2.imwrite("result_ss_left_skeleton.png", frameL)
                    cv2.imwrite("result_ss_right_skeleton.png", frameR)
                    print("Saved: result_ss_left_skeleton.png, result_ss_right_skeleton.png")
                    
                    try:
                        plt.savefig("result_ss_3d_plot.png")
                        print("Saved: result_ss_3d_plot.png")
                    except Exception as e:
                        print(f"Could not save 3D plot: {e}")
                        
                    print("Test run complete. Exiting...")
                    break
                else:
                    plt.draw()
                    plt.pause(0.001)

        # frame rate control
        dt = time.time() - t0
        if dt < 1.0 / FPS:
            time.sleep(max(0.0, (1.0 / FPS) - dt))

        # quit key
        if not args.test_images:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

finally:
    if capL is not None:
        capL.release()
    if capR is not None:
        capR.release()
    cv2.destroyAllWindows()
    pose.close()
    plt.ioff()
    if f_out is not None:
        f_out.close()
    print("Exited cleanly.")
