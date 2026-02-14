# realtime_arm_3d_world_aligned_fixed.py
import time
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
USE_TEST_IMAGES = True   # Set to True to use test images instead of live camera
TEST_LEFT_IMAGE = "test_ss_left.png"          # Individual left image
TEST_RIGHT_IMAGE = "test_ss_right.png"        # Individual right image

LEFT_CAM_IDX = 1
RIGHT_CAM_IDX = 2
WIDTH = 1280
HEIGHT = 720
FPS = 30
PLOT_UPDATE_RATE = 0.06  # secs between plot updates (~16 Hz)
SMOOTH_ALPHA = 0.6       # smoothing for 3D points (0..1), larger -> smoother
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

# ---------------------------
# Helper: camera setup (fast start + lock)
# ---------------------------
def setup_cam(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # Try to force manual exposure/gain; webcams differ by driver
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # manual mode (Windows)
        if cam_index == 1:
            cap.set(cv2.CAP_PROP_EXPOSURE, -9)   # Try between -4 to -10 (log scale)
        else:
            cap.set(cv2.CAP_PROP_EXPOSURE, -8)   # Try between -4 to -10 (log scale) q
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
# MediaPipe Pose init (right-arm landmarks)
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# Indices (MediaPipe)
# Right shoulder = 12, right elbow = 14, right wrist = 16
# LANDMARKS = {"shoulder": 12, "elbow": 14, "wrist": 16}
LANDMARKS =  {"shoulder": 11, "elbow": 13, "wrist": 15}
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
# Visualization setup (Matplotlib interactive)
# ---------------------------
plt.ion()
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X (baseline)")
ax.set_ylabel("Y (up)")
ax.set_zlabel("Z (forward)")
line_plot, = ax.plot([0,0,0], [0,0,0], [0,0,0], marker="o", lw=2)
ax.set_title("Live 3D Right-Arm (shoulder->elbow->wrist) - world aligned")

# set fixed cube once (centered at origin_world -> but plot uses coordinates relative to origin_world)
# ax.set_xlim(-axis_half, axis_half)
# ax.set_ylim(-axis_half, axis_half)
# ax.set_zlim(-axis_half, axis_half)
ax.set_xlim(-1000, 0)
ax.set_ylim(0, 2600)
ax.set_zlim(1500, 2500)
ax.set_box_aspect([1,1,1])

# Keep smoothed 3D points to smooth jitter
smoothed = {"shoulder": None, "elbow": None, "wrist": None}
last_plot_time = 0.0

# Console output only - no file writing
print("\n" + "="*60)
print("ARM MEASUREMENT OUTPUT")
print("="*60)
# ---------------------------
# Load test images or setup cameras
# ---------------------------
def load_test_images():
    """Load test images from individual files."""
    if os.path.exists(TEST_LEFT_IMAGE) and os.path.exists(TEST_RIGHT_IMAGE):
        print(f"Loading test images: {TEST_LEFT_IMAGE} and {TEST_RIGHT_IMAGE}")
        frameL = cv2.imread(TEST_LEFT_IMAGE)
        frameR = cv2.imread(TEST_RIGHT_IMAGE)
        if frameL is None or frameR is None:
            raise RuntimeError(f"Failed to load test images")
        print(f"Loaded left ({frameL.shape}) and right ({frameR.shape}) images")
        return frameL, frameR
    else:
        raise RuntimeError(f"Test images not found. Expected both '{TEST_LEFT_IMAGE}' and '{TEST_RIGHT_IMAGE}'")

# ---------------------------
# Camera or image setup
# ---------------------------
if USE_TEST_IMAGES:
    # Load test images once
    test_frameL, test_frameR = load_test_images()
    capL = None
    capR = None
else:
    # Setup live cameras
    capL = setup_cam(LEFT_CAM_IDX)
    capR = setup_cam(RIGHT_CAM_IDX)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Unable to open one or both cameras. Check indices and connections.")

# ---------------------------
# Main loop
# ---------------------------
try:
    while True:
        t0 = time.time()
        
        if USE_TEST_IMAGES:
            # Use pre-loaded test images
            frameL = test_frameL.copy()
            frameR = test_frameR.copy()
            retL = True
            retR = True
        else:
            # Capture from live cameras
            retL, frameL = capL.read()
            frameL = cv2.flip(frameL, 1)   # mirror horizontally
            retR, frameR = capR.read()
            frameR = cv2.flip(frameR, 1)
            if not retL or not retR:
                print("Camera read error, exiting.")
                break

        # run pose detection on both frames
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

            # gather right-arm pixel coords (if visible)
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

        cv2.imshow("Left (L)", frameL)
        cv2.imshow("Right (R)", frameR)

        if valid:
            # undistort pixel coordinates first
            undL = {k: undistort_px(v, K1, dist1) for k,v in jointsL.items()}
            undR = {k: undistort_px(v, K2, dist2) for k,v in jointsR.items()}

            # triangulate each joint into 3D (in left-camera coords)
            pts3d_cam = {}
            for k in ("shoulder","elbow","wrist"):
                try:
                    X = triangulate_point(undL[k], undR[k])  # 3-vector in cam1 coords
                    if smoothed[k] is None:
                        smoothed[k] = X
                    else:
                        smoothed[k] = SMOOTH_ALPHA * smoothed[k] + (1 - SMOOTH_ALPHA) * X
                    pts3d_cam[k] = smoothed[k]
                except Exception:
                    pts3d_cam[k] = None

            # update 3D plot at lower rate to keep UI responsive
            now = time.time()
            if now - last_plot_time > PLOT_UPDATE_RATE and all(pts3d_cam[k] is not None for k in pts3d_cam):
                # transform to world frame: center at midpoint and rotate so X=baseline, Y=up, Z=forward
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

                # Calculate elbow internal angle between upper-arm and forearm.
                # Use vectors centered at the elbow: v1 = shoulder - elbow, v2 = wrist - elbow.
                # This yields 180 degrees when shoulder, elbow, wrist are colinear and the arm is straight.
                v1 = S - E  # elbow->shoulder (pointing from elbow to shoulder)
                v2 = W - E  # elbow->wrist (pointing from elbow to wrist)
                dot = float(np.dot(v1, v2))
                cross_norm = float(np.linalg.norm(np.cross(v1, v2)))
                # atan2(cross, dot) is numerically stable and returns angle in [0, pi]
                angle_rad = np.arctan2(cross_norm, dot)
                angle_deg = float(np.degrees(angle_rad))  # convert to degrees

                # Print measurements to console
                print(f"\nMeasurements:")
                print(f"  Shoulder -> Elbow: {distance1:.2f} mm")
                print(f"  Elbow -> Wrist:    {distance2:.2f} mm")
                print(f"  Elbow Angle:       {angle_deg:.2f} degrees")
                print(f"  3D Shoulder: ({S[0]:.1f}, {S[1]:.1f}, {S[2]:.1f})")
                print(f"  3D Elbow:    ({E[0]:.1f}, {E[1]:.1f}, {E[2]:.1f})")
                print(f"  3D Wrist:    ({W[0]:.1f}, {W[1]:.1f}, {W[2]:.1f})")
                print("-" * 60)

                # Prepare arrays for plotting
                Xs = [float(S[0]), float(E[0]), float(W[0])]
                Ys = [float(S[1]), float(E[1]), float(W[1])]
                Zs = [float(S[2]), float(E[2]), float(W[2])]

                # ---- Update line plot ----
                line_plot.set_data(Xs, Ys)
                line_plot.set_3d_properties(Zs)

                # ---- Add joint labels: 's' for shoulder, 'E' for elbow, 'W' for wrist ----
                for txt in ax.texts:
                    txt.remove()
                ax.text(Xs[0], Ys[0], Zs[0], 's', color='red', fontsize=12)
                ax.text(Xs[1], Ys[1], Zs[1], 'E', color='blue', fontsize=12)
                ax.text(Xs[2], Ys[2], Zs[2], 'W', color='green', fontsize=12)

                plt.draw()
                plt.pause(0.001)
                last_plot_time = now

        # quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # small sleep to avoid 100% CPU
        dt = time.time() - t0
        if dt < 1.0 / FPS:
            time.sleep(max(0.0, (1.0 / FPS) - dt))

finally:
    if not USE_TEST_IMAGES:
        if capL is not None:
            capL.release()
        if capR is not None:
            capR.release()
    cv2.destroyAllWindows()
    pose.close()
    plt.ioff()
    print("Exited cleanly.")
