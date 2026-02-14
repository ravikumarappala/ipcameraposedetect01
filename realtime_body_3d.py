# realtime_arm_3d_world_aligned_fixed.py
import time
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import csv

# ---------------------------
# Config
# ---------------------------
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
data = np.load("stereo_params.npz")
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
    # legs / feet
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
ax.set_box_aspect([2,2,2])

# Keep smoothed 3D points to smooth jitter
smoothed = {name: None for name in LANDMARKS.keys()}
last_plot_time = 0.0

# Small helper to label the middle of a line segment
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

# Prepare CSV (TSV) output file for per-frame whole-body metrics
output_dir = 'whole_body_measurement'
os.makedirs(output_dir, exist_ok=True)
# make a timestamped filename per run to avoid overwriting previous runs
ts_file = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')[:-3]  # UTC, milliseconds
output_file = os.path.join(output_dir, f'whole_body_measurements_{ts_file}.tsv')
f_out = None
csv_writer = None
try:
    # open in write mode so each run gets a fresh file; newline='' for correct CSV on Windows
    f_out = open(output_file, 'w', newline='', encoding='utf-8')
    # use tab delimiter explicitly
    csv_writer = csv.writer(f_out, delimiter='\t')
    # write header
    try:
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
        try:
            os.fsync(f_out.fileno())
        except Exception:
            pass
    except OSError:
        pass
except Exception:
    print(f"Warning: could not open {output_file} for writing; per-frame metrics will not be saved.")
    f_out = None
    csv_writer = None

# ---------------------------
# Camera start
# ---------------------------
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

            # triangulate each tracked landmark into 3D (in left-camera coords)
            pts3d_cam = {}
            for k in LANDMARKS.keys():
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


                # Get required points (including legs/feet)
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

                # compute shoulder midpoint if available (used for plotting even if other points are missing)
                shoulder_mid = None
                if Ls is not None and Rs is not None:
                    shoulder_mid = (Ls + Rs) / 2.0

                # Only compute/write if the necessary points exist (for whole-body metrics)
                required = [Ls, Rs, Le, Re, Lw, Rw, nose, Lhip, Rhip, Lk, La, Lf, Rk, Ra, Rf]
                if all(p is not None for p in required):
                    # shoulder distance
                    shoulder_dist = float(np.linalg.norm(Rs - Ls))

                    # left arm distances
                    left_sh_el = float(np.linalg.norm(Le - Ls))
                    left_el_wr = float(np.linalg.norm(Lw - Le))

                    # right arm distances
                    right_sh_el = float(np.linalg.norm(Re - Rs))
                    right_el_wr = float(np.linalg.norm(Rw - Re))

                    # shoulder midpoint
                    shoulder_mid = (Ls + Rs) / 2.0

                    # head to shoulder midpoint
                    head_sh_mid = float(np.linalg.norm(nose - shoulder_mid))

                    # hips to shoulder midpoint
                    left_hip_sh_mid = float(np.linalg.norm(Lhip - shoulder_mid))
                    right_hip_sh_mid = float(np.linalg.norm(Rhip - shoulder_mid))

                    # leg distances
                    left_hip_knee = float(np.linalg.norm(Lk - Lhip))
                    left_knee_ankle = float(np.linalg.norm(La - Lk))
                    left_ankle_foot = float(np.linalg.norm(Lf - La))

                    right_hip_knee = float(np.linalg.norm(Rk - Rhip))
                    right_knee_ankle = float(np.linalg.norm(Ra - Rk))
                    right_ankle_foot = float(np.linalg.norm(Rf - Ra))

                    # torso length (shoulder midpoint to hip midpoint)
                    hip_mid = (Lhip + Rhip) / 2.0
                    torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))

                    # elbow angles (degrees)
                    left_elbow_angle = angle_deg(Ls - Le, Lw - Le)
                    right_elbow_angle = angle_deg(Rs - Re, Rw - Re)

                    # knee angles (degrees)
                    left_knee_angle = angle_deg(Lhip - Lk, La - Lk)
                    right_knee_angle = angle_deg(Rhip - Rk, Ra - Rk)

                    # shoulder angles: angle between upper-arm and baseline (shoulder->other_shoulder)
                    left_shoulder_angle = angle_deg(Le - Ls, Rs - Ls)
                    right_shoulder_angle = angle_deg(Re - Rs, Ls - Rs)

                    # spine angle: angle between torso (shoulder_mid->hip_mid) and world up (0,1,0)
                    hip_mid = (Lhip + Rhip) / 2.0
                    torso_vec = shoulder_mid - hip_mid
                    up_axis = np.array([0.0, 1.0, 0.0])
                    spine_angle = angle_deg(torso_vec, up_axis)

                    # total height proxy: nose Y - min(foot Ys)
                    foot_min_y = min(float(Lf[1]), float(Rf[1]))
                    total_height_proxy = abs(float(nose[1]) - foot_min_y)

                    # write values as integers (no decimals) to CSV with timestamp
                    if csv_writer is not None and f_out is not None:
                        try:
                            vals = [
                                shoulder_dist,
                                torso_length,
                                float(np.linalg.norm(Rhip - Lhip)),
                                left_sh_el,
                                left_el_wr,
                                right_sh_el,
                                right_el_wr,
                                left_hip_knee,
                                left_knee_ankle,
                                left_ankle_foot,
                                right_hip_knee,
                                right_knee_ankle,
                                right_ankle_foot,
                                left_elbow_angle,
                                right_elbow_angle,
                                left_knee_angle,
                                right_knee_angle,
                                left_shoulder_angle,
                                right_shoulder_angle,
                                spine_angle,
                                head_sh_mid,
                                total_height_proxy,
                            ]
                            vals_int = [int(round(v)) for v in vals]
                            # ISO timestamp with milliseconds
                            ts = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
                            row = [ts] + vals_int
                            csv_writer.writerow(row)
                            f_out.flush()
                            try:
                                os.fsync(f_out.fileno())
                            except Exception:
                                pass
                        except Exception:
                            pass

                # Prepare arrays for plotting (plot full-body joints)
                plot_joints = [
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
                    "left_foot_index", "right_foot_index", "nose"
                ]
                Xs = [float(pts3d_world[j][0]) for j in plot_joints if j in pts3d_world]
                Ys = [float(pts3d_world[j][1]) for j in plot_joints if j in pts3d_world]
                Zs = [float(pts3d_world[j][2]) for j in plot_joints if j in pts3d_world]

                # ---- Remove previous plotted lines and redraw segments ----
                for ln in ax.lines[:]:
                    try:
                        ln.remove()
                    except Exception:
                        pass

                # draw shoulder baseline
                if Ls is not None and Rs is not None:
                    ax.plot([float(Ls[0]), float(Rs[0])], [float(Ls[1]), float(Rs[1])], [float(Ls[2]), float(Rs[2])], color='k', lw=2)
                    label_line(ax, Ls, Rs, 'shoulder_dist', color='k')

                # draw left arm
                if Ls is not None and Le is not None:
                    ax.plot([float(Ls[0]), float(Le[0])], [float(Ls[1]), float(Le[1])], [float(Ls[2]), float(Le[2])], color='b', lw=2)
                    label_line(ax, Ls, Le, 'left_shoulder_elbow', color='b')
                if Le is not None and Lw is not None:
                    ax.plot([float(Le[0]), float(Lw[0])], [float(Le[1]), float(Lw[1])], [float(Le[2]), float(Lw[2])], color='b', lw=2)
                    label_line(ax, Le, Lw, 'left_elbow_wrist', color='b')

                # draw left leg (hip -> knee -> ankle -> foot)
                if Lhip is not None and Lk is not None:
                    ax.plot([float(Lhip[0]), float(Lk[0])], [float(Lhip[1]), float(Lk[1])], [float(Lhip[2]), float(Lk[2])], color='b', lw=2)
                    label_line(ax, Lhip, Lk, 'left_hip_knee', color='b')
                if Lk is not None and La is not None:
                    ax.plot([float(Lk[0]), float(La[0])], [float(Lk[1]), float(La[1])], [float(Lk[2]), float(La[2])], color='b', lw=2)
                    label_line(ax, Lk, La, 'left_knee_ankle', color='b')
                if La is not None and Lf is not None:
                    ax.plot([float(La[0]), float(Lf[0])], [float(La[1]), float(Lf[1])], [float(La[2]), float(Lf[2])], color='b', lw=2)
                    label_line(ax, La, Lf, 'left_ankle_foot', color='b')

                # draw right arm
                if Rs is not None and Re is not None:
                    ax.plot([float(Rs[0]), float(Re[0])], [float(Rs[1]), float(Re[1])], [float(Rs[2]), float(Re[2])], color='r', lw=2)
                    label_line(ax, Rs, Re, 'right_shoulder_elbow', color='r')
                if Re is not None and Rw is not None:
                    ax.plot([float(Re[0]), float(Rw[0])], [float(Re[1]), float(Rw[1])], [float(Re[2]), float(Rw[2])], color='r', lw=2)
                    label_line(ax, Re, Rw, 'right_elbow_wrist', color='r')

                # draw right leg (hip -> knee -> ankle -> foot)
                if Rhip is not None and Rk is not None:
                    ax.plot([float(Rhip[0]), float(Rk[0])], [float(Rhip[1]), float(Rk[1])], [float(Rhip[2]), float(Rk[2])], color='r', lw=2)
                    label_line(ax, Rhip, Rk, 'right_hip_knee', color='r')
                if Rk is not None and Ra is not None:
                    ax.plot([float(Rk[0]), float(Ra[0])], [float(Rk[1]), float(Ra[1])], [float(Rk[2]), float(Ra[2])], color='r', lw=2)
                    label_line(ax, Rk, Ra, 'right_knee_ankle', color='r')
                if Ra is not None and Rf is not None:
                    ax.plot([float(Ra[0]), float(Rf[0])], [float(Ra[1]), float(Rf[1])], [float(Ra[2]), float(Rf[2])], color='r', lw=2)
                    label_line(ax, Ra, Rf, 'right_ankle_foot', color='r')

                # draw head to shoulder midpoint and hips
                if nose is not None and Ls is not None and Rs is not None:
                    sm = shoulder_mid
                    ax.plot([float(nose[0]), float(sm[0])], [float(nose[1]), float(sm[1])], [float(nose[2]), float(sm[2])], color='m', lw=1)
                    label_line(ax, nose, sm, 'head_sh_mid_dist', color='m')
                if Lhip is not None and Ls is not None and Rs is not None:
                    sm = shoulder_mid
                    ax.plot([float(Lhip[0]), float(sm[0])], [float(Lhip[1]), float(sm[1])], [float(Lhip[2]), float(sm[2])], color='c', lw=1)
                    label_line(ax, Lhip, sm, 'left_hip_shoulder_mid_dist', color='c')
                if Rhip is not None and Ls is not None and Rs is not None:
                    sm = shoulder_mid
                    ax.plot([float(Rhip[0]), float(sm[0])], [float(Rhip[1]), float(sm[1])], [float(Rhip[2]), float(sm[2])], color='c', lw=1)
                    label_line(ax, Rhip, sm, 'right_hip_shoulder_mid_dist', color='c')

                # ---- Add joint labels ----
                for txt in ax.texts:
                    try:
                        txt.remove()
                    except Exception:
                        pass
                for j in plot_joints:
                    if j in pts3d_world:
                        p = pts3d_world[j]
                        ax.text(float(p[0]), float(p[1]), float(p[2]), j[:3], color='green', fontsize=8)

                # Add angle text near elbows, knees and shoulders
                try:
                    if 'left_elbow' in pts3d_world:
                        le = pts3d_world['left_elbow']
                        ax.text(float(le[0]), float(le[1])+50, float(le[2]), f"{int(round(left_elbow_angle))}°", color='blue', fontsize=8)
                    if 'right_elbow' in pts3d_world:
                        re = pts3d_world['right_elbow']
                        ax.text(float(re[0]), float(re[1])+50, float(re[2]), f"{int(round(right_elbow_angle))}°", color='red', fontsize=8)
                    if 'left_knee' in pts3d_world:
                        lk = pts3d_world['left_knee']
                        ax.text(float(lk[0]), float(lk[1])+50, float(lk[2]), f"{int(round(left_knee_angle))}°", color='blue', fontsize=8)
                    if 'right_knee' in pts3d_world:
                        rk = pts3d_world['right_knee']
                        ax.text(float(rk[0]), float(rk[1])+50, float(rk[2]), f"{int(round(right_knee_angle))}°", color='red', fontsize=8)
                    if 'left_shoulder' in pts3d_world:
                        ls = pts3d_world['left_shoulder']
                        ax.text(float(ls[0]), float(ls[1])+50, float(ls[2]), f"{int(round(left_shoulder_angle))}°", color='blue', fontsize=8)
                    if 'right_shoulder' in pts3d_world:
                        rs = pts3d_world['right_shoulder']
                        ax.text(float(rs[0]), float(rs[1])+50, float(rs[2]), f"{int(round(right_shoulder_angle))}°", color='red', fontsize=8)
                    # spine angle near shoulder midpoint
                    if shoulder_mid is not None:
                        sm = shoulder_mid
                        ax.text(float(sm[0]), float(sm[1])+50, float(sm[2]), f"{int(round(spine_angle))}°", color='k', fontsize=8)
                except Exception:
                    pass

                # label torso and hip distances (hip baseline)
                if Lhip is not None and Rhip is not None:
                    # hip distance
                    ax.plot([float(Lhip[0]), float(Rhip[0])], [float(Lhip[1]), float(Rhip[1])], [float(Lhip[2]), float(Rhip[2])], color='k', lw=1)
                    label_line(ax, Lhip, Rhip, 'hip_dist', color='k')

                # torso length (shoulder_mid to hip_mid)
                if shoulder_mid is not None and Lhip is not None and Rhip is not None:
                    hip_mid = (Lhip + Rhip) / 2.0
                    ax.plot([float(shoulder_mid[0]), float(hip_mid[0])], [float(shoulder_mid[1]), float(hip_mid[1])], [float(shoulder_mid[2]), float(hip_mid[2])], color='k', lw=1)
                    label_line(ax, shoulder_mid, hip_mid, 'torso_length', color='k')

                # total_height_proxy: draw from nose to feet midpoint
                if nose is not None and Lf is not None and Rf is not None:
                    feet_mid = (Lf + Rf) / 2.0
                    ax.plot([float(nose[0]), float(feet_mid[0])], [float(nose[1]), float(feet_mid[1])], [float(nose[2]), float(feet_mid[2])], color='m', lw=1, linestyle=':')
                    label_line(ax, nose, feet_mid, 'total_height_proxy', color='m')

                # Autoscale axes to visible points (with padding)
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
    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    pose.close()
    plt.ioff()
    # Close output file handle
    try:
        if f_out is not None:
            f_out.close()
    except Exception:
        pass

    print("Exited cleanly.")
