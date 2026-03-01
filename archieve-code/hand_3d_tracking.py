import cv2
import mediapipe as mp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# -------------------------
# Load stereo calibration
# -------------------------
data = np.load("stereo_params.npz")

P1 = data["P1"]
P2 = data["P2"]

# MediaPipe initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# -------------------------
# Utility: triangulation
# -------------------------
def triangulate(pt_left, pt_right):
    """
    pt_left  = (x,y)
    pt_right = (x,y)
    returns (X,Y,Z)
    """
    pts_4d = cv2.triangulatePoints(P1, P2,
                                   np.array(pt_left, dtype=np.float32).reshape(2,1),
                                   np.array(pt_right, dtype=np.float32).reshape(2,1))
    pts_4d /= pts_4d[3]  # convert from homogeneous
    return pts_4d[:3].flatten()


# -------------------------
# Extract required joints
# -------------------------
def detect_joints(img):
    h, w = img.shape[:2]

    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark

    # Right side (shoulder 12, elbow 14, wrist 16)
    joints = {
        "shoulder": (int(lm[12].x * w), int(lm[12].y * h)),
        "elbow":    (int(lm[14].x * w), int(lm[14].y * h)),
        "wrist":    (int(lm[16].x * w), int(lm[16].y * h))
    }
    return joints


# -------------------------
# 3D plotting
# -------------------------
def plot_3d(pts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    S, E, W = pts["shoulder"], pts["elbow"], pts["wrist"]

    X = [S[0], E[0], W[0]]
    Y = [S[1], E[1], W[1]]
    Z = [S[2], E[2], W[2]]

    ax.plot(X, Y, Z, marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Stereo 3D Arm Reconstruction")

    plt.show()


# -------------------------
# Camera setup
# -------------------------
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(1)

while True:
    retL, imgL = capL.read()
    retR, imgR = capR.read()
    if not retL or not retR:
        print("Camera error")
        break

    jointsL = detect_joints(imgL)
    jointsR = detect_joints(imgR)

    if jointsL and jointsR:
        shoulder_3d = triangulate(jointsL["shoulder"], jointsR["shoulder"])
        elbow_3d    = triangulate(jointsL["elbow"],    jointsR["elbow"])
        wrist_3d    = triangulate(jointsL["wrist"],    jointsR["wrist"])

        pts = {
            "shoulder": shoulder_3d,
            "elbow": elbow_3d,
            "wrist": wrist_3d
        }

        print("3D Shoulder:", shoulder_3d)
        print("3D Elbow   :", elbow_3d)
        print("3D Wrist   :", wrist_3d)

        # Draw 2D points on screen for visual debugging
        for k,v in jointsL.items():
            cv2.circle(imgL, v, 5, (0,255,0), -1)
        for k,v in jointsR.items():
            cv2.circle(imgR, v, 5, (0,255,0), -1)

        cv2.imshow("Left", imgL)
        cv2.imshow("Right", imgR)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
