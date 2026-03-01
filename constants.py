"""
Shared constants for the SMPL Measurement Pipeline.
These are referenced by multiple step files but rarely change.
"""

# SMPL joint names (24 joints)
SMPL_JOINT_NAMES = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee",
    "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
    "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"
]

# MediaPipe to SMPL joint mapping
MP_TO_SMPL = {
    0:  15,  # nose -> head
    11: 16,  # left_shoulder -> l_shoulder
    12: 17,  # right_shoulder -> r_shoulder
    13: 18,  # left_elbow -> l_elbow
    14: 19,  # right_elbow -> r_elbow
    15: 20,  # left_wrist -> l_wrist
    16: 21,  # right_wrist -> r_wrist
    17: 22,  # left_pinky -> l_hand
    18: 23,  # right_pinky -> r_hand
    23: 1,   # left_hip -> l_hip
    24: 2,   # right_hip -> r_hip
    25: 4,   # left_knee -> l_knee
    26: 5,   # right_knee -> r_knee
    27: 7,   # left_ankle -> l_ankle
    28: 8,   # right_ankle -> r_ankle
    29: 10,  # left_heel -> l_foot
    30: 11,  # right_heel -> r_foot
}

# MediaPipe landmark pairs averaged to derive SMPL joints
MP_MIDPOINT_TO_SMPL = {
    (11, 12): 12,  # midpoint(l_shoulder, r_shoulder) -> neck
    (23, 24): 0,   # midpoint(l_hip, r_hip) -> pelvis
}

# Measurement definitions: (name, joint1, joint2)
SEGMENT_MEASUREMENTS = [
    ("Shoulder Width", "l_shoulder", "r_shoulder"),
    ("Left Upper Arm", "l_shoulder", "l_elbow"),
    ("Left Forearm", "l_elbow", "l_wrist"),
    ("Left Full Arm", "l_shoulder", "l_wrist"),
    ("Right Upper Arm", "r_shoulder", "r_elbow"),
    ("Right Forearm", "r_elbow", "r_wrist"),
    ("Right Full Arm", "r_shoulder", "r_wrist"),
    ("Neck to Head", "neck", "head"),
    ("Torso (Pelvis to Neck)", "pelvis", "neck"),
    ("Spine (Pelvis to Spine3)", "pelvis", "spine3"),
    ("Hip Width", "l_hip", "r_hip"),
    ("Left Thigh", "l_hip", "l_knee"),
    ("Left Shin", "l_knee", "l_ankle"),
    ("Left Full Leg", "l_hip", "l_ankle"),
    ("Right Thigh", "r_hip", "r_knee"),
    ("Right Shin", "r_knee", "r_ankle"),
    ("Right Full Leg", "r_hip", "r_ankle"),
]

# Height path: from ankle up through body
HEIGHT_PATH = ["l_ankle", "l_knee", "l_hip", "pelvis", "spine1", "spine2", "spine3", "neck", "head"]

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12),
    (12, 13), (12, 14),
    (13, 16), (16, 18), (18, 20), (20, 22),
    (14, 17), (17, 19), (19, 21), (21, 23),
    (12, 15),
]

# Default paths
import os as _os
_PROJECT_DIR = _os.path.dirname(_os.path.abspath(__file__))
SMPL_MODEL_PATH = _os.path.join(_PROJECT_DIR, "SMPL", "SMPL_NEUTRAL.pkl")
DEFAULT_CALIB_PATH = "stereo_params_sidebyside.npz"
