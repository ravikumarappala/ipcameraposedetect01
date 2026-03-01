#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  setupenv-macbook.sh
#  Master reference for ALL commands to set up and run this project
#  on a MacBook. Run each section once in order.
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# SECTION 1 – Install Miniconda (skip if already installed)
# ──────────────────────────────────────────────────────────────────

# Download Miniconda installer for macOS (Apple Silicon / Intel auto-detect)
# Apple Silicon (M1/M2/M3):
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p "$HOME/miniconda3"

# Intel Mac:
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
# bash Miniconda3-latest-MacOSX-x86_64.sh -b -p "$HOME/miniconda3"

# Initialise conda in your shell (adds conda to PATH permanently)
# NOTE: silent install (-b flag) skips this step — must be done manually once
"$HOME/miniconda3/bin/conda" init zsh    # use 'bash' if you're on bash
source ~/.zshrc                          # reload shell config
# After this, open a NEW terminal and 'conda' will be available in PATH

# Verify conda is available
conda --version


# ──────────────────────────────────────────────────────────────────
# SECTION 2 – Create the conda environment  (first time only)
# ──────────────────────────────────────────────────────────────────

# Navigate to project directory
cd /Users/ravia/cb-poc/moiq/ipcameraposedetect01

# Create env from environment.yml  (Python 3.10 + all deps)
conda env create --name posedetect --file environment.yml

# Install tk for matplotlib TkAgg backend on macOS
conda activate posedetect
conda install -y tk -c conda-forge

# Verify key packages
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import cv2; print('opencv', cv2.__version__)"
python -c "import mediapipe; print('mediapipe', mediapipe.__version__)"
python -c "import matplotlib; print('matplotlib', matplotlib.__version__)"

# ⚠️  IMPORTANT – numpy pin (macOS fix, applied 2026-02-28):
# pip-built opencv wheels are compiled against numpy 1.x ABI.
# numpy 2.x (from conda-forge) causes: "numpy.core.multiarray failed to import"
# Fix: pin numpy to 1.26.4 (already baked into environment.yml)

# ⚠️  IMPORTANT – mediapipe version (macOS fix, applied 2026-02-28):
# mediapipe 0.10.11 has a graph config mismatch (ValidatedGraphConfig Initialization failed)
# Fix: use mediapipe 0.10.7 (already baked into environment.yml)

# ⚠️  step7_rendering.py – matplotlib API fix (applied 2026-02-28):
# matplotlib 3.10 removed fig.canvas.tostring_rgb()
# Fix: replaced with fig.canvas.buffer_rgba() in step7_rendering.py


# ──────────────────────────────────────────────────────────────────
# SECTION 3 – Update the environment  (when environment.yml changes)
# ──────────────────────────────────────────────────────────────────

conda env update --name posedetect --file environment.yml --prune


# ──────────────────────────────────────────────────────────────────
# SECTION 4 – Activate env (every new terminal session)
# ──────────────────────────────────────────────────────────────────

conda activate posedetect
cd /Users/ravia/cb-poc/moiq/ipcameraposedetect01


# ──────────────────────────────────────────────────────────────────
# SECTION 5 – Grant camera access (macOS)
# ──────────────────────────────────────────────────────────────────
# Open System Settings → Privacy & Security → Camera
# Enable access for Terminal (or iTerm2 / VS Code).
# This only needs to be done once; a dialog will also appear
# the first time a script opens a camera.

open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"


# ──────────────────────────────────────────────────────────────────
# SECTION 6 – Run the scripts
# ──────────────────────────────────────────────────────────────────

# (a) Find/probe connected USB or RTSP cameras
python probe_cameras.py

# (b) Test opening both cameras (USB indices or RTSP URLs)
#     USB webcams (macOS device indices start at 0)
python open_one_camera.py --camA 0 --camB 1 --width 1280 --height 720 --fps 30

#     IP / RTSP cameras on local network
python open_one_camera.py \
  --camA "rtsp://admin:Test12345@172.16.1.11:554/11" \
  --camB "rtsp://admin:Test12345@172.16.1.13:554/11"

# (c) Capture stereo calibration images (3-phase: left-only, right-only, shared)
python capture_calibration_90deg.py

# (d) Run stereo calibration and save stereo_params.npz
python stereo_calibration_90deg.py

# (e) Print / verify calibration parameters
python print_camera_params.py

# (f) Run real-time full-body 3D pose tracking
python realtime_body_3d.py

# (g) Run real-time arm-only 3D tracking
python realtime_arm_3d.py


# ──────────────────────────────────────────────────────────────────
# SECTION 7 – macOS-specific code fixes already applied
# ──────────────────────────────────────────────────────────────────
# realtime_body_3d.py:
#   - Removed cv2.CAP_DSHOW  (Windows-only DirectShow backend)
#   - Added: import matplotlib; matplotlib.use('TkAgg')  before pyplot
#
# open_one_camera.py:
#   - Uses cv2.CAP_V4L2 for /dev/videoX paths (Linux only).
#     On macOS, pass integer indices (0,1,2) or rtsp:// URLs instead.


# ──────────────────────────────────────────────────────────────────
# SECTION 8 – Remove / recreate the environment (if needed)
# ──────────────────────────────────────────────────────────────────

conda deactivate
conda env remove --name posedetect
