#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  setup_mac.sh  –  one-shot Miniconda env setup for posedetect
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="posedetect"

echo "=============================================="
echo "  posedetect – Miniconda Environment Setup"
echo "=============================================="

# 1. Make sure conda is available
if ! command -v conda &> /dev/null; then
    echo ""
    echo "ERROR: conda not found."
    echo "Please install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[1/4] conda found: $(conda --version)"

# 2. Create (or update) the environment from environment.yml
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[2/4] Environment '${ENV_NAME}' already exists – updating..."
    conda env update --name "${ENV_NAME}" --file "${SCRIPT_DIR}/environment.yml" --prune
else
    echo "[2/4] Creating conda environment '${ENV_NAME}'..."
    conda env create --name "${ENV_NAME}" --file "${SCRIPT_DIR}/environment.yml"
fi

# 3. Install tk (needed for matplotlib TkAgg backend on macOS)
echo "[3/4] Installing tk for matplotlib TkAgg backend..."
conda run -n "${ENV_NAME}" conda install -y tk -c conda-forge 2>/dev/null || true

echo "[4/4] Done!"
echo ""
echo "=============================================="
echo "  Activate with:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  Then run:"
echo "    python open_one_camera.py        # test cameras"
echo "    python realtime_body_3d.py       # full pose tracking"
echo "=============================================="
