# Detailed Biomechanical Analysis Tool - Usage Guide

## Overview
`image_body_3d_det.py` provides comprehensive biomechanical analysis with:
- Triangle-based segment measurements (as per diagram)
- All joint angles
- Both MediaPipe and HRNet support
- Height-based joint scaling

## Usage

### Basic Usage
```bash
# Using MediaPipe (default, faster)
python3 image_body_3d_det.py --num 1

# Using HRNet (more accurate)
python3 image_body_3d_det.py --num 1 --model hrnet

# With height scaling (168 cm example)
python3 image_body_3d_det.py --num 1 --height 168

# All combined
python3 image_body_3d_det.py --num 1 --model hrnet --height 168
```

### Parameters
- `--num, -n`: Image number suffix (e.g., 1 for test_ss_left1.png)
- `--model, -m`: Pose model - 'mediapipe' (default) or 'hrnet'
- `--height`: True height in cm for scaling (optional)

## Measurements Calculated

### Triangle-Based Segments (from diagram)
**Triangle 1 & 2: Head/Shoulders**
- Head to L/R Shoulder
- Shoulder Width

**Triangle 3: Torso**
- L/R Shoulder to L/R Hip (all combinations)
- Torso Length
- Hip Width

**Triangle 4: Thighs**
- L/R Hip to opposite Knee
- L/R Femur (thigh)

**Triangle 5 & 6: Lower Legs**
- L/R Tibia (shin)
- Ankle segments (MediaPipe)
- Foot segments (MediaPipe)

### All Joint Angles
- Shoulder angles (L/R)
- Elbow angles (L/R)
- Hip angles (L/R)
- Knee angles (L/R)
- Ankle angles (L/R, MediaPipe only)

### Additional Measurements
- Upper/Lower arm segments
- Hand length (MediaPipe only)
- Complete foot measurements (MediaPipe only)
- Height estimation

## Model Comparison

| Feature | MediaPipe | HRNet |
|---------|-----------|-------|
| **Speed** | Fast (~1s) | Slower (~5-10s) |
| **Accuracy** | ±2-5 pixels | ±1-2 pixels |
| **Keypoints** | 33 points | 17 points |
| **Hands/Feet** | Yes (detailed) | No |
| **Best For** | Quick analysis, foot details | High precision |

## Outputs

All results saved to `biomech_analysis_<model>/`:
- CSV file with all measurements and angles
- Organized by type (lengths vs angles)
- Timestamped filenames

## Example Workflow

```bash
# 1. Capture test images
python3 capture_test_images_sidebyside.py

# 2. Quick MediaPipe analysis
python3 image_body_3d_det.py --num 1

# 3. Precise HRNet analysis with height correction
python3 image_body_3d_det.py --num 1 --model hrnet --height 168

# 4. Check results
cat biomech_analysis_hrnet/measurements_*.csv
```

## Notes

- **Height scaling**: If provided, all 3D joint positions are scaled proportionally
- **Missing keypoints**: Some measurements may not appear if keypoints weren't detected
- **HRNet setup**: Requires HRNet installation at `/media/raviappala/edgeextvol2/jetson-inference/3dpose/HRnet_0.1`
- **Conda environment**: Activate `frfd01` environment before running
