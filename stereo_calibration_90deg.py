"""
Stereo Calibration for 90-degree Camera Setup

Uses images captured in 3 phases:
1. LEFT camera only images -> for left intrinsics
2. RIGHT camera only images -> for right intrinsics
3. SHARED images -> for stereo extrinsics (R, T)
"""

import cv2
import numpy as np
import glob
import os
import math

# --- Checkerboard settings ---
CHECKERBOARD = (9, 7)   # INTERNAL corners (count the inner corners!)
SQUARE_SIZE = 50.0      # mm - measure your actual square size!

# --- Image folders ---

# Folder names as produced by capture_calibration_90deg.py
SAVE_DIR_LEFT_ONLY = "calib_left_only"
SAVE_DIR_LEFT_ONLY_RIGHT_IGNORE = "calib_left_only_right_ignore"
SAVE_DIR_RIGHT_ONLY = "calib_right_only"
SAVE_DIR_RIGHT_ONLY_LEFT_IGNORE = "calib_right_only_left_ignore"
SAVE_DIR_SHARED_LEFT = "calib_shared_left"
SAVE_DIR_SHARED_RIGHT = "calib_shared_right"

# Fallback to old folders if new ones don't exist
SAVE_DIR_LEFT = "calib_left"
SAVE_DIR_RIGHT = "calib_right"

# --- Prepare 3D object points ---
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE


def detect_corners(image_paths, camera_name, min_size_ratio=0.1):
    """Detect checkerboard corners in images"""
    objpoints = []
    imgpoints = []
    img_size = None
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Check corner spread
            c = corners.reshape(-1, 2)
            width = c[:, 0].max() - c[:, 0].min()
            
            if width > img_size[0] * min_size_ratio:
                objpoints.append(objp)
                imgpoints.append(corners)
                print(f"  [{len(objpoints):2d}] {os.path.basename(img_path)}: {width:.0f}px wide")
    
    return objpoints, imgpoints, img_size


def main():
    print("="*60)
    print("STEREO CALIBRATION FOR 90° CAMERA SETUP")
    print("="*60)
    
    # Check which folder structure exists
    use_separate_folders = os.path.exists(SAVE_DIR_LEFT_ONLY) and os.path.exists(SAVE_DIR_RIGHT_ONLY)
    
    if use_separate_folders:
        print("\nUsing 3-phase capture folders:")
        print(f"  LEFT only:  {SAVE_DIR_LEFT_ONLY}/")
        print(f"  LEFT only right ignore:  {SAVE_DIR_LEFT_ONLY_RIGHT_IGNORE}/")
        print(f"  RIGHT only: {SAVE_DIR_RIGHT_ONLY}/")
        print(f"  RIGHT only left ignore: {SAVE_DIR_RIGHT_ONLY_LEFT_IGNORE}/")
        print(f"  SHARED:     {SAVE_DIR_SHARED_LEFT}/ & {SAVE_DIR_SHARED_RIGHT}/")

        # Use only the correct folders for calibration
        left_only_images = sorted(glob.glob(f"{SAVE_DIR_LEFT_ONLY}/*.png"))
        right_only_images = sorted(glob.glob(f"{SAVE_DIR_RIGHT_ONLY}/*.png"))
        shared_left_images = sorted(glob.glob(f"{SAVE_DIR_SHARED_LEFT}/*.png"))
        shared_right_images = sorted(glob.glob(f"{SAVE_DIR_SHARED_RIGHT}/*.png"))

        # Ignore *_ignore folders (they are not used for calibration)

        print(f"\nFound: {len(left_only_images)} left-only, {len(right_only_images)} right-only, {len(shared_left_images)} shared pairs")
    else:
        print("\nUsing combined folders (old style):")
        print(f"  {SAVE_DIR_LEFT}/ and {SAVE_DIR_RIGHT}/")
        
        # Use all images for both intrinsics and stereo
        left_only_images = sorted(glob.glob(f"{SAVE_DIR_LEFT}/*.png"))
        right_only_images = sorted(glob.glob(f"{SAVE_DIR_RIGHT}/*.png"))
        shared_left_images = left_only_images
        shared_right_images = right_only_images
        
        print(f"\nFound: {len(left_only_images)} left, {len(right_only_images)} right")
    
    # ============================================================
    # STEP 1: Calibrate LEFT camera
    # ============================================================
    print("\n" + "="*60)
    print("STEP 1: LEFT Camera Intrinsic Calibration")
    print("="*60)
    
    # Combine left-only and shared images for left intrinsics
    all_left_images = left_only_images + shared_left_images
    print(f"\nUsing {len(all_left_images)} images for LEFT camera")
    
    objpoints_left, imgpoints_left, left_size = detect_corners(all_left_images, "LEFT")
    
    if len(objpoints_left) < 10:
        print(f"WARNING: Only {len(objpoints_left)} valid images for LEFT camera")
    
    print(f"\nCalibrating LEFT camera with {len(objpoints_left)} images...")
    retL, K1, dist1, _, _ = cv2.calibrateCamera(
        objpoints_left, imgpoints_left, left_size, None, None
    )
    
    print(f"  RMS Error: {retL:.4f}")
    print(f"  fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f}")
    print(f"  cx={K1[0,2]:.1f}, cy={K1[1,2]:.1f}")
    print(f"  fx/fy ratio: {K1[0,0]/K1[1,1]:.4f} (should be ~1.0)")
    
    # ============================================================
    # STEP 2: Calibrate RIGHT camera
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: RIGHT Camera Intrinsic Calibration")
    print("="*60)
    
    # Combine right-only and shared images for right intrinsics
    all_right_images = right_only_images + shared_right_images
    print(f"\nUsing {len(all_right_images)} images for RIGHT camera")
    
    objpoints_right, imgpoints_right, right_size = detect_corners(all_right_images, "RIGHT")
    
    if len(objpoints_right) < 10:
        print(f"WARNING: Only {len(objpoints_right)} valid images for RIGHT camera")
    
    print(f"\nCalibrating RIGHT camera with {len(objpoints_right)} images...")
    retR, K2, dist2, _, _ = cv2.calibrateCamera(
        objpoints_right, imgpoints_right, right_size, None, None
    )
    
    print(f"  RMS Error: {retR:.4f}")
    print(f"  fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f}")
    print(f"  cx={K2[0,2]:.1f}, cy={K2[1,2]:.1f}")
    print(f"  fx/fy ratio: {K2[0,0]/K2[1,1]:.4f} (should be ~1.0)")
    
    # Validate intrinsics
    left_ok = abs(K1[0,0]/K1[1,1] - 1.0) < 0.1
    right_ok = abs(K2[0,0]/K2[1,1] - 1.0) < 0.1
    
    if not left_ok:
        print("\n⚠ WARNING: LEFT camera fx/fy ratio is off!")
    if not right_ok:
        print("\n⚠ WARNING: RIGHT camera fx/fy ratio is off!")
    
    # ============================================================
    # STEP 3: Stereo Calibration using SHARED images only
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: Stereo Calibration (R, T)")
    print("="*60)
    
    # Use only shared images for stereo calibration
    print(f"\nUsing {len(shared_left_images)} shared image pairs")
    
    objpoints_stereo = []
    imgpoints_left_stereo = []
    imgpoints_right_stereo = []
    
    for lp, rp in zip(shared_left_images, shared_right_images):
        imgL = cv2.imread(lp)
        imgR = cv2.imread(rp)
        
        if imgL is None or imgR is None:
            continue
        
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        
        retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, flags)
        retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, flags)
        
        if retL and retR:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            
            objpoints_stereo.append(objp)
            imgpoints_left_stereo.append(cornersL)
            imgpoints_right_stereo.append(cornersR)
            print(f"  [OK] {os.path.basename(lp)}")
        else:
            print(f"  [--] {os.path.basename(lp)} - L:{'OK' if retL else 'FAIL'} R:{'OK' if retR else 'FAIL'}")
    
    print(f"\nValid stereo pairs: {len(objpoints_stereo)}")
    
    if len(objpoints_stereo) < 5:
        print("ERROR: Not enough valid stereo pairs!")
        print("Make sure checkerboard is visible to BOTH cameras in shared images")
        exit(1)
    
    # Fix intrinsics, only find R and T
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    
    print("\nRunning stereo calibration...")
    retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints_stereo,
        imgpoints_left_stereo,
        imgpoints_right_stereo,
        K1, dist1,
        K2, dist2,
        left_size,
        criteria=criteria,
        flags=stereo_flags
    )
    
    print(f"\nStereo RMS Error: {retS:.4f}")
    
    # Compute baseline and angle
    baseline = np.linalg.norm(T)
    angle = math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / math.pi
    
    print(f"Baseline: {baseline:.1f} mm ({baseline/1000:.2f} m)")
    print(f"Camera angle: {angle:.1f}°")
    
    # ============================================================
    # STEP 4: Compute Projection Matrices
    # ============================================================
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))
    
    # ============================================================
    # SAVE
    # ============================================================
    np.savez("stereo_params.npz",
             K1=K1, dist1=dist1,
             K2=K2, dist2=dist2,
             R=R, T=T,
             P1=P1, P2=P2,
             E=E, F=F,
             img_size=np.array(left_size))
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    print(f"Left camera:")
    print(f"  Images used: {len(objpoints_left)}")
    print(f"  RMS Error:   {retL:.4f}")
    print(f"  fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f}, ratio={K1[0,0]/K1[1,1]:.3f}")
    
    print(f"\nRight camera:")
    print(f"  Images used: {len(objpoints_right)}")
    print(f"  RMS Error:   {retR:.4f}")
    print(f"  fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f}, ratio={K2[0,0]/K2[1,1]:.3f}")
    
    print(f"\nStereo:")
    print(f"  Pairs used:  {len(objpoints_stereo)}")
    print(f"  RMS Error:   {retS:.4f}")
    print(f"  Baseline:    {baseline:.1f} mm")
    print(f"  Angle:       {angle:.1f}°")
    
    print("\nR (rotation):")
    print(R)
    
    print("\nT (translation in mm):")
    print(T.flatten())
    
    # Quality assessment
    print("\n" + "-"*60)
    if retS < 1.0 and left_ok and right_ok:
        print("✓ CALIBRATION LOOKS GOOD!")
    elif retS < 2.0 and left_ok and right_ok:
        print("⚠ CALIBRATION IS OK - usable but could be improved")
    else:
        print("✗ CALIBRATION QUALITY IS POOR")
        if not left_ok:
            print("  - Left camera intrinsics are off")
        if not right_ok:
            print("  - Right camera intrinsics are off")
        if retS >= 2.0:
            print("  - Stereo RMS error is high")
    print("-"*60)
    
    print("\nSaved to stereo_params.npz")


if __name__ == "__main__":
    main()
