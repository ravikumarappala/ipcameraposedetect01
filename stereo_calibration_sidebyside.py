"""
Stereo Calibration for Side-by-Side Camera Setup

This script performs stereo calibration for a horizontal camera setup where
both cameras are mounted on the same wall, approximately 4 feet (1.2m) apart,
facing the same direction.

Improvements:
1. Enforces consistent image sizes (rejects images with different resolutions)
2. Performs iterative stereo calibration with outlier rejection
3. Outputs baseline distance in both mm and feet

Input: Images from calib_ss_left/ and calib_ss_right/
Output: stereo_params_sidebyside.npz
"""

import cv2
import numpy as np
import glob
import os
import math
import sys

# --- Checkerboard settings ---
CHECKERBOARD = (18,29) #TERNAL corners
SQUARE_SIZE = 50.0      # mm

# --- Image folders (side-by-side setup) ---
SAVE_DIR_LEFT = "calib_ss_left"
SAVE_DIR_RIGHT = "calib_ss_right"

# Calibration Flags
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK


def get_image_size(image_paths):
    """Scan images to find the most common resolution"""
    sizes = {}
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = (w, h)
        sizes[s] = sizes.get(s, 0) + 1
    
    if not sizes:
        return None
    
    # Return most frequent size
    best_size = max(sizes, key=sizes.get)
    print(f"DEBUG: Found image sizes: {sizes}. Selected: {best_size}")
    return best_size


def detect_corners(image_paths, board_size, square_size, required_size=None):
    """
    Detect corners in a list of images.
    Filters out images that do not match required_size (width, height).
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    valid_paths = []
    
    print(f"Scanning {len(image_paths)} images...")
    
    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        cur_size = (w, h)
        
        if required_size is not None and cur_size != required_size:
            print(f"  [SKIP] {os.path.basename(fname)}: Size {cur_size} != {required_size}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, FLAGS)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            imgpoints.append(corners2)
            objpoints.append(objp)
            valid_paths.append(fname)
        else:
            print(f"  [FAIL] {os.path.basename(fname)} (Corner detection failed)")
            
    return objpoints, imgpoints, valid_paths


def calibrate_camera_intrinsics(image_paths, cam_name, expected_size):
    """Calibrate single camera intrinsics"""
    print(f"\n--- Calibrating {cam_name} Intrinsics ---")
    objp, imgp, valid = detect_corners(image_paths, CHECKERBOARD, SQUARE_SIZE, expected_size)
    
    if len(valid) < 5:
        print(f"ERROR: Not enough valid images for {cam_name}!")
        return None, None, None, None
    
    print(f"Calibrating {cam_name} with {len(valid)} images (Size: {expected_size})...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, expected_size, None, None)
    
    print(f"{cam_name} RMS: {ret:.4f}")
    print(f"{cam_name} Camera Matrix:\n{K}")
    return K, dist, objp, imgp


def filter_stereo_outliers(objp_list, imgpL_list, imgpR_list, pair_names, K1, D1, K2, D2, img_size, threshold=2.0):
    """
    Iteratively run stereo calibration and remove worst offenders.
    Uses Epipolar Error as the metric.
    """
    
    current_objp = list(objp_list)
    current_imgpL = list(imgpL_list)
    current_imgpR = list(imgpR_list)
    current_names = list(pair_names)
    removed_pairs = []
    
    # Initial Calibration
    print(f"\nInitial Stereo Calibration with {len(current_objp)} pairs...")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC
    
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        current_objp, current_imgpL, current_imgpR,
        K1, D1, K2, D2, img_size,
        criteria=criteria, flags=flags
    )
    print(f"  -> Initial RMS: {ret:.4f}")
    
    # Iterative outlier removal loop
    MAX_ITER = 10
    
    for iteration in range(MAX_ITER):
        if ret < 1.0:
            print("  -> RMS is good (< 1.0). Stopping.")
            break
            
        # Calculate epipolar errors for each pair
        pair_errors = []
        for i in range(len(current_imgpL)):
            pts1 = current_imgpL[i].reshape(-1, 2)
            pts2 = current_imgpR[i].reshape(-1, 2)
            
            # Compute epipolar lines from right points to left image
            lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
            lines1 = lines1.reshape(-1, 3)
            
            # Distance of left points to corresponding epipolar lines
            errs = []
            for j in range(len(pts1)):
                l = lines1[j]
                p = pts1[j]
                d = abs(l[0]*p[0] + l[1]*p[1] + l[2]) / np.sqrt(l[0]**2 + l[1]**2)
                errs.append(d)
                
            mean_error = np.mean(errs)
            pair_errors.append(mean_error)
            
        pair_errors = np.array(pair_errors)
        
        # Identify worst outlier
        worst_idx = np.argmax(pair_errors)
        worst_err = pair_errors[worst_idx]
        
        # Stop if worst error is acceptable
        if worst_err < threshold and ret < 3.0:
            break
             
        # Remove worst pair
        removed_name = current_names[worst_idx]
        removed_pairs.append(removed_name)
        print(f"  Iter {iteration+1}: Removing '{removed_name}' (Epipolar Err: {worst_err:.4f})")
        
        current_objp.pop(worst_idx)
        current_imgpL.pop(worst_idx)
        current_imgpR.pop(worst_idx)
        current_names.pop(worst_idx)
        
        if len(current_objp) < 5:
            print("  -> Too few pairs remaining. Stopping.")
            break
            
        # Re-calibrate
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            current_objp, current_imgpL, current_imgpR,
            K1, D1, K2, D2, img_size,
            criteria=criteria, flags=flags
        )
        print(f"  -> New RMS: {ret:.4f}")
    
    # Print summary of removed files
    if removed_pairs:
        print(f"\n  --- REMOVED {len(removed_pairs)} BAD PAIRS ---")
        for name in removed_pairs:
            print(f"    DELETE: {SAVE_DIR_LEFT}/{name} and {SAVE_DIR_RIGHT}/{name}")
        
    return ret, K1, D1, K2, D2, R, T, E, F, len(current_objp)


def main():
    print("="*60)
    print("SIDE-BY-SIDE STEREO CALIBRATION")
    print("="*60)
    print("\nExpected Setup:")
    print("  - Both cameras on same wall, ~4 feet apart")
    print("  - Both cameras facing the same direction")
    print()
    
    # 1. Scan for image paths
    paths_left = sorted(glob.glob(f"{SAVE_DIR_LEFT}/calib_ss_*.png"))
    paths_right = sorted(glob.glob(f"{SAVE_DIR_RIGHT}/calib_ss_*.png"))
    
    if not paths_left:
        # Try alternate pattern
        paths_left = sorted(glob.glob(f"{SAVE_DIR_LEFT}/*.png"))
        paths_right = sorted(glob.glob(f"{SAVE_DIR_RIGHT}/*.png"))
    
    print(f"Found {len(paths_left)} left images, {len(paths_right)} right images")
    
    all_imgs = paths_left + paths_right
    if not all_imgs:
        print("No images found! Run capture_calibration_sidebyside.py first.")
        sys.exit(1)
        
    # 2. Determine Primary Resolution
    common_size = get_image_size(all_imgs)
    print(f"Using Resolution: {common_size}")
    
    # 3. Intrinsic Calibration for each camera
    K1, D1, objL, imgL = calibrate_camera_intrinsics(paths_left, "LEFT", common_size)
    K2, D2, objR, imgR = calibrate_camera_intrinsics(paths_right, "RIGHT", common_size)
    
    if K1 is None or K2 is None:
        print("Intrinsic calibration failed.")
        sys.exit(1)

    # 4. Prepare Stereo Pairs (match by filename)
    print("\nMatching stereo pairs...")
    
    left_dict = {os.path.basename(p): p for p in paths_left}
    right_dict = {os.path.basename(p): p for p in paths_right}
    
    common_names = sorted(list(set(left_dict.keys()) & set(right_dict.keys())))
    
    stereo_obj = []
    stereo_imgL = []
    stereo_imgR = []
    stereo_names = []
    
    valid_pair_count = 0
    
    for name in common_names:
        pL = left_dict[name]
        pR = right_dict[name]
        
        imgL_chk = cv2.imread(pL)
        imgR_chk = cv2.imread(pR)
        
        if imgL_chk is None or imgR_chk is None:
            continue
            
        if imgL_chk.shape[:2][::-1] != common_size or imgR_chk.shape[:2][::-1] != common_size:
            continue
            
        grayL = cv2.cvtColor(imgL_chk, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR_chk, cv2.COLOR_BGR2GRAY)
        
        retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, FLAGS)
        retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, FLAGS)
        
        if retL and retR:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), crit)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), crit)
            
            obj_pt = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
            obj_pt[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
            obj_pt *= SQUARE_SIZE
            
            stereo_obj.append(obj_pt)
            stereo_imgL.append(cornersL)
            stereo_imgR.append(cornersR)
            stereo_names.append(name)
            valid_pair_count += 1
        else:
            if not retL:
                print(f"  [FAIL L] {name}")
            if not retR:
                print(f"  [FAIL R] {name}")
            
    print(f"Found {valid_pair_count} valid stereo pairs.")
    
    if valid_pair_count < 5:
        print("Not enough pairs for stereo calibration.")
        sys.exit(1)
        
    # 5. Robust Stereo Calibration with outlier rejection
    retS, K1, D1, K2, D2, R, T, E, F, final_count = filter_stereo_outliers(
        stereo_obj, stereo_imgL, stereo_imgR, stereo_names, K1, D1, K2, D2, common_size
    )
    
    # Calculate baseline
    baseline_mm = np.linalg.norm(T)
    baseline_ft = baseline_mm / 304.8  # mm to feet
    baseline_m = baseline_mm / 1000.0
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Stereo Pairs Used: {final_count} (from {valid_pair_count})")
    print(f"Final Stereo RMS: {retS:.4f}")
    print(f"\nBaseline (camera separation):")
    print(f"  {baseline_mm:.2f} mm")
    print(f"  {baseline_m:.3f} m")
    print(f"  {baseline_ft:.2f} feet")
    
    # Rotation check (should be near identity for side-by-side)
    rot_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
    print(f"\nRotation between cameras: {rot_angle:.2f} degrees")
    if rot_angle > 10:
        print("  WARNING: High rotation angle - cameras may not be aligned")
    
    # Translation direction (should be mostly horizontal for side-by-side)
    T_norm = T.flatten() / np.linalg.norm(T)
    print(f"Translation direction (normalized): X={T_norm[0]:.3f}, Y={T_norm[1]:.3f}, Z={T_norm[2]:.3f}")
    
    # For side-by-side setup, translation should be mostly in X direction
    if abs(T_norm[0]) > 0.9:
        print("  GOOD: Cameras are aligned horizontally (X-dominant translation)")
    elif abs(T_norm[1]) > 0.9:
        print("  INFO: Translation is vertical - cameras may be stacked vertically")
    else:
        print("  WARNING: Translation has significant Z component - cameras may be angled")
    
    # Save calibration
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))
    
    output_file = "stereo_params_sidebyside.npz"
    np.savez(output_file,
             K1=K1, dist1=D1,
             K2=K2, dist2=D2,
             R=R, T=T,
             P1=P1, P2=P2,
             E=E, F=F,
             img_size=common_size,
             baseline_mm=baseline_mm,
             stereo_rms=retS)
    print(f"\nSaved to {output_file}")
    
    # Also print intrinsics for reference
    print("\n--- Left Camera Intrinsics ---")
    print(f"K1 = \n{K1}")
    print(f"dist1 = {D1.flatten()}")
    
    print("\n--- Right Camera Intrinsics ---")
    print(f"K2 = \n{K2}")
    print(f"dist2 = {D2.flatten()}")
    
    print("\n--- Extrinsics (Right w.r.t. Left) ---")
    print(f"R = \n{R}")
    print(f"T = {T.flatten()} mm")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
