
"""
Stereo Calibration for 90-degree Camera Setup (Modified)

 Improvements:
 1. Enforces consistent image sizes (rejects images with different resolutions).
 2. Performs iterative stereo calibration with outlier rejection (removes pairs with high Reprojection Error).
 3. Validates intrinsics logic.
"""

import cv2
import numpy as np
import glob
import os
import math
import sys

    # --- Checkerboard settings ---
    CHECKERBOARD = (9,7)   # INTERNAL corners
    SQUARE_SIZE = 50.0      # mm

# --- Image folders ---
SAVE_DIR_LEFT_ONLY = "calib_left_only"
SAVE_DIR_LEFT_ONLY_RIGHT_IGNORE = "calib_left_only_right_ignore"
SAVE_DIR_RIGHT_ONLY = "calib_right_only"
SAVE_DIR_RIGHT_ONLY_LEFT_IGNORE = "calib_right_only_left_ignore"
SAVE_DIR_SHARED_LEFT = "calib_shared_left"
SAVE_DIR_SHARED_RIGHT = "calib_shared_right"

# Fallback
SAVE_DIR_LEFT = "calib_left"
SAVE_DIR_RIGHT = "calib_right"

# Calibration Flags
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

def get_image_size(image_paths):
    """Scan images to find the most common resolution"""
    sizes = {}
    for p in image_paths:
        img = cv2.imread(p)
        if img is None: continue
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
            # print(f"  [OK]   {os.path.basename(fname)}")
        else:
            print(f"  [FAIL] {os.path.basename(fname)} (Corner detection failed)")
            
    return objpoints, imgpoints, valid_paths

def calibrate_camera_intrinsics(image_paths, cam_name, expected_size):
    print(f"\n--- Calibrating {cam_name} Intrinsics ---")
    objp, imgp, valid = detect_corners(image_paths, CHECKERBOARD, SQUARE_SIZE, expected_size)
    
    if len(valid) < 5:
        print(f"ERROR: Not enough valid images for {cam_name}!")
        return None, None, None, None
    
    print(f"Calibrating {cam_name} with {len(valid)} images (Size: {expected_size})...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, expected_size, None, None)
    
    print(f"{cam_name} RMS: {ret:.4f}")
    return K, dist, objp, imgp

def compute_stereo_reprojection_errors(objpoints, imgpointsL, imgpointsR, K1, D1, K2, D2, R, T):
    """
    Compute RMS re-projection error for each stereo pair.
    """
    total_points = 0
    total_err = 0
    errors = []
    
    n_pairs = len(objpoints)
    for i in range(n_pairs):
        p_obj = objpoints[i]
        p_imgL = imgpointsL[i]
        p_imgR = imgpointsR[i]
        
        # Project points to Left
        imgPointsL_proj, _ = cv2.projectPoints(p_obj, np.zeros(3), np.zeros(3), K1, D1)
        # Project points to Right (transform object points manually? No, projectPoints uses object points in camera frame)
        # For Right camera, points in Right frame are R*P + T. 
        # But cv2.stereoCalibrate returns R, T from Left to Right.
        # So P_right = R * P_left + T.
        # But we have object points in "Pattern Coordinate System".
        # Actually, for stereo error per frame, we can use the rvecs/tvecs from stereoCalibrate? 
        # stereoCalibrate does not return per-frame extrinsic easily for verification without re-solving.
        
        # Simpler approach: stereoCalibrate output is valid, but we need per-pair error.
        # We can use cv2.computeCorrespondEpilines?
        # Or just trust the overall RMS?
        # To filter OUTLIERS, we typically check epipolar constraint.
        
        # We will use StereoRectify and check vertical disparity?
        # OR, we can just run stereoCalibrate on a SUBSET and see if RMS drops.
        pass

    # Actually, a better way to filter huge outliers is to check the individual calibration error
    # But usually the issue is "sync", so L and R view different poses.
    # The solver tries to find an R,T that minimizes global error.
    # IF we have bad pairs, the global error is huge.
    
    # We will implement an iterative removal:
    # 1. Calibrate Stereo
    # 2. Iterate all pairs, calculate "local" error contribution?
    # Hard to get individual error from stereoCalibrate directly.
    # BUT we can compute E depending on R,T.
    # Fundamental Matrix F = inv(K2.T) * E * inv(K1)
    # E = [T]_x * R
    # Epipolar constraint: x2.T * F * x1 = 0
    # We can measure average |x2.T * F * x1| for each pair.
    
    pass

def filter_stereo_outliers(objp_list, imgpL_list, imgpR_list, pair_names, K1, D1, K2, D2, img_size, threshold=2.0):
    """
    Iteratively run stereo calibration and remove worst offenders.
    metric: Epipolar Error.
    pair_names: list of filenames corresponding to each pair.
    """
    
    current_objp = list(objp_list)
    current_imgpL = list(imgpL_list)
    current_imgpR = list(imgpR_list)
    current_names = list(pair_names)
    removed_pairs = []  # Track removed files
    
    best_rms = 10000.0
    best_params = None
    
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
            # undistort points makes it easier? No, standard F works on raw pixels if F is derived from calibrated K?
            # F = inv(K2').E.inv(K1) works with undistorted coords? 
            # Actually standard epipolar constraint: p2' * F * p1 = 0
            # p1, p2 are pixel coords.
            
            pts1 = current_imgpL[i].reshape(-1, 2)
            pts2 = current_imgpR[i].reshape(-1, 2)
            
            # Compute epipolar lines
            lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
            lines1 = lines1.reshape(-1, 3)
            
            # Distance of pt1 to line1? 
            # No, lines1 corresponds to pts2. computeCorrespondEpilines(points_in_image_2, 2, F) -> lines_in_image_1
            
            # err = |ax + by + c| / sqrt(a^2 + b^2)
            errs = []
            for j in range(len(pts1)):
                l = lines1[j]
                p = pts1[j]
                d = abs(l[0]*p[0] + l[1]*p[1] + l[2]) / np.sqrt(l[0]**2 + l[1]**2)
                errs.append(d)
                
            mean_error = np.mean(errs)
            pair_errors.append(mean_error)
            
        pair_errors = np.array(pair_errors)
        
        # Identify outliers
        # If RMS is huge, we likely have some VERY bad pairs.
        # Let's remove the worst 10% or worst single if count is low
        
        worst_idx = np.argmax(pair_errors)
        worst_err = pair_errors[worst_idx]
        
        # Threshold? If worst error is small, stop.
        if worst_err < threshold and ret < 3.0: # Allow some tolerance if RMS is acceptable
             break
             
        # Remove worst
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
            print(f"    DELETE: calib_shared_left/{name} and calib_shared_right/{name}")
        
    return ret, K1, D1, K2, D2, R, T, E, F, len(current_objp)

def main():
    print("="*60)
    print("ROBUST STEREO CALIBRATION (50mm Square Analysis)")
    print("="*60)
    
    # 1. Scan for image paths
    # Use separate folders if valid
    paths_left_only = sorted(glob.glob(f"{SAVE_DIR_LEFT_ONLY}/*.png"))
    paths_right_only = sorted(glob.glob(f"{SAVE_DIR_RIGHT_ONLY}/*.png"))
    paths_shared_left = sorted(glob.glob(f"{SAVE_DIR_SHARED_LEFT}/*.png"))
    paths_shared_right = sorted(glob.glob(f"{SAVE_DIR_SHARED_RIGHT}/*.png"))
    
    if not paths_left_only and os.path.exists(SAVE_DIR_LEFT):
         print("Using legacy folders...")
         paths_left_only = sorted(glob.glob(f"{SAVE_DIR_LEFT}/*.png"))
         paths_right_only = sorted(glob.glob(f"{SAVE_DIR_RIGHT}/*.png"))
         paths_shared_left = paths_left_only
         paths_shared_right = paths_right_only

    # 2. Determine Primary Resolution
    # We check ALL images and pick the mode
    all_imgs = paths_left_only + paths_right_only + paths_shared_left + paths_shared_right
    if not all_imgs:
        print("No images found!")
        sys.exit(1)
        
    common_size = get_image_size(all_imgs)
    print(f"Goal Resolution: {common_size}")
    
    # 3. Intrinsic Calibration
    all_left = paths_left_only + paths_shared_left
    all_right = paths_right_only + paths_shared_right
    
    K1, D1, objL, imgL = calibrate_camera_intrinsics(all_left, "LEFT", common_size)
    K2, D2, objR, imgR = calibrate_camera_intrinsics(all_right, "RIGHT", common_size)
    
    if K1 is None or K2 is None:
        print("Intrinsic calibration failed.")
        sys.exit(1)

    # 4. Prepare Stereo Pairs
    # Match shared images by filename
    print("\nMatching stereo pairs...")
    sl_dict = {os.path.basename(p): p for p in paths_shared_left}
    sr_dict = {os.path.basename(p): p for p in paths_shared_right}
    
    common_names = sorted(list(set(sl_dict.keys()) & set(sr_dict.keys())))
    
    stereo_obj = []
    stereo_imgL = []
    stereo_imgR = []
    stereo_names = []  # Track filenames for each pair
    
    valid_pair_count = 0
    
    for name in common_names:
        pL = sl_dict[name]
        pR = sr_dict[name]
        
        # Check size again
        imgL_chk = cv2.imread(pL)
        imgR_chk = cv2.imread(pR)
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
            
            # Add to list
            obj_pt = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
            obj_pt[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
            obj_pt *= SQUARE_SIZE
            
            stereo_obj.append(obj_pt)
            stereo_imgL.append(cornersL)
            stereo_imgR.append(cornersR)
            stereo_names.append(name)  # Store filename
            valid_pair_count += 1
            
    print(f"Found {valid_pair_count} candidate stereo pairs.")
    
    if valid_pair_count < 5:
        print("Not enough pairs for stereo calibration.")
        sys.exit(1)
        
    # 5. Robust Stereo Calibration
    retS, K1, D1, K2, D2, R, T, E, F, final_count = filter_stereo_outliers(
        stereo_obj, stereo_imgL, stereo_imgR, stereo_names, K1, D1, K2, D2, common_size
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Stereo Pairs Used: {final_count} (from {valid_pair_count})")
    print(f"Final Stereo RMS: {retS:.4f}")
    
    baseline = np.linalg.norm(T)
    print(f"Baseline: {baseline:.2f} mm")
    
    # Save
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, T))
    
    np.savez("stereo_params_robust.npz",
             K1=K1, dist1=D1,
             K2=K2, dist2=D2,
             R=R, T=T,
             P1=P1, P2=P2,
             E=E, F=F,
             img_size=common_size)
    print("Saved to stereo_params_robust.npz")

if __name__ == "__main__":
    main()
