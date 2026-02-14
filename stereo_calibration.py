import cv2
import numpy as np
import glob
import os
import math

# --- Checkerboard settings ---
CHECKERBOARD = (9, 6)   # (columns, rows) of INTERNAL corners
SQUARE_SIZE = 27.0      # mm - measure your actual square size!

# --- Prepare 3D object points ---
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_left = []
imgpoints_right = []

left_images = sorted(glob.glob("calib_left/*.png"))
right_images = sorted(glob.glob("calib_right/*.png"))

print(f"Found {len(left_images)} left images, {len(right_images)} right images")

if len(left_images) != len(right_images) or len(left_images) == 0:
    print("ERROR: Image count mismatch or no images!")
    exit(1)

os.makedirs("debug_left", exist_ok=True)
os.makedirs("debug_right", exist_ok=True)

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

good_pairs = 0
img_size = None

for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    
    if imgL is None or imgR is None:
        continue

    if img_size is None:
        img_size = (imgL.shape[1], imgL.shape[0])
        print(f"Image size: {img_size}")

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, flags)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, flags)

    if retL and retR:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
        good_pairs += 1

        visL = imgL.copy()
        visR = imgR.copy()
        cv2.drawChessboardCorners(visL, CHECKERBOARD, cornersL, retL)
        cv2.drawChessboardCorners(visR, CHECKERBOARD, cornersR, retR)
        
        scale = 0.5
        cv2.imwrite(f"debug_left/debug_{idx:04d}.png", cv2.resize(visL, None, fx=scale, fy=scale))
        cv2.imwrite(f"debug_right/debug_{idx:04d}.png", cv2.resize(visR, None, fx=scale, fy=scale))
        
        print(f"[OK {good_pairs:2d}] Pair {idx}")
    else:
        print(f"[BAD] Pair {idx} - L:{'OK' if retL else 'FAIL'} R:{'OK' if retR else 'FAIL'}")

print(f"\nGood pairs: {good_pairs} / {len(left_images)}\n")

if good_pairs < 5:
    print("ERROR: Too few good pairs!")
    exit(1)

# ============================================================
# CALIBRATION WITH RELAXED CONSTRAINTS FOR WIDE-ANGLE SETUP
# ============================================================
print("="*60)
print("CALIBRATION (Optimized for 90° camera setup)")
print("="*60)

# For cameras at 90 degrees, we need to be more careful
# Use rational model for potential wide-angle lenses
calib_flags = cv2.CALIB_RATIONAL_MODEL

print("\nCalibrating LEFT camera...")
retL, K1, dist1, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_size, None, None, flags=calib_flags
)
print(f"  RMS: {retL:.4f}")
print(f"  fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f}, ratio={K1[0,0]/K1[1,1]:.4f}")

print("\nCalibrating RIGHT camera...")
retR, K2, dist2, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_size, None, None, flags=calib_flags
)
print(f"  RMS: {retR:.4f}")
print(f"  fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f}, ratio={K2[0,0]/K2[1,1]:.4f}")

# Check if calibration is reasonable
def check_calibration(K, name, rms):
    fx, fy = K[0,0], K[1,1]
    ratio = fx / fy
    ok = True
    
    if abs(ratio - 1.0) > 0.15:
        print(f"  WARNING: {name} fx/fy ratio {ratio:.3f} is unusual")
        ok = False
    if rms > 2.0:
        print(f"  WARNING: {name} RMS {rms:.3f} is high")
        ok = False
    return ok

print()
left_ok = check_calibration(K1, "LEFT", retL)
right_ok = check_calibration(K2, "RIGHT", retR)

if not right_ok:
    print("\n" + "-"*60)
    print("RIGHT camera calibration looks problematic.")
    print("Attempting calibration with simpler distortion model...")
    print("-"*60)
    
    # Try simpler model
    calib_flags_simple = 0  # Basic 5-coefficient model
    retR2, K2_new, dist2_new, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None, flags=calib_flags_simple
    )
    print(f"  Simple model RMS: {retR2:.4f}")
    print(f"  fx={K2_new[0,0]:.1f}, fy={K2_new[1,1]:.1f}, ratio={K2_new[0,0]/K2_new[1,1]:.4f}")
    
    # Use the better result
    if abs(K2_new[0,0]/K2_new[1,1] - 1.0) < abs(K2[0,0]/K2[1,1] - 1.0):
        print("  Using simple model (better ratio)")
        K2, dist2, retR = K2_new, dist2_new, retR2
    else:
        print("  Keeping rational model")

# Final validation
fx_ratio_L = K1[0,0] / K1[1,1]
fx_ratio_R = K2[0,0] / K2[1,1]

if abs(fx_ratio_R - 1.0) > 0.2:
    print("\n" + "!"*60)
    print("CALIBRATION STILL HAS ISSUES!")
    print("!"*60)
    print(f"\nRight camera fx/fy = {fx_ratio_R:.3f} (should be ~1.0)")
    print("\nThe checkerboard detection might be ambiguous.")
    print("Suggestions:")
    print("  1. Use a LARGER checkerboard that fills more of the frame")
    print("  2. Hold the board at ~45° angle so BOTH cameras see it well")
    print("  3. Move closer to the cameras when capturing")
    print("  4. Ensure the board is perfectly flat")
    print("\nProceeding anyway, but results may be inaccurate...")

# ============================================================
# STEREO CALIBRATION
# ============================================================
print("\n" + "="*60)
print("STEREO CALIBRATION")
print("="*60)

# Don't fix intrinsics since they might need adjustment
stereo_flags = 0  # Let stereo calibration refine everything

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)

retS, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K1, dist1,
    K2, dist2,
    img_size,
    criteria=criteria,
    flags=stereo_flags
)

print(f"\nStereo RMS: {retS:.4f}")
print(f"\nRefined K1: fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f}")
print(f"Refined K2: fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f}")

# ============================================================
# PROJECTION MATRICES
# ============================================================
R1 = np.eye(3)
T1 = np.zeros((3, 1))
P1 = K1 @ np.hstack((R1, T1))
P2 = K2 @ np.hstack((R, T))

baseline = np.linalg.norm(T)
angle = math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / math.pi

# ============================================================
# SAVE
# ============================================================
np.savez("stereo_params.npz",
         K1=K1, dist1=dist1,
         K2=K2, dist2=dist2,
         R=R, T=T,
         P1=P1, P2=P2,
         E=E, F=F,
         img_size=np.array(img_size))

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("CALIBRATION SUMMARY")
print("="*60)
print(f"Good pairs:     {good_pairs}")
print(f"Left RMS:       {retL:.4f}")
print(f"Right RMS:      {retR:.4f}")
print(f"Stereo RMS:     {retS:.4f}")
print(f"Baseline:       {baseline:.1f} mm ({baseline/1000:.2f} m)")
print(f"Camera angle:   {angle:.1f}°")

print("\nK1 (Left camera):")
print(f"  fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f}")
print(f"  cx={K1[0,2]:.1f}, cy={K1[1,2]:.1f}")

print("\nK2 (Right camera):")
print(f"  fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f}")
print(f"  cx={K2[0,2]:.1f}, cy={K2[1,2]:.1f}")

print("\nT (translation in mm):", T.flatten())

# Quality assessment
print("\n" + "-"*60)
if retS < 0.5 and abs(K1[0,0]/K1[1,1] - 1.0) < 0.05 and abs(K2[0,0]/K2[1,1] - 1.0) < 0.05:
    print("✓ CALIBRATION LOOKS GOOD!")
elif retS < 1.0:
    print("⚠ CALIBRATION IS OK - may work but could be improved")
else:
    print("✗ CALIBRATION QUALITY IS POOR - recommend recapturing")
print("-"*60)

print("\nSaved to stereo_params.npz")
print("Debug images in debug_left/ and debug_right/")
