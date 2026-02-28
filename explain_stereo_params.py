#!/usr/bin/env python3
"""
Simple program to explain stereo calibration parameters
Supports two modes: summary (key parameters) and detailed (full explanation)
"""
import numpy as np
import sys
import argparse
import math

def print_summary(npz_file):
    """
    Print a concise summary of key stereo parameters
    
    Args:
        npz_file: Path to the .npz file containing stereo parameters
    """
    data = np.load(npz_file)
    
    print(f"\n📷 STEREO CALIBRATION SUMMARY")
    print("=" * 80)
    
    # RMS Error
    rms_keys = ['rms_error', 'stereo_rms', 'rms']
    rms_value = None
    for key in rms_keys:
        if key in data:
            rms_value = float(data[key])
            break
    
    if rms_value is not None:
        print(f"\n✓ RMS Reprojection Error: {rms_value:.6f} pixels")
        quality = "Excellent" if rms_value < 0.5 else "Good" if rms_value < 1.0 else "Fair"
        print(f"  Calibration Quality: {quality}")
    
    # Image Size
    img_size_keys = ['imageSize', 'img_size', 'image_size']
    for key in img_size_keys:
        if key in data:
            img_size = data[key]
            print(f"\n✓ Image Resolution: {int(img_size[0])} × {int(img_size[1])} pixels")
            break
    
    # Camera Intrinsics
    cam1_keys = ['cameraMatrix1', 'K1', 'camera_matrix1']
    cam2_keys = ['cameraMatrix2', 'K2', 'camera_matrix2']
    
    K1, K2 = None, None
    for key in cam1_keys:
        if key in data:
            K1 = data[key]
            break
    for key in cam2_keys:
        if key in data:
            K2 = data[key]
            break
    
    if K1 is not None:
        print(f"\n✓ Left Camera Intrinsics:")
        print(f"  Focal Length: fx={K1[0,0]:.2f}, fy={K1[1,1]:.2f} pixels")
        print(f"  Principal Point: cx={K1[0,2]:.2f}, cy={K1[1,2]:.2f} pixels")
    
    if K2 is not None:
        print(f"\n✓ Right Camera Intrinsics:")
        print(f"  Focal Length: fx={K2[0,0]:.2f}, fy={K2[1,1]:.2f} pixels")
        print(f"  Principal Point: cx={K2[0,2]:.2f}, cy={K2[1,2]:.2f} pixels")
    
    # Distortion Coefficients
    dist1_keys = ['distCoeffs1', 'dist1', 'dist_coeffs1']
    dist2_keys = ['distCoeffs2', 'dist2', 'dist_coeffs2']
    
    for key in dist1_keys:
        if key in data:
            dist1 = data[key].flatten()
            print(f"\n✓ Left Camera Distortion: k1={dist1[0]:.4f}, k2={dist1[1]:.4f}, k3={dist1[4]:.4f}")
            break
    
    for key in dist2_keys:
        if key in data:
            dist2 = data[key].flatten()
            print(f"\n✓ Right Camera Distortion: k1={dist2[0]:.4f}, k2={dist2[1]:.4f}, k3={dist2[4]:.4f}")
            break
    
    # Rotation and Translation between cameras
    if 'R' in data:
        R = data['R']
        # Calculate rotation angles (in degrees)
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        print(f"\n✓ Rotation Between Cameras:")
        print(f"  X-axis: {math.degrees(x):+.2f}°, Y-axis: {math.degrees(y):+.2f}°, Z-axis: {math.degrees(z):+.2f}°")
    
    # Translation / Baseline
    baseline_keys = ['baseline_mm', 'baseline']
    baseline_found = False
    for key in baseline_keys:
        if key in data:
            baseline = float(data[key])
            print(f"\n✓ Baseline (Camera Distance): {baseline:.2f} mm ({baseline/10:.2f} cm)")
            baseline_found = True
            break
    
    if not baseline_found and 'T' in data:
        T = data['T'].flatten()
        baseline = np.linalg.norm(T)
        print(f"\n✓ Translation Between Cameras:")
        print(f"  X: {T[0]:.2f}, Y: {T[1]:.2f}, Z: {T[2]:.2f}")
        print(f"  Baseline (Distance): {baseline:.2f} mm ({baseline/10:.2f} cm)")
    
    print("\n" + "=" * 80)
    print("\nℹ️  Use --mode detailed for full parameter explanations")
    print("=" * 80 + "\n")


def explain_stereo_params(npz_file):
    """
    Load and explain stereo calibration parameters from an NPZ file
    
    Args:
        npz_file: Path to the .npz file containing stereo parameters
    """
    # Load the parameters
    print(f"Loading stereo parameters from: {npz_file}")
    print("=" * 80)
    
    data = np.load(npz_file)
    
    # List all available keys
    print("\nAvailable parameters in the file:")
    for key in data.files:
        print(f"  - {key}")
    print("\n" + "=" * 80)
    
    # Explain each parameter
    print("\n📷 STEREO CALIBRATION PARAMETERS EXPLAINED\n")
    
    # Camera Matrix for Left Camera
    if 'cameraMatrix1' in data:
        print("1️⃣  LEFT CAMERA INTRINSIC MATRIX (cameraMatrix1)")
        print("-" * 60)
        print(data['cameraMatrix1'])
        print("\nThis 3x3 matrix contains the left camera's intrinsic parameters:")
        fx = data['cameraMatrix1'][0, 0]
        fy = data['cameraMatrix1'][1, 1]
        cx = data['cameraMatrix1'][0, 2]
        cy = data['cameraMatrix1'][1, 2]
        print(f"  • fx (focal length in x): {fx:.2f} pixels")
        print(f"  • fy (focal length in y): {fy:.2f} pixels")
        print(f"  • cx (principal point x): {cx:.2f} pixels")
        print(f"  • cy (principal point y): {cy:.2f} pixels")
        print("\n")
    
    # Distortion Coefficients for Left Camera
    if 'distCoeffs1' in data:
        print("2️⃣  LEFT CAMERA DISTORTION COEFFICIENTS (distCoeffs1)")
        print("-" * 60)
        print(data['distCoeffs1'])
        print("\nThese coefficients correct lens distortion:")
        dist = data['distCoeffs1'].flatten()
        if len(dist) >= 5:
            print(f"  • k1 (radial): {dist[0]:.6f}")
            print(f"  • k2 (radial): {dist[1]:.6f}")
            print(f"  • p1 (tangential): {dist[2]:.6f}")
            print(f"  • p2 (tangential): {dist[3]:.6f}")
            print(f"  • k3 (radial): {dist[4]:.6f}")
        print("\n")
    
    # Camera Matrix for Right Camera
    if 'cameraMatrix2' in data:
        print("3️⃣  RIGHT CAMERA INTRINSIC MATRIX (cameraMatrix2)")
        print("-" * 60)
        print(data['cameraMatrix2'])
        print("\nThis 3x3 matrix contains the right camera's intrinsic parameters:")
        fx = data['cameraMatrix2'][0, 0]
        fy = data['cameraMatrix2'][1, 1]
        cx = data['cameraMatrix2'][0, 2]
        cy = data['cameraMatrix2'][1, 2]
        print(f"  • fx (focal length in x): {fx:.2f} pixels")
        print(f"  • fy (focal length in y): {fy:.2f} pixels")
        print(f"  • cx (principal point x): {cx:.2f} pixels")
        print(f"  • cy (principal point y): {cy:.2f} pixels")
        print("\n")
    
    # Distortion Coefficients for Right Camera
    if 'distCoeffs2' in data:
        print("4️⃣  RIGHT CAMERA DISTORTION COEFFICIENTS (distCoeffs2)")
        print("-" * 60)
        print(data['distCoeffs2'])
        print("\nThese coefficients correct lens distortion:")
        dist = data['distCoeffs2'].flatten()
        if len(dist) >= 5:
            print(f"  • k1 (radial): {dist[0]:.6f}")
            print(f"  • k2 (radial): {dist[1]:.6f}")
            print(f"  • p1 (tangential): {dist[2]:.6f}")
            print(f"  • p2 (tangential): {dist[3]:.6f}")
            print(f"  • k3 (radial): {dist[4]:.6f}")
        print("\n")
    
    # Rotation Matrix
    if 'R' in data:
        print("5️⃣  ROTATION MATRIX (R)")
        print("-" * 60)
        print(data['R'])
        print("\nThis 3x3 matrix describes the rotation from left to right camera:")
        print("  • Rotates the left camera coordinate system to align with right camera")
        
        # Convert rotation matrix to Euler angles for easier understanding
        R = data['R']
        # Calculate rotation angles (in degrees)
        import math
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        print(f"  • Rotation around X-axis: {math.degrees(x):.2f}°")
        print(f"  • Rotation around Y-axis: {math.degrees(y):.2f}°")
        print(f"  • Rotation around Z-axis: {math.degrees(z):.2f}°")
        print("\n")
    
    # Translation Vector
    if 'T' in data:
        print("6️⃣  TRANSLATION VECTOR (T)")
        print("-" * 60)
        print(data['T'])
        print("\nThis vector describes the position of right camera relative to left:")
        T = data['T'].flatten()
        print(f"  • X (horizontal): {T[0]:.4f} (positive = right camera is to the right)")
        print(f"  • Y (vertical): {T[1]:.4f} (positive = right camera is above)")
        print(f"  • Z (depth): {T[2]:.4f} (positive = right camera is forward)")
        baseline = np.linalg.norm(T)
        print(f"  • Baseline (distance between cameras): {baseline:.4f} units")
        print("\n")
    
    # Essential Matrix
    if 'E' in data:
        print("7️⃣  ESSENTIAL MATRIX (E)")
        print("-" * 60)
        print(data['E'])
        print("\nThis matrix encodes both rotation and translation between cameras:")
        print("  • Used in epipolar geometry calculations")
        print("  • Relates corresponding points in stereo images")
        print("\n")
    
    # Fundamental Matrix
    if 'F' in data:
        print("8️⃣  FUNDAMENTAL MATRIX (F)")
        print("-" * 60)
        print(data['F'])
        print("\nThis matrix relates corresponding points in pixel coordinates:")
        print("  • For a point p1 in left image and p2 in right image:")
        print("  • The epipolar constraint is: p2ᵀ · F · p1 = 0")
        print("  • Used to find corresponding points between stereo images")
        print("\n")
    
    # Rectification transforms
    if 'R1' in data:
        print("9️⃣  LEFT RECTIFICATION TRANSFORM (R1)")
        print("-" * 60)
        print(data['R1'])
        print("\nRotation applied to left image for stereo rectification:")
        print("  • Makes epipolar lines horizontal")
        print("  • Simplifies stereo matching")
        print("\n")
    
    if 'R2' in data:
        print("🔟 RIGHT RECTIFICATION TRANSFORM (R2)")
        print("-" * 60)
        print(data['R2'])
        print("\nRotation applied to right image for stereo rectification:")
        print("  • Makes epipolar lines horizontal")
        print("  • Aligns with left rectified image")
        print("\n")
    
    # Projection matrices
    if 'P1' in data:
        print("1️⃣1️⃣  LEFT PROJECTION MATRIX (P1)")
        print("-" * 60)
        print(data['P1'])
        print("\nProjects 3D points to left rectified image:")
        print("  • Used after rectification for triangulation")
        print(f"  • Shape: {data['P1'].shape}")
        print("\n")
    
    if 'P2' in data:
        print("1️⃣2️⃣  RIGHT PROJECTION MATRIX (P2)")
        print("-" * 60)
        print(data['P2'])
        print("\nProjects 3D points to right rectified image:")
        print("  • Used after rectification for triangulation")
        print(f"  • Shape: {data['P2'].shape}")
        print("\n")
    
    # Disparity-to-depth mapping matrix
    if 'Q' in data:
        print("1️⃣3️⃣  DISPARITY-TO-DEPTH MAPPING MATRIX (Q)")
        print("-" * 60)
        print(data['Q'])
        print("\nThis 4x4 matrix converts disparity to 3D coordinates:")
        print("  • Input: (x, y, disparity)")
        print("  • Output: (X, Y, Z) in 3D space")
        print("  • Formula: [X, Y, Z, W]ᵀ = Q · [x, y, disparity, 1]ᵀ")
        print("  • Final 3D point: (X/W, Y/W, Z/W)")
        print("\n")
    
    # RMS error
    if 'rms_error' in data:
        print("1️⃣4️⃣  RMS REPROJECTION ERROR (rms_error)")
        print("-" * 60)
        print(f"{data['rms_error']}")
        print("\nRoot Mean Square reprojection error:")
        print(f"  • RMS error: {data['rms_error']:.6f} pixels")
        print("  • Lower is better (< 1.0 pixel is good)")
        print("  • Indicates calibration quality")
        print("\n")
    
    # Image size
    if 'imageSize' in data:
        print("1️⃣5️⃣  IMAGE SIZE (imageSize)")
        print("-" * 60)
        print(data['imageSize'])
        print("\nSize of the calibration images:")
        print(f"  • Width: {data['imageSize'][0]} pixels")
        print(f"  • Height: {data['imageSize'][1]} pixels")
        print("\n")
    
    print("=" * 80)
    print("\n✅ SUMMARY")
    print("-" * 60)
    print("These parameters allow you to:")
    print("  1. Undistort images from both cameras")
    print("  2. Rectify stereo images (align epipolar lines)")
    print("  3. Calculate disparity maps")
    print("  4. Triangulate 3D points from 2D correspondences")
    print("  5. Convert disparity to depth/3D coordinates")
    print("\n" + "=" * 80)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Explain stereo calibration parameters from NPZ file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                    # Summary mode (default)
  %(prog)s --mode summary                     # Summary mode
  %(prog)s --mode detailed                    # Detailed explanations
  %(prog)s stereo_params.npz --mode detailed  # Specify file and mode
        '''
    )
    parser.add_argument(
        'npz_file',
        nargs='?',
        default='stereo_params_sidebyside.npz',
        help='Path to stereo parameters NPZ file (default: stereo_params_sidebyside.npz)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['summary', 'detailed'],
        default='summary',
        help='Display mode: summary (key parameters only) or detailed (full explanations)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'summary':
            print_summary(args.npz_file)
        else:
            explain_stereo_params(args.npz_file)
    except FileNotFoundError:
        print(f"❌ Error: File '{args.npz_file}' not found!")
        print(f"\nUsage: python {sys.argv[0]} [npz_file] [--mode summary|detailed]")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

