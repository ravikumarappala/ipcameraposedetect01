"""
Calibration Image Capture for 90-degree Camera Setup

This script captures calibration images in 3 phases:
1. LEFT camera only - face the checkerboard towards LEFT camera
2. RIGHT camera only - face the checkerboard towards RIGHT camera  
3. SHARED - hold checkerboard at 45° so BOTH cameras see it

This ensures good intrinsic calibration for each camera separately,
then captures shared views for stereo calibration.
"""

import cv2
import os
import time
import numpy as np
import threading
import sys
import select
import shutil

# --- SETTINGS ---
SAVE_DIR_LEFT = "calib_left"
SAVE_DIR_RIGHT = "calib_right"
SAVE_DIR_LEFT_ONLY = "calib_left_only"
SAVE_DIR_RIGHT_ONLY = "calib_right_only"
SAVE_DIR_SHARED_LEFT = "calib_shared_left"
SAVE_DIR_SHARED_RIGHT = "calib_shared_right"

MAX_CAPTURE_TIME = 10 * 60  # 10 minutes per phase

# --- FLIP SETTINGS (Set to True if camera is mirrored/upside down) ---
FLIP_LEFT_HORIZONTAL = False
FLIP_LEFT_VERTICAL = False
FLIP_RIGHT_HORIZONTAL = False
FLIP_RIGHT_VERTICAL = False



class RTSPCamera:
    def __init__(self, url, name="camera"):
        self.url = url
        self.name = name
        self.cap = None
        self.frame = None
        self.timestamp = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
    def open(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(30):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.frame = frame.copy()
                    self.timestamp = time.time()
                return True
            time.sleep(0.1)
        self.cap.release()
        self.cap = None
        return False
    
    def start_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self._grab_frames, daemon=True)
        self.thread.start()
        
    def _grab_frames(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.time()
            else:
                time.sleep(0.01)
                
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.timestamp
            return None, None
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None


def open_rtsp_camera(url, name="camera"):
    cam = RTSPCamera(url, name)
    if cam.open():
        cam.start_thread()
        time.sleep(0.5)
        return cam
    return None


def check_terminal_input():
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip()
    except:
        pass
    return None


def capture_phase(cam_left, cam_right, phase, save_dir_left, save_dir_right, 
                  show_left=True, show_right=True, min_images=15):
    """
    Capture images for a specific phase
    
    phase: "LEFT_ONLY", "RIGHT_ONLY", or "SHARED"
    """
    os.makedirs(save_dir_left, exist_ok=True)
    os.makedirs(save_dir_right, exist_ok=True)
    
    count = 0
    start_time = time.time()
    
    print("\n" + "="*60)
    print(f"PHASE: {phase}")
    print("="*60)
    
    if phase == "LEFT_ONLY":
        print("Face the checkerboard DIRECTLY towards the LEFT camera")
        print("The RIGHT camera view doesn't matter for this phase")
        print(f"Capture at least {min_images} images at different angles/distances")
    elif phase == "RIGHT_ONLY":
        print("Face the checkerboard DIRECTLY towards the RIGHT camera")
        print("The LEFT camera view doesn't matter for this phase")
        print(f"Capture at least {min_images} images at different angles/distances")
    else:  # SHARED
        print("Hold the checkerboard at ~45° angle")
        print("so BOTH cameras can see the full checkerboard!")
        print(f"Capture at least {min_images} image pairs")
    
    print("\nControls:")
    print("  Press 's' or ENTER = save image(s)")
    print("  Press 'n' = done with this phase (next)")
    print("  Press 'q' = quit entirely")
    print("="*60 + "\n")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= MAX_CAPTURE_TIME:
            print(f"\nMax time ({MAX_CAPTURE_TIME//60} min) reached for this phase!")
            break
        
        frame_left, _ = cam_left.get_frame()
        frame_right, _ = cam_right.get_frame()
        
        if frame_left is None or frame_right is None:
            time.sleep(0.1)
            continue
        
        if FLIP_LEFT_HORIZONTAL:
            frame_left = cv2.flip(frame_left, 1)
        if FLIP_LEFT_VERTICAL:
            frame_left = cv2.flip(frame_left, 0)
            
        if FLIP_RIGHT_HORIZONTAL:
            frame_right = cv2.flip(frame_right, 1)
        if FLIP_RIGHT_VERTICAL:
            frame_right = cv2.flip(frame_right, 0)

        # Don't flip - keep original orientation for calibration
        frame_left_disp = frame_left.copy()
        frame_right_disp = frame_right.copy()
        
        remaining = int(MAX_CAPTURE_TIME - elapsed)
        mins, secs = divmod(remaining, 60)
        
        # Status text
        status = f"{phase} | Saved: {count} | Time: {mins}m{secs}s"
        progress = f"Need {max(0, min_images - count)} more" if count < min_images else "Ready for next phase!"
        
        fs = max(0.5, min(frame_left_disp.shape[1] / 1280, 2.0))
        th = max(1, int(fs * 2))
        
        # Draw on left frame
        cv2.putText(frame_left_disp, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th)
        cv2.putText(frame_left_disp, progress, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fs*0.8, (0,255,255), th)
        cv2.putText(frame_left_disp, "s/ENTER=save | n=next | q=quit", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, fs*0.6, (255,255,255), th)
        
        # Draw on right frame
        cv2.putText(frame_right_disp, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th)
        cv2.putText(frame_right_disp, progress, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fs*0.8, (0,255,255), th)
        
        # Highlight which camera matters for this phase
        if phase == "LEFT_ONLY":
            cv2.putText(frame_left_disp, "<<< CAPTURE THIS >>>", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,255), th+1)
        elif phase == "RIGHT_ONLY":
            cv2.putText(frame_right_disp, "<<< CAPTURE THIS >>>", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,255), th+1)
        else:
            cv2.putText(frame_left_disp, "BOTH CAMERAS!", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,255), th+1)
            cv2.putText(frame_right_disp, "BOTH CAMERAS!", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,255), th+1)
        
        # Resize for display
        max_w = 960
        if frame_left_disp.shape[1] > max_w:
            sc = max_w / frame_left_disp.shape[1]
            sz = (int(frame_left_disp.shape[1]*sc), int(frame_left_disp.shape[0]*sc))
            show_l = cv2.resize(frame_left_disp, sz)
            show_r = cv2.resize(frame_right_disp, sz)
        else:
            show_l, show_r = frame_left_disp, frame_right_disp
        
        cv2.imshow("Left Camera", show_l)
        cv2.imshow("Right Camera", show_r)
        
        key = cv2.waitKey(1) & 0xFF
        term_in = check_terminal_input()
        
        do_save = False
        do_next = False
        do_quit = False
        
        if key == ord("s"):
            do_save = True
        elif key == ord("n"):
            do_next = True
        elif key == ord("q"):
            do_quit = True
        
        if term_in is not None:
            if term_in.lower() == 'q':
                do_quit = True
            elif term_in.lower() == 'n':
                do_next = True
            else:
                do_save = True
        
        if do_save:
            # Save original frames (not display versions with text)
            lp = f"{save_dir_left}/img_{count:04d}.png"
            rp = f"{save_dir_right}/img_{count:04d}.png"
            cv2.imwrite(lp, frame_left)
            cv2.imwrite(rp, frame_right)
            print(f"[{count}] SAVED: {lp}, {rp}")
            count += 1
        
        if do_next:
            if count < min_images:
                print(f"\nWarning: Only {count} images captured (recommended: {min_images})")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            print(f"\nPhase {phase} complete with {count} images")
            break
        
        if do_quit:
            return count, "QUIT"
    
    return count, "NEXT"


def main():
    # Create all directories
    for d in [SAVE_DIR_LEFT, SAVE_DIR_RIGHT, SAVE_DIR_LEFT_ONLY, SAVE_DIR_RIGHT_ONLY,
              SAVE_DIR_SHARED_LEFT, SAVE_DIR_SHARED_RIGHT]:
        os.makedirs(d, exist_ok=True)
    
    left_rtsp = "rtsp://admin:Test12345@172.16.1.4:554/11"
    right_rtsp = "rtsp://admin:Test12345@172.16.1.3:554/11"
    
    print("="*60)
    print("90-DEGREE STEREO CALIBRATION IMAGE CAPTURE")
    print("="*60)
    print("\nThis will capture images in 3 phases:")
    print("  1. LEFT camera only (for left intrinsics)")
    print("  2. RIGHT camera only (for right intrinsics)")
    print("  3. SHARED views (for stereo extrinsics)")
    print()
    
    print(f"Opening LEFT camera: {left_rtsp}")
    cam_left = open_rtsp_camera(left_rtsp, "left")
    if cam_left is None:
        print("ERROR: Could not open left camera")
        return
    
    print(f"Opening RIGHT camera: {right_rtsp}")
    cam_right = open_rtsp_camera(right_rtsp, "right")
    if cam_right is None:
        cam_left.stop()
        print("ERROR: Could not open right camera")
        return
    
    print("Waiting for streams to stabilize...")
    time.sleep(2.0)
    
    test_left, _ = cam_left.get_frame()
    test_right, _ = cam_right.get_frame()
    
    # Apply initial flips for test
    if test_left is not None:
        if FLIP_LEFT_HORIZONTAL: test_left = cv2.flip(test_left, 1)
        if FLIP_LEFT_VERTICAL: test_left = cv2.flip(test_left, 0)
    if test_right is not None:
        if FLIP_RIGHT_HORIZONTAL: test_right = cv2.flip(test_right, 1)
        if FLIP_RIGHT_VERTICAL: test_right = cv2.flip(test_right, 0)
    
    if test_left is None or test_right is None:
        print("ERROR: Cannot get frames")
        cam_left.stop()
        cam_right.stop()
        return
    
    print(f"Left frame: {test_left.shape}, Right frame: {test_right.shape}")
    
    input("\nPress ENTER to start PHASE 1 (LEFT camera only)...")
    
    # PHASE 1: Left camera only
    count_left, status = capture_phase(
        cam_left, cam_right, 
        "LEFT_ONLY",
        SAVE_DIR_LEFT_ONLY, SAVE_DIR_LEFT_ONLY + "_right_ignore",
        min_images=20
    )
    
    if status == "QUIT":
        print("\nQuitting...")
        cam_left.stop()
        cam_right.stop()
        cv2.destroyAllWindows()
        return
    
    input("\nPress ENTER to start PHASE 2 (RIGHT camera only)...")
    
    # PHASE 2: Right camera only
    count_right, status = capture_phase(
        cam_left, cam_right,
        "RIGHT_ONLY", 
        SAVE_DIR_RIGHT_ONLY + "_left_ignore", SAVE_DIR_RIGHT_ONLY,
        min_images=20
    )
    
    if status == "QUIT":
        print("\nQuitting...")
        cam_left.stop()
        cam_right.stop()
        cv2.destroyAllWindows()
        return
    
    input("\nPress ENTER to start PHASE 3 (SHARED - checkerboard at 45°)...")
    
    # PHASE 3: Shared images
    count_shared, status = capture_phase(
        cam_left, cam_right,
        "SHARED",
        SAVE_DIR_SHARED_LEFT, SAVE_DIR_SHARED_RIGHT,
        min_images=25
    )
    
    # Clean up
    cam_left.stop()
    cam_right.stop()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("CAPTURE COMPLETE!")
    print("="*60)
    print(f"LEFT only images:   {count_left} (in {SAVE_DIR_LEFT_ONLY}/)")
    print(f"RIGHT only images:  {count_right} (in {SAVE_DIR_RIGHT_ONLY}/)")
    print(f"SHARED image pairs: {count_shared} (in {SAVE_DIR_SHARED_LEFT}/ and {SAVE_DIR_SHARED_RIGHT}/)")
    
    # Copy to standard calibration folders for the calibration script
    print("\nPreparing files for calibration...")
    
    # Clear old calib folders
    for d in [SAVE_DIR_LEFT, SAVE_DIR_RIGHT]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    
    # Copy left-only images to calib_left
    idx = 0
    for f in sorted(os.listdir(SAVE_DIR_LEFT_ONLY)):
        if f.endswith('.png'):
            src = os.path.join(SAVE_DIR_LEFT_ONLY, f)
            dst = os.path.join(SAVE_DIR_LEFT, f"left_{idx:04d}.png")
            shutil.copy(src, dst)
            idx += 1
    
    # Copy shared left images to calib_left
    for f in sorted(os.listdir(SAVE_DIR_SHARED_LEFT)):
        if f.endswith('.png'):
            src = os.path.join(SAVE_DIR_SHARED_LEFT, f)
            dst = os.path.join(SAVE_DIR_LEFT, f"left_{idx:04d}.png")
            shutil.copy(src, dst)
            idx += 1
    
    # Copy right-only images to calib_right
    idx = 0
    for f in sorted(os.listdir(SAVE_DIR_RIGHT_ONLY)):
        if f.endswith('.png'):
            src = os.path.join(SAVE_DIR_RIGHT_ONLY, f)
            dst = os.path.join(SAVE_DIR_RIGHT, f"right_{idx:04d}.png")
            idx += 1
    
    # Copy shared right images to calib_right  
    for f in sorted(os.listdir(SAVE_DIR_SHARED_RIGHT)):
        if f.endswith('.png'):
            src = os.path.join(SAVE_DIR_SHARED_RIGHT, f)
            dst = os.path.join(SAVE_DIR_RIGHT, f"right_{idx:04d}.png")
            shutil.copy(src, dst)
            idx += 1
    
    print(f"\nFiles prepared in {SAVE_DIR_LEFT}/ and {SAVE_DIR_RIGHT}/")
    print("\nNow run: python stereo_calibration_90deg.py")
    print("="*60)


if __name__ == "__main__":
    main()
