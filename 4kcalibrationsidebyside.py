#copy this code
#identify the test image resolution by putting google/ai
#we need teost vania and identify bad images
#move bad images to archive/bad_images/right and archive/bad_images/left folders respectively
# from importlib.metadata import files
# from cv2.detail import resultRoiIntersection
# from _socket import EAI_AGAIN
# run code EAI_AGAIN
# fix this calibration to get better resultRoiIntersection
# should includ rms saving in .npz files
# so when we run print params rms also shoudl print
#add option to take images or libe
# from turtle import right
# from cv2.dnn import imagesFromBlob
# test_ss_left test_ss_right imagesFromBlob
# inputs calib_ss_elf, right
# test calibation with realtime_image_arm_3d.py and ensure the .npz file  sterio_params_sidebyside.npz



"""
Calibration Image Capture for Side-by-Side Camera Setup

This script captures calibration images for a horizontal camera setup where
both cameras are mounted on the same wall, on the same horizontal line,
approximately 4 feet (1.2m) apart facing the same direction.

Capture Strategy:
1. SHARED views only - Both cameras can always see the checkerboard
   since they face the same direction. Just capture at various angles/distances.

Images are saved with suffix: calib_ss__**
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
# ss = side-by-side
SAVE_DIR_LEFT = "calib_ss_left_4k"
SAVE_DIR_RIGHT = "calib_ss_right_4k"

MAX_CAPTURE_TIME = 15 * 60  # 15 minutes

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


def capture_sidebyside(cam_left, cam_right, save_dir_left, save_dir_right, min_images=30):
    """
    Capture images for side-by-side camera setup.
    
    Both cameras face the same direction, so we just need to capture
    the checkerboard at various angles and distances.
    """
    os.makedirs(save_dir_left, exist_ok=True)
    os.makedirs(save_dir_right, exist_ok=True)
    
    count = 0
    start_time = time.time()
    
    print("\n" + "="*60)
    print("SIDE-BY-SIDE STEREO CALIBRATION CAPTURE")
    print("="*60)
    print("\nCamera Setup: Both cameras on wall, 4 feet apart, facing same direction")
    print("\nInstructions:")
    print("  - Hold the checkerboard facing TOWARDS the cameras")
    print("  - Move it around the capture volume")
    print("  - Vary the distance (near to far)")
    print("  - Tilt/rotate the board at various angles")
    print("  - Cover left side, center, and right side of the view")
    print(f"\n  Target: at least {min_images} image pairs")
    print("\nControls:")
    print("  Press 's' or ENTER = save image pair")
    print("  Press 'q' = quit")
    print("="*60 + "\n")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= MAX_CAPTURE_TIME:
            print(f"\nMax time ({MAX_CAPTURE_TIME//60} min) reached!")
            break
        
        frame_left, _ = cam_left.get_frame()
        frame_right, _ = cam_right.get_frame()
        
        if frame_left is None or frame_right is None:
            time.sleep(0.1)
            continue
        
        # Apply flip corrections if needed
        if FLIP_LEFT_HORIZONTAL:
            frame_left = cv2.flip(frame_left, 1)
        if FLIP_LEFT_VERTICAL:
            frame_left = cv2.flip(frame_left, 0)
            
        if FLIP_RIGHT_HORIZONTAL:
            frame_right = cv2.flip(frame_right, 1)
        if FLIP_RIGHT_VERTICAL:
            frame_right = cv2.flip(frame_right, 0)

        # Display copies with overlays
        frame_left_disp = frame_left.copy()
        frame_right_disp = frame_right.copy()
        
        remaining = int(MAX_CAPTURE_TIME - elapsed)
        mins, secs = divmod(remaining, 60)
        
        # Status text
        status = f"Side-by-Side | Saved: {count} | Time: {mins}m{secs}s"
        if count < min_images:
            progress = f"Need {min_images - count} more"
        else:
            progress = "Target reached! Keep going or press 'q' to finish"
        
        fs = max(0.5, min(frame_left_disp.shape[1] / 1280, 2.0))
        th = max(1, int(fs * 2))
        
        # Draw on left frame
        cv2.putText(frame_left_disp, "LEFT CAMERA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fs*0.7, (0,255,0), th)
        cv2.putText(frame_left_disp, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, fs*0.6, (0,255,0), th)
        cv2.putText(frame_left_disp, progress, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, fs*0.6, (0,255,255), th)
        cv2.putText(frame_left_disp, "s/ENTER=save | q=quit", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, fs*0.5, (255,255,255), th)
        
        # Draw on right frame
        cv2.putText(frame_right_disp, "RIGHT CAMERA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fs*0.7, (0,255,0), th)
        cv2.putText(frame_right_disp, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, fs*0.6, (0,255,0), th)
        cv2.putText(frame_right_disp, progress, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, fs*0.6, (0,255,255), th)
        
        # Resize for display
        max_w = 960
        if frame_left_disp.shape[1] > max_w:
            sc = max_w / frame_left_disp.shape[1]
            sz = (int(frame_left_disp.shape[1]*sc), int(frame_left_disp.shape[0]*sc))
            show_l = cv2.resize(frame_left_disp, sz)
            show_r = cv2.resize(frame_right_disp, sz)
        else:
            show_l, show_r = frame_left_disp, frame_right_disp
        
        # Show side by side in one window or separate
        cv2.imshow("Left Camera (calib_ss)", show_l)
        cv2.imshow("Right Camera (calib_ss)", show_r)
        
        key = cv2.waitKey(1) & 0xFF
        term_in = check_terminal_input()
        
        do_save = False
        do_quit = False
        
        if key == ord("s"):
            do_save = True
        elif key == ord("q"):
            do_quit = True
        
        if term_in is not None:
            if term_in.lower() == 'q':
                do_quit = True
            else:
                do_save = True
        
        if do_save:
            # Save original frames (not display versions with text)
            lp = f"{save_dir_left}/calib_ss_{count:04d}.png"
            rp = f"{save_dir_right}/calib_ss_{count:04d}.png"
            cv2.imwrite(lp, frame_left)
            cv2.imwrite(rp, frame_right)
            print(f"[{count}] SAVED: {lp}, {rp}")
            count += 1
        
        if do_quit:
            break
    
    return count


def main():
    # Create directories
    os.makedirs(SAVE_DIR_LEFT, exist_ok=True)
    os.makedirs(SAVE_DIR_RIGHT, exist_ok=True)
    
    # RTSP URLs for your cameras
    left_rtsp = "rtsp://admin:Test12345@172.16.1.3:554/11"
    right_rtsp = "rtsp://admin:Test12345@172.16.1.4:554/11"
    
    print("="*60)
    print("SIDE-BY-SIDE STEREO CALIBRATION IMAGE CAPTURE")
    print("="*60)
    print("\nCamera Setup:")
    print("  - Both cameras mounted on the SAME WALL")
    print("  - Approximately 4 feet (1.2m) apart horizontally")
    print("  - Both cameras facing the SAME direction")
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
    
    input("\nPress ENTER to start capturing calibration images...")
    
    # Capture images
    count = capture_sidebyside(
        cam_left, cam_right, 
        SAVE_DIR_LEFT, SAVE_DIR_RIGHT,
        min_images=30
    )
    
    # Clean up
    cam_left.stop()
    cam_right.stop()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("CAPTURE COMPLETE!")
    print("="*60)
    print(f"Total image pairs captured: {count}")
    print(f"  LEFT:  {SAVE_DIR_LEFT}/")
    print(f"  RIGHT: {SAVE_DIR_RIGHT}/")
    print("\nNext step: Run stereo_calibration_sidebyside.py")
    print("="*60)


if __name__ == "__main__":
    main()
