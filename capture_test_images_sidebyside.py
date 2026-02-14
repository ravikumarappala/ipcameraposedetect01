"""
Capture Test Images for Side-by-Side Camera Setup
-------------------------------------------------
This script connects to the two RTSP cameras in a side-by-side (horizontal)
configuration and allows you to capture a single pair of synchronized images.

Camera Setup:
- Both cameras on same wall, ~4 feet apart
- Both cameras facing the same direction

Usage:
1. Run the script.
2. Press 's' to save the current frame from both cameras.
3. Images will be saved as:
   - test_ss_left.png
   - test_ss_right.png
   - test_ss_combined.png (Side-by-side view)
"""

import cv2
import time
import threading
import numpy as np
import sys
import select

# --- SETTINGS ---
# Update these flags if your cameras are mirrored/upside down
FLIP_LEFT_HORIZONTAL = False
FLIP_LEFT_VERTICAL = False
FLIP_RIGHT_HORIZONTAL = False
FLIP_RIGHT_VERTICAL = False

LEFT_RTSP = "rtsp://admin:Test12345@172.16.1.3:554/11"
RIGHT_RTSP = "rtsp://admin:Test12345@172.16.1.4:554/11"

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
        # Warmup
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
    
    def start(self):
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

def apply_flips(img, h_flip, v_flip):
    if img is None: return None
    if h_flip:
        img = cv2.flip(img, 1)
    if v_flip:
        img = cv2.flip(img, 0)
    return img

def main():
    print("="*60)
    print("SIDE-BY-SIDE CAMERA TEST CAPTURE")
    print("="*60)
    print("\nCamera Setup: Both cameras on wall, ~4 feet apart")
    print()
    
    print("Connecting to cameras...")
    cam_left = RTSPCamera(LEFT_RTSP, "Left")
    cam_right = RTSPCamera(RIGHT_RTSP, "Right")
    
    if not cam_left.open() or not cam_right.open():
        print("Error connecting to cameras!")
        return

    cam_left.start()
    cam_right.start()
    
    print("Waiting for streams to stabilize...")
    time.sleep(2.0)
    
    print("\n" + "="*50)
    print("  PRESS 's' to CAPTURE and SAVE")
    print("  PRESS 'q' to QUIT")
    print("="*50 + "\n")
    
    while True:
        imgL, ts_l = cam_left.get_frame()
        imgR, ts_r = cam_right.get_frame()
        
        if imgL is None or imgR is None:
            time.sleep(0.01)
            continue
            
        # Apply Flips
        imgL = apply_flips(imgL, FLIP_LEFT_HORIZONTAL, FLIP_LEFT_VERTICAL)
        imgR = apply_flips(imgR, FLIP_RIGHT_HORIZONTAL, FLIP_RIGHT_VERTICAL)
        
        # Display resize
        disp_h = 540
        scale = disp_h / imgL.shape[0]
        disp_w = int(imgL.shape[1] * scale)
        
        viewL = cv2.resize(imgL, (disp_w, disp_h))
        viewR = cv2.resize(imgR, (disp_w, disp_h))
        
        # Combine side-by-side for display
        combined = np.hstack((viewL, viewR))
        
        # Add labels
        cv2.putText(combined, "LEFT Camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(combined, "RIGHT Camera", (disp_w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Show sync info
        if ts_l and ts_r:
            sync_diff_ms = abs(ts_l - ts_r) * 1000
            sync_text = f"Sync: {sync_diff_ms:.1f} ms"
            color = (0, 255, 0) if sync_diff_ms < 100 else (0, 165, 255)
            cv2.putText(combined, sync_text, (combined.shape[1]//2 - 100, combined.shape[0]-70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(combined, "Press 's' to SAVE | 'q' to QUIT", (combined.shape[1]//2 - 200, combined.shape[0]-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        cv2.imshow("Side-by-Side Test Capture", combined)
        
        # Check GUI Key
        key = cv2.waitKey(10) & 0xFF
        
        # Check Terminal Input
        term_input = ""
        if select.select([sys.stdin], [], [], 0)[0]:
            term_input = sys.stdin.readline().strip().lower()
            
        if key == ord('q') or term_input == 'q':
            print("\nQuitting...")
            break
        elif key == ord('s') or term_input == 's':
            print("\nSaving images...")
            cv2.imwrite("test_ss_left11.png", imgL)
            cv2.imwrite("test_ss_right11.png", imgR)
            
            # Save combined full res
            full_combined = np.hstack((imgL, imgR))
            cv2.imwrite("test_ss_combined.png", full_combined)
            
            print(f"✓ Saved: test_ss_left.png")
            print(f"✓ Saved: test_ss_right.png")
            print(f"✓ Saved: test_ss_combined.png")
            print(f"  Image size: {imgL.shape[1]}x{imgL.shape[0]}")
            if ts_l and ts_r:
                print(f"  Sync difference: {abs(ts_l - ts_r)*1000:.1f} ms")
            print("\nDone! Exiting...")
            break
            
    cam_left.stop()
    cam_right.stop()
    cv2.destroyAllWindows()
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
