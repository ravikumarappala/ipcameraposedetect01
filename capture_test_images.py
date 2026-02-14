"""
Capture Test Images for Joint Measurement
-----------------------------------------
This script connects to the two RTSP cameras and allows you to capture
a single pair of synchronized images (Left and Right).

Usage:
1. Run the script.
2. Press 's' to save the current frame from both cameras.
3. Images will be saved as:
   - test_left.png
   - test_right.png
   - test_combined.png (Side-by-side view)
"""

import cv2
import time
import threading
import numpy as np
import sys
import select

# --- SETTINGS ---
# Update these loops if your cameras are mirrored/upside down
FLIP_LEFT_HORIZONTAL = False
FLIP_LEFT_VERTICAL = False
FLIP_RIGHT_HORIZONTAL = False
FLIP_RIGHT_VERTICAL = False

LEFT_RTSP = "rtsp://admin:Test12345@172.16.1.4:554/11"
RIGHT_RTSP = "rtsp://admin:Test12345@172.16.1.3:554/11"

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
        imgL, _ = cam_left.get_frame()
        imgR, _ = cam_right.get_frame()
        
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
        
        cv2.putText(combined, "Left Camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(combined, "Right Camera", (disp_w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(combined, "Press 's' to SAVE", (combined.shape[1]//2 - 150, combined.shape[0]-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        cv2.imshow("Test Capture", combined)
        
        # Check GUI Key
        key = cv2.waitKey(10) & 0xFF
        
        # Check Terminal Input
        term_input = ""
        if select.select([sys.stdin], [], [], 0)[0]:
            term_input = sys.stdin.readline().strip().lower()
            
        if key == ord('q') or term_input == 'q':
            break
        elif key == ord('s') or term_input == 's':
            print("Saving images...")
            cv2.imwrite("test_left.png", imgL)
            cv2.imwrite("test_right.png", imgR)
            
            # Save combined full res
            full_combined = np.hstack((imgL, imgR))
            cv2.imwrite("test_combined.png", full_combined)
            
            print(f"Saved: test_left.png, test_right.png, test_combined.png")
            print("Done! Exiting...")
            break
            
    cam_left.stop()
    cam_right.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
