import cv2
import os
import time
import numpy as np
import threading
import sys
import select

# --- SETTINGS ---
SAVE_DIR_LEFT = "calib_left"
SAVE_DIR_RIGHT = "calib_right"
NUM_IMAGES = 500
MAX_CAPTURE_TIME = 10 * 60  # 10 minutes
WIDTH = 2880
HEIGHT = 1620
FPS = 50


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


def main():
    os.makedirs(SAVE_DIR_LEFT, exist_ok=True)
    os.makedirs(SAVE_DIR_RIGHT, exist_ok=True)

    left_rtsp = "rtsp://admin:Test12345@172.16.1.9:554/11"
    right_rtsp = "rtsp://admin:Test12345@172.16.1.3:554/11"
    
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
    
    if test_left is None or test_right is None:
        print("ERROR: Cannot get frames")
        cam_left.stop()
        cam_right.stop()
        return
    
    print(f"Left frame: {test_left.shape}, Right frame: {test_right.shape}")

    mode = input("Enter mode (m = manual, a = auto): ").strip().lower()
    if mode not in ["m", "a"]:
        print("Invalid mode!")
        cam_left.stop()
        cam_right.stop()
        return

    interval = 2.0
    if mode == "a":
        try:
            user_input = input("Enter interval in seconds (default 2.0): ").strip()
            interval = float(user_input) if user_input else 2.0
        except:
            interval = 2.0

    count = 0
    last_capture_time = 0
    start_time = time.time()

    print("\n" + "="*50)
    print("CALIBRATION CAPTURE STARTED")
    print("="*50)
    if mode == "m":
        print("MANUAL MODE - To save:")
        print("  * Press 's' with camera window focused")
        print("  * OR press ENTER in terminal")
        print("To quit: press 'q' or Ctrl+C")
    else:
        print(f"AUTO MODE - Saving every {interval}s")
    print("="*50 + "\n")

    try:
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

            frame_left_disp = cv2.flip(frame_left, 1)
            frame_right_disp = cv2.flip(frame_right, 1)
            
            remaining = int(MAX_CAPTURE_TIME - elapsed)
            mins, secs = divmod(remaining, 60)
            status = f"Saved: {count} | Time: {int(elapsed)}s | Left: {mins}m{secs}s"
            
            fs = max(0.5, min(frame_left_disp.shape[1] / 1280, 2.0))
            th = max(1, int(fs * 2))
            
            cv2.putText(frame_left_disp, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th)
            cv2.putText(frame_right_disp, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,255,0), th)
            
            hint = "Press 's' or ENTER=save | 'q'=quit" if mode=="m" else f"AUTO every {interval}s"
            cv2.putText(frame_left_disp, hint, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fs*0.7, (0,255,255), th)
            cv2.putText(frame_right_disp, hint, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, fs*0.7, (0,255,255), th)

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
            do_quit = False
            
            if key == ord("s"):
                do_save = True
            elif key == ord("q"):
                do_quit = True
            
            if term_in is not None:
                if term_in.lower() == 'q':
                    do_quit = True
                elif mode == "m":
                    do_save = True

            if mode == "m":
                if do_save and count < NUM_IMAGES:
                    lp = f"{SAVE_DIR_LEFT}/left_{count:04d}.png"
                    rp = f"{SAVE_DIR_RIGHT}/right_{count:04d}.png"
                    cv2.imwrite(lp, frame_left_disp)
                    cv2.imwrite(rp, frame_right_disp)
                    print(f"[{count}] SAVED: {lp}, {rp}")
                    count += 1
            else:
                now = time.time()
                if now - last_capture_time >= interval and count < NUM_IMAGES:
                    lp = f"{SAVE_DIR_LEFT}/left_{count:04d}.png"
                    rp = f"{SAVE_DIR_RIGHT}/right_{count:04d}.png"
                    cv2.imwrite(lp, frame_left_disp)
                    cv2.imwrite(rp, frame_right_disp)
                    print(f"[{count}] AUTO-SAVED: {lp}, {rp}")
                    count += 1
                    last_capture_time = now

            if do_quit:
                print("Quit by user.")
                break
                
    except KeyboardInterrupt:
        print("\nCtrl+C pressed")
    finally:
        print(f"\nTotal captured: {count}")
        cam_left.stop()
        cam_right.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
