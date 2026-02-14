import cv2
import os
import time
import numpy as np

# --- SETTINGS ---
SAVE_DIR_LEFT = "calib_left"
SAVE_DIR_RIGHT = "calib_right"
NUM_IMAGES = 5
#WIDTH = 1280
#HEIGHT = 720
WIDTH = 2880
HEIGHT = 1620
FPS = 50


def try_open_v4l2(devnode, width=WIDTH, height=HEIGHT, fps=FPS):
    cap = cv2.VideoCapture(devnode, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    # Prefer MJPG (your USB cam supports 1280x720@30 on MJPG)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))

    # Warm/probe frames
    for _ in range(20):
        r, f = cap.read()
        if r and f is not None and f.size > 0:
            return cap
        time.sleep(0.02)

    cap.release()
    return None


def open_physical_camera(base_index, width=WIDTH, height=HEIGHT, fps=FPS):
    """
    Each physical UVC cam often exposes two nodes (N and N+1).
    We try both and keep the one that actually returns frames.
    """
    for idx in (base_index, base_index + 1):
        dev = f"/dev/video{idx}"
        cap = try_open_v4l2(dev, width, height, fps)
        if cap is not None:
            return cap, dev
    return None, None


def open_rtsp(url, width=WIDTH, height=HEIGHT, fps=FPS):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None
    
    # Set resolution and fps for RTSP stream
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    
    # Test if we can read a frame
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return cap
        time.sleep(0.1)
    
    cap.release()
    return None


def open_camera_source(source, width=WIDTH, height=HEIGHT, fps=FPS):
    """
    Open camera from either RTSP URL or physical camera index
    """
    if isinstance(source, str) and source.startswith('rtsp://'):
        cap = open_rtsp(source, width, height, fps)
        return cap, source
    else:
        # Physical camera - use existing logic
        cap, dev = open_physical_camera(source, width, height, fps)
        return cap, dev


def apply_best_effort_controls(cap, exposure=None):
    """
    On Linux/UVC, many of these OpenCV setters are driver-dependent.
    This function tries them; if ignored, use v4l2-ctl externally.
    """
    try:
        # disable autofocus where supported
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        # try disable auto white balance
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)

        # try set gain low
        cap.set(cv2.CAP_PROP_GAIN, 0)

        # exposure: OpenCV mapping differs across drivers; best-effort only
        if exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    except Exception:
        pass


def main():
    os.makedirs(SAVE_DIR_LEFT, exist_ok=True)
    os.makedirs(SAVE_DIR_RIGHT, exist_ok=True)

    # Use RTSP feeds with lower resolution stream
    # Stream endpoint "/12" typically provides lower resolution (720p)
    # Stream endpoint "/11" typically provides higher resolution (4K)
    left_rtsp = "rtsp://admin:Test12345@172.16.1.11:554/12"
    right_rtsp = "rtsp://admin:Test12345@172.16.1.13:554/12"
    
    cam_left, dev_left = open_camera_source(left_rtsp)
    if cam_left is None:
        print(f"ERROR: Could not open left camera from {left_rtsp}")
        return

    cam_right, dev_right = open_camera_source(right_rtsp)
    if cam_right is None:
        cam_left.release()
        print(f"ERROR: Could not open right camera from {right_rtsp}")
        return

    print(f"Opened LEFT  camera: {dev_left}")
    print(f"Opened RIGHT camera: {dev_right}")

    # Best-effort controls (adjust if your driver supports it)
    # For many UVC cams, exposure control is better set via v4l2-ctl.
    apply_best_effort_controls(cam_left, exposure=-8)
    apply_best_effort_controls(cam_right, exposure=-9)

    time.sleep(1.0)

    mode = input("Enter mode (m = manual, a = auto): ").strip().lower()
    if mode not in ["m", "a"]:
        print("Invalid mode! Use m or a.")
        cam_left.release()
        cam_right.release()
        return

    interval = None
    if mode == "a":
        try:
            interval = float(input("Enter interval in seconds: "))
        except Exception:
            print("Invalid number.")
            cam_left.release()
            cam_right.release()
            return
        print(f"Auto mode: capturing every {interval} sec")

    count = 0
    last_capture_time = time.time()

    print("\n--- Calibration Capture Started ---")
    if mode == "m":
        print("Press 's' to save pair, 'q' to quit.")
    else:
        print("Auto mode running... Press 'q' to stop early.")

    while True:
        ret1, frame_left = cam_left.read()
        ret2, frame_right = cam_right.read()

        if not ret1 or frame_left is None:
            print(f"Camera read error (LEFT: {dev_left})!")
            break
        if not ret2 or frame_right is None:
            print(f"Camera read error (RIGHT: {dev_right})!")
            break

        # Mirror if you want (keep your original behavior)
        frame_left = cv2.flip(frame_left, 1)
        frame_right = cv2.flip(frame_right, 1)

        cv2.imshow(f"Left Camera ({dev_left})", frame_left)
        cv2.imshow(f"Right Camera ({dev_right})", frame_right)

        key = cv2.waitKey(1) & 0xFF

        if mode == "m":
            if key == ord("s"):
                if count < NUM_IMAGES:
                    left_path = f"{SAVE_DIR_LEFT}/left_{count}.png"
                    right_path = f"{SAVE_DIR_RIGHT}/right_{count}.png"
                    cv2.imwrite(left_path, frame_left)
                    cv2.imwrite(right_path, frame_right)

                    print(f"[{count}] Saved pair → {left_path}, {right_path}")
                    count += 1
                    time.sleep(0.4)  # debounce
                else:
                    print("Done saving all images.")
                    break
        else:
            now = time.time()
            if now - last_capture_time >= interval:
                if count < NUM_IMAGES:
                    left_path = f"{SAVE_DIR_LEFT}/left_{count}.png"
                    right_path = f"{SAVE_DIR_RIGHT}/right_{count}.png"
                    cv2.imwrite(left_path, frame_left)
                    cv2.imwrite(right_path, frame_right)

                    print(f"[{count}] Auto-saved pair → {left_path}, {right_path}")
                    count += 1
                    last_capture_time = now
                else:
                    print("Finished auto capture.")
                    break

        if key == ord("q"):
            print("Stopped by user.")
            break

    cam_left.release()
    cam_right.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
abs