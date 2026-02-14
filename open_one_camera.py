import cv2
import time
import numpy as np
import argparse

WINDOW_HEIGHT = 480

def resize_to_height(frame, target_h=WINDOW_HEIGHT):
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / float(h)
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h))

def placeholder(text, w=640, h=WINDOW_HEIGHT):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img

def open_v4l2(devnode, width, height, fps):
    cap = cv2.VideoCapture(devnode, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    # Request settings (driver may choose closest supported)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # your cam supports MJPG well
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))

    # Probe a few frames
    for _ in range(20):
        r, f = cap.read()
        if r and f is not None and f.size > 0:
            return cap
        time.sleep(0.02)

    cap.release()
    return None

def open_physical(base_index, width, height, fps):
    # UVC cams often expose 2 nodes per camera; try both
    for idx in (base_index, base_index + 1):
        dev = f"/dev/video{idx}"
        cap = open_v4l2(dev, width, height, fps)
        if cap is not None:
            return cap, dev
    return None, None

def open_rtsp(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return None
    
    # Test if we can read a frame
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return cap
        time.sleep(0.1)
    
    cap.release()
    return None



def open_and_show(camA_source, camB_source, width, height, fps, warmup=2.0):
    
    # Check if source is RTSP URL or physical camera index
    if isinstance(camA_source, str) and camA_source.startswith('rtsp://'):
        capA = open_rtsp(camA_source)
        devA = camA_source
    else:
        capA, devA = open_physical(camA_source, width, height, fps)
    
    if capA is None:
        print(f"ERROR: Could not open cam A from {camA_source}")
        return

    if isinstance(camB_source, str) and camB_source.startswith('rtsp://'):
        capB = open_rtsp(camB_source)
        devB = camB_source
    else:
        capB, devB = open_physical(camB_source, width, height, fps)
    
    if capB is None:
        capA.release()
        print(f"ERROR: Could not open cam B from {camB_source}")
        return

    print(f"Opened cam A: {devA}")
    print(f"Opened cam B: {devB}")

    # warmup
    t0 = time.time()
    while time.time() - t0 < warmup:
        capA.read()
        capB.read()
        time.sleep(0.01)

    winA = f"Cam A ({devA})"
    winB = f"Cam B ({devB})"

    try:
        while True:
            rA, fA = capA.read()
            rB, fB = capB.read()

            outA = resize_to_height(fA, WINDOW_HEIGHT) if (rA and fA is not None) else placeholder(f"No frame {devA}")
            outB = resize_to_height(fB, WINDOW_HEIGHT) if (rB and fB is not None) else placeholder(f"No frame {devB}")

            cv2.putText(outA, winA, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(outB, winB, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow(winA, outA)
            cv2.imshow(winB, outB)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        capA.release()
        capB.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camA", default=0, help="Camera A source (device index or RTSP URL)")
    p.add_argument("--camB", default=2, help="Camera B source (device index or RTSP URL)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    # Use RTSP URLs by default
    camA = "rtsp://admin:Test12345@172.16.1.11:554/11"
    camB = "rtsp://admin:Test12345@172.16.1.13:554/11"
    
    # Override with command line args if provided
    if args.camA != 0:
        camA = args.camA if str(args.camA).startswith('rtsp://') else int(args.camA)
    if args.camB != 2:
        camB = args.camB if str(args.camB).startswith('rtsp://') else int(args.camB)

    open_and_show(camA, camB, args.width, args.height, args.fps)
