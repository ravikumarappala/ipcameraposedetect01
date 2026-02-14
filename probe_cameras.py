import cv2
import time

indices_to_test = range(8)
print('Testing camera indices 0-7')
for i in indices_to_test:
    cap = cv2.VideoCapture(i)
    opened = cap.isOpened()
    read_ok = False
    if opened:
        # give camera a moment
        time.sleep(0.5)
        ret, frame = cap.read()
        read_ok = bool(ret and frame is not None)
    print(f"Index {i}: opened={opened}, read_frame={read_ok}")
    try:
        cap.release()
    except Exception:
        pass
print('Probe complete')
