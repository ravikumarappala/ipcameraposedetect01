import cv2
import os
import time
import threading
import sys
import select
from datetime import datetime


# ======================
# CONFIGURATION
# ======================

BASE_DIR = "calibration"

# Stereo USB cameras
STEREO_LEFT = "/dev/video1"
STEREO_RIGHT = "/dev/video2"

# IP Cameras
IP_LEFT = "rtsp://admin:Test12345@172.16.1.3:554/11"
IP_RIGHT = "rtsp://admin:Test12345@172.16.1.4:554/11"

MAX_CAPTURE_TIME = 30 * 60


# ======================
# CAMERA CLASS
# ======================

class Camera:

    def __init__(self, source):

        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.running = False
        self.lock = threading.Lock()


    def start(self):

        if not self.cap.isOpened():
            return False

        self.running = True

        threading.Thread(target=self.update, daemon=True).start()

        time.sleep(1)

        return True


    def update(self):

        while self.running:

            ret, frame = self.cap.read()

            if ret:

                with self.lock:
                    self.frame = frame


    def get(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()


    def stop(self):

        self.running = False
        self.cap.release()



# ======================
# TERMINAL INPUT
# ======================

def check_terminal():

    try:

        if select.select([sys.stdin], [], [], 0)[0]:

            return sys.stdin.readline().strip()

    except:

        pass

    return None



# ======================
# IMAGE CAPTURE
# ======================

def capture(mode):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_dir = os.path.join(BASE_DIR, mode, timestamp)

    left_dir = os.path.join(session_dir, "left")
    right_dir = os.path.join(session_dir, "right")

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)


    if mode == "stereo":

        left_src = STEREO_LEFT
        right_src = STEREO_RIGHT

    else:

        left_src = IP_LEFT
        right_src = IP_RIGHT


    print("\nSaving images to:")
    print(session_dir)


    cam_left = Camera(left_src)
    cam_right = Camera(right_src)


    if not cam_left.start():

        print("Cannot open left camera")
        return


    if not cam_right.start():

        print("Cannot open right camera")
        return


    print("\nControls:")
    print("Press S or ENTER → Capture Image Pair")
    print("Press Q → Quit\n")


    count = 0

    start_time = time.time()


    while True:

        if time.time() - start_time > MAX_CAPTURE_TIME:

            print("Max capture time reached")
            break


        left = cam_left.get()
        right = cam_right.get()


        if left is None or right is None:
            continue


        cv2.imshow("LEFT", left)
        cv2.imshow("RIGHT", right)


        key = cv2.waitKey(1) & 0xFF

        term = check_terminal()


        save = False


        if key == ord('s'):
            save = True

        elif key == ord('q'):
            break


        if term is not None:

            if term.lower() == 'q':
                break

            else:
                save = True


        if save:

            cv2.imwrite(f"{left_dir}/{count:04d}.png", left)
            cv2.imwrite(f"{right_dir}/{count:04d}.png", right)

            print("Saved:", count)

            count += 1


    cam_left.stop()
    cam_right.stop()

    cv2.destroyAllWindows()

    print("\nCapture complete")



# ======================
# MAIN
# ======================

if __name__ == "__main__":

    print("\nSelect camera type")

    print("1 → Stereo Camera (/dev/video)")
    print("2 → IP Camera (RTSP)")

    choice = input("Choice: ")


    if choice == "1":

        capture("stereo")

    elif choice == "2":

        capture("ip")

    else:

        print("Invalid choice")
