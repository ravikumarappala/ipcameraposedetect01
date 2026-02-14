import cv2
import numpy as np

# Camera indices to try. Keep 0 for built-in laptop camera.
# If your external cameras use other indices, update this list.
CAMERA_INDICES = [0, 1, 2]
WINDOW_HEIGHT = 480


import time


def open_cameras(indices, warmup_time=35.0):
    """Open VideoCapture objects for given indices and return list of dicts with metadata.

    warmup_time: seconds to wait for a first good frame per camera (some USB cams need time).
    We keep the camera object even if no frame arrived within warmup_time so windows open.
    """
    caps = []
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Warning: camera index {idx} could not be opened.")
            try:
                cap.release()
            except Exception:
                pass
            continue

        # try to set resolution (some cameras ignore this)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        opened_at = time.time()
        # warm up: try to get a first good frame for up to warmup_time seconds
        warmed = False
        start = time.time()
        while time.time() - start < warmup_time:
            ret, frame = cap.read()
            if ret and frame is not None:
                warmed = True
                break
            time.sleep(0.1)

        caps.append({
            'idx': idx,
            'cap': cap,
            'opened_at': opened_at,
            'warmed': warmed,
        })
        print(f"Opened camera index {idx} (warmed={warmed})")
    return caps


def resize_to_height(frame, target_h=WINDOW_HEIGHT):
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / float(h)
    new_w = int(w * scale)
    return cv2.resize(frame, (new_w, target_h))


def main():
    caps = open_cameras(CAMERA_INDICES)
    if not caps:
        print("No cameras available. Exiting.")
        return

    # Create window names
    window_names = {idx: f"Camera {idx}" for idx, _ in caps}

    print("Press 'q' in any window (or on the terminal) to quit.")

    try:
        while True:
            any_frame_shown = False
            for cinfo in caps:
                idx = cinfo['idx']
                cap = cinfo['cap']
                warmed = cinfo.get('warmed', False)

                ret, frame = cap.read()
                if not ret or frame is None:
                    # show a black frame with a message
                    blank = np.zeros((WINDOW_HEIGHT, 640, 3), dtype=np.uint8)
                    status = 'ready' if warmed else 'warming...'
                    cv2.putText(blank, f"Camera {idx} - {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if warmed else (0, 255, 255), 2)
                    cv2.putText(blank, f"No frame yet", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(window_names[idx], blank)
                else:
                    resized = resize_to_height(frame, WINDOW_HEIGHT)
                    # overlay index and small marker
                    cv2.putText(resized, f"Camera {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow(window_names[idx], resized)
                    any_frame_shown = True

            # waitKey handles GUI events; check for 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # If none of the cameras produced frames, avoid busy loop
            if not any_frame_shown:
                cv2.waitKey(100)

    finally:
        for cinfo in caps:
            try:
                cinfo['cap'].release()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
