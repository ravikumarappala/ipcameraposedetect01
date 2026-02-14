import cv2
import sys
import time

WINDOW_HEIGHT = 480


def resize_to_height(frame, target_h=WINDOW_HEIGHT):
	h, w = frame.shape[:2]
	if h == target_h:
		return frame
	scale = target_h / float(h)
	new_w = int(w * scale)
	return cv2.resize(frame, (new_w, target_h))


def open_and_show(camera_index=1, warmup_time=5.0):
	"""Open a single camera and display its feed.

	camera_index: integer camera device index (default 0)
	warmup_time: seconds to wait for the first good frame (default 5s)
	"""
	cap = cv2.VideoCapture(0)
	# cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
	# cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

	cap2 = cv2.VideoCapture(1)
	if not cap.isOpened():
		print(f"Error: cannot open camera {camera_index}")
		return
	if not cap2.isOpened():
		print(f"Error: cannot open camera {camera_index+1}")
		return
	print("both cameras opened")
	# Try to set a reasonable resolution
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	# Warm up: try to get a first frame
	start = time.time()
	got_frame = False
	print("Warming up cameras...")
	while time.time() - start < warmup_time:
		ret, frame = cap.read()
		ret2, frame2 = cap2.read()
		if ret and frame is not None and ret2 and frame2 is not None:
			got_frame = True
			break
		time.sleep(0.1)
	print("both cameras warmed")
	window_name = f"Camera 0"
	window_name2 = f"Camera 1"
	print(f"Showing {window_name} (warmed={got_frame}) - press 'q' to quit")

	try:
		while True:
			
			ret, frame = cap.read()
			ret2, frame2 = cap2.read()
			if not (ret or frame is None) and not (ret2 or frame2 is None):
				# show placeholder
				placeholder = 255 * (np.zeros((WINDOW_HEIGHT, 640, 3), dtype='uint8'))
				placeholder2 = 255 * (np.zeros((WINDOW_HEIGHT, 640, 3), dtype='uint8'))
				cv2.putText(placeholder, f"No frame from camera {camera_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				cv2.putText(placeholder2, f"No frame from camera {camera_index+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
				cv2.imshow(window_name, placeholder)
				cv2.imshow(window_name2, placeholder2)
			else:
				frame = resize_to_height(frame, WINDOW_HEIGHT)
				frame2 = resize_to_height(frame2, WINDOW_HEIGHT)
				cv2.putText(frame, window_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
				cv2.putText(frame2, window_name2, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
				cv2.imshow(window_name, frame)
				cv2.imshow(window_name2, frame2)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	finally:
		try:
			cap.release()
		except Exception:
			pass
		cv2.destroyAllWindows()


if __name__ == '__main__':
	import argparse
	import numpy as np

	parser = argparse.ArgumentParser(description='Open and show a single camera feed')
	parser.add_argument('--index', '-i', type=int, default=2, help='camera device index (default 0)')
	parser.add_argument('--warmup', '-w', type=float, default=5.0, help='warmup time in seconds')
	args = parser.parse_args()

	open_and_show(args.index, args.warmup)

