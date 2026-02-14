
import cv2
print("OpenCV:", cv2.__version__)
print("GStreamer enabled:", "GStreamer" in cv2.getBuildInformation())
print(cv2.getBuildInformation().split("Video I/O:")[1].split("Parallel framework:")[0])
