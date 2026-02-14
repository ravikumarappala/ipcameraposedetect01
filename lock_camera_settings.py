import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Manual mode
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_FOCUS, 0)            # Disable autofocus (if supported)

# You can print values:
print(cap.get(cv2.CAP_PROP_EXPOSURE))
print(cap.get(cv2.CAP_PROP_FOCUS))
