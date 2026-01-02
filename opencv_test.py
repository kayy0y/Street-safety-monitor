import cv2

print("OpenCV imported successfully")
print("Version:", cv2.__version__)

cap = cv2.VideoCapture(0)
print("VideoCapture object created:", cap.isOpened())
