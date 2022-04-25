"""Script for camera configuration

This script shows the camera feed, with a box drawn around the
tube section being observed by the network, allowing the user
to correctly place and configure the camera.
"""

# Imports
import cv2
from utils import ArducamUtils
import time

# Initialize Arducam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
arducam_utils = ArducamUtils(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, arducam_utils.convert2rgb)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Read new frame
    ret, frame = cap.read()

    # Format frame
    frame = cv2.flip(frame, 0)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame = frame.reshape(int(h), int(w))
    frame = arducam_utils.convert(frame)
    frame = frame[:, 240:-240]
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Calculations to ensure the observed tube section has the same aspect
    # ratio as that seen at training time
    half_width = 29
    half_height = int(half_width * (512 / 140))

    # Draw lines on the image to highlight the section being observed
    frame[:, 256 - half_width, 0] = 0
    frame[:, 256 - half_width, 1] = 0
    frame[:, 256 - half_width, 2] = 255
    frame[:, 256 + half_width, 0] = 0
    frame[:, 256 + half_width, 1] = 0
    frame[:, 256 + half_width, 2] = 255
    frame[256 + half_height, :, 0] = 0
    frame[256 + half_height, :, 1] = 0
    frame[256 + half_height, :, 1] = 255
    frame[256 - half_height, :, 1] = 0
    frame[256 - half_height, :, 1] = 0
    frame[256 - half_height, :, 1] = 255

    # Show image
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
