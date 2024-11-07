import cv2
import numpy as np

# Use a standard chessboard image from OpenCV (or any clear chessboard image)
chessboard_image_path = 'chessboard1_60cm.jpg'  # Use a known good chessboard image

# Load the test image
image = cv2.imread(chessboard_image_path)
if image is None:
    print("Error loading the test chessboard image.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Attempt to find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

    print(f"Chessboard detection in test image: {ret}")

    if ret:
        cv2.drawChessboardCorners(image, (7, 9), corners, ret)
        cv2.imshow('Detected Corners', image)
        cv2.waitKey(0)
    else:
        print("No corners detected in the test image.")

cv2.destroyAllWindows()
