import cv2
import numpy as np
import glob

# Define real-world coordinates for the chessboard corners
checkerboard_size = (7, 9)  # Change based on your checkerboard pattern
square_size = 0.025  # Set the size of each square in meters

# Prepare object points
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load images and detect chessboard corners
image_paths = ["chessboard2_60cm.jpg", "chessboard2_80cm.jpg", "chessboard2_100cm.jpg"]

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate the camera using the object points and image points
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Display the camera matrix and distortion coefficients
print("Camera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)
