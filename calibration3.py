import cv2
import numpy as np

# Define the chessboard size and square size (based on your specifications)
chessboard_size = (7, 9)  # Inner corners of the chessboard
square_size = 0.025  # Each square is 2.5 cm

# Prepare object points based on real-world dimensions
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale the points by the square size in centimeters

# Arrays to store 3D points in real-world space and 2D points in the image plane
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load images and detect chessboard corners
image_paths = ["chessboard2_60cm.jpg", "chessboard2_80cm.jpg", "chessboard2_100cm.jpg"]

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Unable to load image {path}. Check if the file path is correct.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        print(f"Corners found in {path}")
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    else:
        print(f"Could not find corners in {path}. Please ensure the chessboard is clearly visible in the image.")

cv2.destroyAllWindows()

# Check if any corners were detected before calibration
if len(objpoints) > 0 and len(imgpoints) > 0:
    # Calibrate the camera using the object points and image points
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Display the camera matrix and distortion coefficients
    print("Camera matrix:\n", camera_matrix)
    print("\nDistortion coefficients:\n", dist_coeffs)
else:
    print("Error: No corners were detected in any images. Calibration cannot proceed.")
