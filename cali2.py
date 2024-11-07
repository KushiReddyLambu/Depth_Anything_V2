import cv2
import numpy as np

# Load the image
image_path = "chessboard1_40cm.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges in the image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use the Hough Line Transform to detect straight lines in the image
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Draw detected lines on the image
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        # Draw the line on the image
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the image with detected lines
cv2.imshow("Detected Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
