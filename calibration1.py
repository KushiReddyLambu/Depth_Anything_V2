import cv2
import numpy as np

# Define the chessboard dimensions (number of inner corners per row and column)
chessboard_size = (7, 9)  # Adjust this to match your chessboard's inner corners
square_size = 0.025  # Actual square size in cm (if needed for later calculations)

# List of file paths for images at different distances
image_paths = ["chessboard1_40cm.jpg", "chessboard1_60cm.jpg", "chessboard1_80cm.jpg", "chessboard1_10+0cm.jpg"]

# Store results
results = {}

# Process each image
for path in image_paths:
    # Load the image
    image = cv2.imread(path)
    if image is None:
        print(f"Error loading image at {path}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    chessboard_contour = None

    # Find the largest contour which is likely to be the chessboard
    for contour in contours:
        # Approximate the contour and check if it has four corners
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            chessboard_contour = approx
            break

    if chessboard_contour is not None:
        # Draw the contour on the original image
        cv2.drawContours(image, [chessboard_contour], -1, (0, 255, 0), 3)

        # Get the corner points
        corners = chessboard_contour.reshape(4, 2)

        # Calculate pixel width and height based on detected corners
        widths = []
        heights = []

        for i in range(len(corners) - 1):
            # Calculate width and height between consecutive corners
            width = np.linalg.norm(corners[i] - corners[i + 1])
            widths.append(width)

        # Since chessboards are squares, the heights should be similar
        height = np.linalg.norm(
            corners[0] - corners[2])  # Assuming corners[0] and corners[2] are top-left and bottom-left
        heights = [height] * 3  # For the three heights between corners

        # Calculate average pixel width and height
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        # Save results
        results[path] = {
            "Average Pixel Width": avg_width,
            "Average Pixel Height": avg_height
        }

        # Show the processed image with detected chessboard
        cv2.imshow(f'Chessboard at {path}', image)
        cv2.waitKey(500)  # Display for 500 ms
    else:
        print(f"Chessboard not detected in {path}. Please check the image clarity.")

cv2.destroyAllWindows()

# Print results
for path, measurements in results.items():
    print(f"Results for {path}:")
    print(f"Average Pixel Width: {measurements['Average Pixel Width']:.2f} pixels")
    print(f"Average Pixel Height: {measurements['Average Pixel Height']:.2f} pixels")
