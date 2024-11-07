import cv2

# Open a connection to the webcam (0 is usually the default camera index)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


def detect_shape(contour):
    # Approximate the contour shape
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Determine the shape based on the number of vertices
    num_vertices = len(approx)
    shape = "Unknown"

    if num_vertices == 3:
        shape = "Triangle"
    elif num_vertices == 4:
        # Calculate aspect ratio to differentiate between square and rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif num_vertices > 4:
        shape = "Circle"

    return shape


try:
    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to grayscale and apply thresholding
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Ignore small contours to filter out noise
            if cv2.contourArea(contour) < 500:
                continue

            # Get the shape name from the detect_shape function
            shape_name = detect_shape(contour)

            # Draw the contour on the frame
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Calculate the centroid using moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the centroid on the frame
                cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
                # Put the shape name near the centroid
                cv2.putText(frame, shape_name, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with contours, shape names, and centroids
        cv2.imshow("Real-Time Shape Detection with Centroid", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()
