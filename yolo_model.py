import cv2
from ultralytics import YOLO

# Load the pretrained YOLOv8 model (replace 'yolov8n.pt' with the path to your specific model)
model = YOLO('yolov8n.pt')

# Open a connection to the webcam (0 is usually the default camera index)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the real-time inference loop
try:
    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Run inference on the frame
        results = model(frame)

        # Annotate the frame with detections
        annotated_frame = frame.copy()  # Copy the frame to draw on it

        for detection in results[0].boxes:
            # Check if the detected object is a bottle (usually class ID 39 in COCO)
            if detection.cls == 39:  # Replace 39 with the correct class ID if needed
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Convert to integers

                # Calculate the centroid
                cX = (x1 + x2) // 2
                cY = (y1 + y2) // 2

                # Draw the bounding box and centroid on the frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (cX, cY), 5, (255, 0, 0), -1)  # Draw centroid
                cv2.putText(annotated_frame, f"Bottle Centroid: ({cX}, {cY})", (cX - 40, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the annotated frame
        cv2.imshow('YOLOv8 Real-Time Bottle Detection with Centroid', annotated_frame)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()
