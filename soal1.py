import cv2
import torch
from ultralytics import YOLO
import time

# Load YOLOv8 model (ensure you have the pre-trained weights)
model = YOLO('yolov8n.pt')  # YOLOv8n (nano) for high performance and real-time capability

# Define real-time object detection function
def real_time_detection(save_output=False, output_path="output.avi"):
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses.")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS for smoother video

    # Initialize video writer if saving output
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))

    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera.")
            break

        # Start timing for performance measurement
        start_time = time.time()

        # Perform object detection
        results = model(frame, stream=True)

        # Annotate frame with detection results
        for result in results:
            annotated_frame = result.plot()  # Plot detection boxes on the frame

        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("Real-Time Object Detection", annotated_frame)

        # Save the output frame if enabled
        if save_output:
            out.write(annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

# Run real-time detection with video saving enabled
real_time_detection(save_output=True, output_path="detected_output.avi")
