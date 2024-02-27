from datetime import datetime
import cv2
import os
import time

# Open a connection to the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Create a folder to save frames
output_folder = 'Hallway_PlaceOnShelf_NoLights_AtBend'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set the webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize OpenCV window
cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)

# Capture frames at 30 frames per second for 10 seconds
frame_rate = 30
capture_duration = 60

start_time = time.time()

try:
    for i in range(frame_rate * capture_duration):
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Get the current timestamp
        #timestamp = time.strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Add microseconds
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]

        # Save the frame with timestamp in the name
        frame_name = os.path.join(output_folder, f'frame_{timestamp}.jpg')
        cv2.imwrite(frame_name, frame)
        print(f'Captured {frame_name}')

        # Display the frame
        cv2.imshow('Live Feed', frame)
        cv2.waitKey(1)  # Wait for a short time to show the frame

        # Calculate the time to sleep to achieve 30 frames per second
        time_to_sleep = start_time + i / frame_rate - time.time()
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

finally:
    # Release the webcam and destroy the OpenCV window
    cap.release()
    cv2.destroyAllWindows()