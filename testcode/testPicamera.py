from picamera2 import Picamera2
import cv2
import numpy as np

# Initialize Picamera2
picam2 = Picamera2(1)

# Configure the camera
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)

# Start the camera
picam2.start()

while True:
    # Capture an image
    image = picam2.capture_array()

    # Convert the image to a format OpenCV can use
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Display the image
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

# Stop the camera and close OpenCV windows
picam2.stop()
cv2.destroyAllWindows()