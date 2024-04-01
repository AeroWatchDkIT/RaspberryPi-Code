# facehaar.py

import os
import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2

def perform_face_recognition():
    # Start the webcam
    #cap = cv2.VideoCapture(0)
    # Initialize Picamera2
    picam2 = Picamera2(1)

    # Configure the camera
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)

    # Start the camera
    picam2.start()
    # Initialize the Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load all images from the operatorimages folder and create encodings
    image_folder = "operatorImages"
    known_face_encodings = []
    known_face_names = []

    # Iterate over the files in the operatorimages directory
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  
            print("Loading:", filename)  # Print the filename
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            # Check if face encodings are found
            image_encodings = face_recognition.face_encodings(image)
            if image_encodings:
                face_encoding = image_encodings[0]
                # Extract the person's name from the filename
                person_name = os.path.splitext(os.path.basename(filename))[0]
                
                known_face_encodings.append(face_encoding)
                known_face_names.append(person_name)
   
    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     print("Failed to grab frame")
        #     break
        # Capture frame-by-frame from the PiCamera
        frame = picam2.capture()

        # Convert the frame to grayscale for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the face from the frame
            face_frame = frame[y:y + h, x:x + w]

            # Convert the face to the RGB color space
            face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            # Create face encodings for the extracted face
            current_face_encoding = face_recognition.face_encodings(face_frame_rgb)

            if current_face_encoding:
                # Compare the extracted face with the known faces
                matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding[0])
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding[0])
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print(f"Access Granted for {name}")
                    cv2.putText(frame, f"{name} - Access Granted", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Frame', frame)
                    cv2.waitKey(4000)  # Pause for  (2 seconds)
                    #cap.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    name = "Unknown"
                    print("Access Not Granted")
                    cv2.putText(frame, "Unknown - Access Not Granted", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # If access is granted, return True
        # if matches[best_match_index]:
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     return True

    #cap.release()
    cv2.destroyAllWindows()
    return False
