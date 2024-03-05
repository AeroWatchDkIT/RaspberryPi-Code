import face_recognition
import os, sys
import cv2
import numpy as np
import math
from picamera.array import PiRGBArray
from picamera import PiCamera
import time


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        # Encode face

    def encode_faces(self):
        for image in os.listdir('operatorImages'):
            face_image = face_recognition.load_image_file(f'operatorImages/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split('.')[0])  # This will add just the number, without '.jpg'

        print(self.known_face_names)

    def run_recognition(self):
        camera = PiCamera()
        camera.resolution = (640, 480)  # Set the resolution
        rawCapture = PiRGBArray(camera, size=(640, 480))

        # Allow the camera to warm up
        time.sleep(0.1)

        frame_skip = 5  # Skip every 5 frames
        frame_count = 0

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            if frame_count % frame_skip == 0:
                image = frame.array

                if self.process_current_frame:
                    # Process frame
                    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = small_frame[:, :, ::-1]

                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = 'Unknown'
                        confidence = 'Unknown'

                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                        self.face_names.append(f'{name} ({confidence})')

                self.process_current_frame = not self.process_current_frame

                # Display annotations
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                    cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                cv2.imshow('Face Recognition', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            rawCapture.truncate(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()

# import cv2
# import face_recognition

# # Load a photograph of a certified driver
# known_image = face_recognition.load_image_file("/home/teo/Desktop/CloneNewTwoRois/RaspberryPi-Code/operatorImages/1.jpg")
# known_encoding = face_recognition.face_encodings(known_image)[0]

# # Initialize webcam
# video_capture = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     # Convert the image from BGR color (which OpenCV uses) to RGB color
#     rgb_frame = frame[:, :, ::-1]

#     # Find all the faces and face encodings in the current frame
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     for face_encoding in face_encodings:
#         # See if the face is a match for the known driver's face
#         matches = face_recognition.compare_faces([known_encoding], face_encoding)

#         if True in matches:
#             # The recognized face matches the certified driver
#             print("Driver recognized. Proceeding with OCR process.")
#             # Proceed with OCR process
#             break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
