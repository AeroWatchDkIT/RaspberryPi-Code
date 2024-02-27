import cv2
import face_recognition

# Load a photograph of a certified driver
known_image = face_recognition.load_image_file("/home/teo/Desktop/CloneNewTwoRois/RaspberryPi-Code/operatorImages/1.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known driver's face
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        if True in matches:
            # The recognized face matches the certified driver
            print("Driver recognized. Proceeding with OCR process.")
            # Proceed with OCR process
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
