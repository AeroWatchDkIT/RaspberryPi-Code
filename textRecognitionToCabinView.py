from flask import Flask, render_template, Response
import cv2
import pytesseract
import re
from datetime import datetime
from picamera.array import PiRGBArray
from picamera import PiCamera

app = Flask(__name__)

# Setting the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image_for_ocr(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Resize for consistency
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    return resized

def detect_features(frame):
    preprocessed_frame = preprocess_image_for_ocr(frame)
    
    # OCR Configuration: Using psm 6 for detecting a block of text
    custom_config = r'--oem 3 --psm 6'
    
    # Use pytesseract to get the bounding box coordinates and other data of each text segment
    data = pytesseract.image_to_data(preprocessed_frame, config=custom_config, output_type=pytesseract.Output.DICT)

    full_text = ""  # String to accumulate detected text

    for i in range(len(data['text'])):
        # Only process if confidence is high and text is non-empty
        if int(data['conf'][i]) > 60 and data['text'][i].strip() != '':
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Accumulate text
            full_text += data['text'][i] + " "
            
            # Draw a green rectangle around the text
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Regular expression to match pattern 'A1.C1.R1'
    pattern = r'\b[A-Z]\d\.[C][1-9]\.[R][1-9]\b'
    matches = re.findall(pattern, full_text)

    # Print the entire text if it matches the pattern
    if matches:
        current_time = datetime.now().strftime("%H:%M:%S")  # Get the current time
        print(f"Detected Text: {' '.join(matches)} at {current_time}")  # Print text with time

    return frame

def gen_frames():  # Generate frame by frame from camera
    with PiCamera() as camera:
        camera.resolution = (640, 480)
        rawCapture = PiRGBArray(camera, size=(640, 480))
        # Allow the camera to warmup
        time.sleep(0.1)

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            image = detect_features(image)  # Apply OCR to each frame

            # Show the frame with OCR applied
            cv2.imshow('OCR Camera View', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
                break

            # Convert the image to bytes and yield it
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            rawCapture.truncate(0)

@app.route('/')
def index():
    # Render template (make sure you have an 'index.html' template in 'templates' directory)
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag in 'index.html'
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
