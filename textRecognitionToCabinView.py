from flask import Flask, render_template, Response
import cv2
import pytesseract
import re
import time
from datetime import datetime
from picamera.array import PiRGBArray
from picamera import PiCamera
import threading
import os

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
# function to perform OCR on the frame
def perform_ocr(frame,roi, x_offset, y_offset):
    preprocessed_frame = preprocess_image_for_ocr(frame)
    
    # OCR Configuration: Using psm 6 for detecting a block of text
    custom_config = r'--oem 3 --psm 6'
    
    # Use pytesseract to get the bounding box coordinates and other data of each text segment
    data = pytesseract.image_to_data(preprocessed_frame, config=custom_config, output_type=pytesseract.Output.DICT)

    full_text = ""  # String to accumulate detected text
    # Regular expression to match pattern 'A1.C1.R1'
    pattern = r'\b[A-Z]\d\.[C][1-9]\.[R][1-9]\b'
    
    # Padding or margin to apply around the text
    padding = 5  # Adjust the padding as needed

    for i in range(len(data['text'])):
        # Only process if confidence is high and text is non-empty
        #if int(data['conf'][i]) > 40 and data['text'][i].strip() != '':
            # These coordinates are relative to the preprocessed_frame (ROI)
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Accumulate text
            full_text += data['text'][i] + " "
            
            # # To draw on the full frame, offset by the ROI's positionq
            # x_full_frame = x_offset + x - padding  # Apply padding to the left
            # y_full_frame = y_offset + y - padding  # Apply padding to the top
            # # Draw a green rectangle around the text on the full frame
            # frame = cv2.rectangle(frame, (x_full_frame, y_full_frame),                              (x_full_frame , y_full_frame ), (0, 255, 0), 2)
            
                        
            # Check if the current text segment matches the pattern
            if re.search(pattern, full_text):
                # These coordinates are relative to the preprocessed_frame (ROI)
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                print(f"Detected text: '{full_text}' with confidence: {data['conf'][i]}")
                print(f"Bounding box coordinates: x={x}, y={y}, w={w}, h={h}")
                
                # Adjust coordinates for the original frame
                x_full_frame = x_offset + x
                y_full_frame = y_offset + y
                
                print(f"Drawing rectangle around matched text: '{full_text}'")
                # Draw a green rectangle around the text that matches the pattern
                frame = cv2.rectangle(frame, (x_full_frame, y_full_frame),
                                      (x_full_frame + w, y_full_frame + h), (0, 255, 0), 2)

    

    matches = re.findall(pattern, full_text)

    # Print the entire text if it matches the pattern
    if matches:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get the current time
        print(f"Detected Text: {' '.join(matches)} at {current_time}")  # Print text with time

    return frame

def get_centered_roi(frame, scale=0.5):
    """
    Get a centered ROI in the frame.
    :param frame: The input frame.
    :param scale: Fraction of width and height to include in the ROI.
    :return: Cropped ROI, x-coordinate of the top-left corner, y-coordinate of the top-left corner, and size.
    """
    h, w = frame.shape[:2]
    roi_width, roi_height = int(scale * w), int(scale * h)
    x0, y0 = (w - roi_width) // 2, (h - roi_height) // 2
    return frame[y0:y0 + roi_height, x0:x0 + roi_width], x0, y0


def views(mode: int, confidence: int):
    """
    View modes changes the style of text-boxing in OCR.

    View mode 1: Draws boxes on text with >75 confidence level

    View mode 2: Draws red boxes on low-confidence text and green on high-confidence text

    View mode 3: Color changes according to each word's confidence; brighter indicates higher confidence

    View mode 4: Draws a box around detected text regardless of confidence

    :param mode: view mode
    :param confidence: The confidence of OCR text detection

    :returns: confidence threshold and (B, G, R) color tuple for specified view mode
    """
    conf_thresh = None
    color = None

    if mode == 1:
        conf_thresh = 75  # Only shows boxes with confidence greater than 75
        color = (0, 255, 0)  # Green

    if mode == 2:
        conf_thresh = 0  # Will show every box
        if confidence >= 50:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 255)  # Red

    if mode == 3:
        conf_thresh = 0  # Will show every box
        color = (int(float(confidence)) * 2.55, int(float(confidence)) * 2.55, 0)

    if mode == 4:
        conf_thresh = 0  # Will show every box
        color = (0, 0, 255)  # Red

    return conf_thresh, color

          
# function to measure the frames per second
def measure_fps(camera,rawCapture, num_frames=60):
    start = time.time()
    for _ in range(num_frames):
        camera.capture(rawCapture, format="bgr", use_video_port=True)
        rawCapture.truncate(0)
    end = time.time()
    return num_frames / (end - start)

# function to   
def generate_frames():  # Generate frame by frame from camera

    with PiCamera() as camera:
        camera.resolution = (640, 480)
        rawCapture = PiRGBArray(camera, size=(640, 480))
        # Allow the camera to warmup
        time.sleep(0.1)
        
        frame_count = 0
        ocr_interval = 5  # Perform OCR every 30 frames
        fps = measure_fps(camera,rawCapture)
        #ocr_interval = int(fps * 0.33)  # Adjust OCR frequency based on FPS
        
        print(f"FPS: {fps:.2f}")
        print(f"OCR Interval: {ocr_interval}")


        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
        
            
            # Draw FPS on the frame
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"OCR Interval: {ocr_interval}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw the blue ROI rectangle here
            # Get the ROI and its top-left corner coordinates
            roi, x_offset, y_offset = get_centered_roi(image, scale=0.5)
            # Modify y_offset to start the ROI from halfway down the frame
            h = image.shape[0]  # Height of the frame
            y_offset = h // 2  # Update y_offset to half the height of the frame

            # Calculate the new height to be half of the ROI height
            new_height = roi.shape[0] // 2
        
            # Draw a blue rectangle around the ROI on the original frame
            cv2.rectangle(image, (x_offset, y_offset), (x_offset + roi.shape[1], y_offset + new_height), (255, 0, 0), 2)

            if frame_count % ocr_interval == 0:
                image = perform_ocr(image,roi,x_offset,y_offset)  # Apply OCR to every 30th frame
                # Start OCR in a new thread
                #ocr_thread = threading.Thread(target=perform_ocr, args=(image,))
                #ocr_thread.start()

            # Show the frame with OCR applied
            cv2.imshow('OCR Camera View', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
                break

            # Convert the image to bytes and yield it
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
            frame_count += 1
            rawCapture.truncate(0)

@app.route('/')
def index():
    # Render template (make sure you have an 'index.html' template in 'templates' directory)
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag in 'index.html'
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

if __name__ == '__main__':
    # Run Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    print("Enter 'q' to quit:")
    while True:  # Main thread for user input
        if input() == 'q':
            # Use os._exit(0) to close the entire process including all threads
            os._exit(0)

