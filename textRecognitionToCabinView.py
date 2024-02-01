from flask import Flask, render_template, Response
import cv2
import pytesseract
import re
import time
from datetime import datetime ,timedelta
from picamera.array import PiRGBArray
from picamera import PiCamera
import threading
import os
import numpy as np
from flask_cors import CORS
from flask import jsonify
import logging
import requests
import json
from requests.adapters import HTTPAdapter, Retry
import urllib3
import RPi.GPIO as GPIO

from tag_parser import TagParser
# from ocr_utils import OCRUtils
#from camera_utils import CameraUtils

#from pallet_database  import Database

# Use GPIO numbers not pin numbers
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin (e.g., GPIO 17) to which the buzzer is connected
buzzer_pin = 18
GPIO.setup(buzzer_pin, GPIO.OUT)

# Function to buzz the buzzer
def buzz(duration):
    GPIO.output(buzzer_pin, True)
    time.sleep(duration)
    GPIO.output(buzzer_pin, False)

# Suppress only the single InsecureRequestWarning from urllib3 needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
CORS(app)

detected_text = None  # Global variable to store the detected text
last_sent_text = None
global_detection_time = None # Global variable for detection time
camera_is_active = False # Global flag to indicate the status of the camera


# Setting the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# # function to preprocess the image for OCR
def preprocess_image_for_ocr(frame, gamma = 2):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #gamma = 1.5
    # Gamma correction for brightness adjustment
    #invGamma = 1.0 / gamma
    #table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    #brightened_gray = cv2.LUT(gray, table)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Resize for consistency
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    return resized

# # function to perform OCR on the frame
def perform_ocr(frame,roi, x_offset, y_offset):
    preprocessed_frame = preprocess_image_for_ocr(frame)
    global detected_text, last_sent_text  # Use the global variable to store the detected text
    global global_detection_time  # Use the global variable to store the detection time
    
    # OCR Configuration: Using psm 6 for detecting a block of text
    custom_config = r'--oem 3 --psm 6'
    
    # Use pytesseract to get the bounding box coordinates and other data of each text segment
    data = pytesseract.image_to_data(preprocessed_frame, config=custom_config, output_type=pytesseract.Output.DICT)

    full_text = ""  # String to accumulate detected text
    # Regular expression to match pattern 'A1.C1.R1'
    #pattern = r'\b[A-Z]\d\.[C][1-9]\.[R][1-9]\b'

    # Regular expression to match pattern 'P-A1.C1.R1'
    pattern = r'[A-Z]-[A-Z]\d{1,2}\.[C][1-9]\.[R][1-9]'


    # Regular expression to match COMBILIFT pattern A03G08/ Letter followed by 2 digits followed by letter followed by 2 digits
    #pattern = r'\b[A-Z]\d{2}[A-Z]\d{2}\b'
    
    # Padding or margin to apply around the text
    padding = 5  # Adjust the padding as needed

    for i in range(len(data['text'])):
        # Only process if confidence is high and text is non-empty
        #if int(data['conf'][i]) > 40 and data['text'][i].strip() != '':
            # These coordinates are relative to the preprocessed_frame (ROI)
            #x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Accumulate text
        full_text += data['text'][i] + " "
            
    matches = re.findall(pattern, full_text)

    # Print the entire text if it matches the pattern
    if matches:
        
        detected_text = ' '.join(matches)
        # Check if the detected text is different from the last sent text
        if detected_text != last_sent_text:
            # Update the global detection time variable
            last_sent_text = detected_text
            global_detection_time = datetime.now().strftime("%H:%M:%S")
            current_time = global_detection_time
            print(f"Detected Text: {detected_text } at {current_time}")  # Print text with time
            
            # Buzz the buzzer
            buzz(0.5)  # Buzz for half a second

    return frame

# function to split the frame into two ROIs
def get_split_roi(frame, scale=0.5):
    """
    Get two split ROIs in the frame, one for the top half and one for the bottom half.
    :param frame: The input frame.
    :param scale: Fraction of width to include in the ROIs.
    :return: Tuple containing the two ROIs and their x, y coordinates.
    """
    h, w = frame.shape[:2]
    roi_width = int(scale * w)
    roi_height = h // 2  # Split the height into two

    # Top ROI (for pallet tag)
    x0_top, y0_top = (w - roi_width) // 2, 0
    top_roi = frame[y0_top:y0_top + roi_height, x0_top:x0_top + roi_width]

    # Bottom ROI (for shelf tag)
    x0_bottom, y0_bottom = (w - roi_width) // 2, roi_height
    bottom_roi = frame[y0_bottom:y0_bottom + roi_height, x0_bottom:x0_bottom + roi_width]

    return (top_roi, x0_top, y0_top), (bottom_roi, x0_bottom, y0_bottom)
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
         
# function to measure the frames per second
def measure_fps(camera,rawCapture, num_frames=60):
    start = time.time()
    for _ in range(num_frames):
        camera.capture(rawCapture, format="bgr", use_video_port=True)
        rawCapture.truncate(0)
    end = time.time()
    return num_frames / (end - start)

# Generate frame by frame from camera   
def generate_frames():  
    global camera_is_active
    with PiCamera() as camera:
        
        camera_is_active = True

        #different resolutions
        camera.resolution = (640, 480)
        rawCapture = PiRGBArray(camera, size=(640, 480))
        #camera.resolution = (1280, 720)
        #rawCapture = PiRGBArray(camera, size=(1280, 720))

        # Allow the camera to warmup
        time.sleep(0.1)
        
        frame_count = 0
        ocr_interval = 10  # Perform OCR every 10 frames
        fps = measure_fps(camera,rawCapture)
        #ocr_interval = int(fps * 0.33)  # Adjust OCR frequency based on FPS
        
        print(f"FPS: {fps:.2f}")
        print(f"OCR Interval: {ocr_interval}")

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
        
            # Set line length
            line_length = 30  # Length of the line (30 pixels)

            # Width and height of the frame
            frame_width = image.shape[1]
            frame_height = image.shape[0]

            # Middle position for left and right lines (vertical)
            mid_vertical = frame_height // 2 # Position at half the height of the frame

            # Middle position for top and bottom lines (horizontal)
            mid_horizontal = frame_width // 2  # Position at half the width of the frame

            # Draw lines on each side of the frame
            # Left side (horizontal line at the middle)
            cv2.line(image,  (0, mid_vertical), (line_length, mid_vertical),   (0, 255, 0), 2)

            # Right side (horizontal line at the middle)
            cv2.line(image,  (frame_width - line_length, mid_vertical), (frame_width, mid_vertical),  (0, 255, 0), 2)

            # Top side (vertical line at the middle)
            cv2.line(image, (mid_horizontal, 0),  (mid_horizontal, line_length), (0, 255, 0), 2)

            # Bottom side (vertical line at the middle)
            cv2.line(image,  (mid_horizontal, frame_height - line_length), (mid_horizontal, frame_height), (0, 255, 0), 2)
            
            # Draw FPS on the frame
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"OCR Interval: {ocr_interval}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw the blue ROI rectangle here
            # Get the ROI and its top-left corner coordinates
            roi, x_offset, y_offset = get_centered_roi(image, scale=0.5)
            
            # Variables for adjusting the ROI position
            x_left_adjustment = 0  # Adjust this value to move the ROI left/right
            x_right_adjustment = 0  # Adjust this value to move the ROI left/right
            y_top_adjustment = 0  # Adjust this value to move the ROI up/down
            y_bottom_adjustment = 0  # Adjust this value to move the ROI up/down
           
            # Modify y_offset to start the ROI from halfway down the frame
            h = image.shape[0]  # Height of the frame
            y_offset = h // 2 + y_top_adjustment # Update y_offset to half the height of the frame


            # Calculate the new height to be half of the ROI height
            new_height = (roi.shape[0] // 2 ) - y_bottom_adjustment
        
            # Draw a blue rectangle around the ROI on the original frame
            cv2.rectangle(image, (x_offset + x_left_adjustment, y_offset), (x_offset + roi.shape[1] - x_right_adjustment, y_offset + new_height), (255, 0, 0), 2)

            # Calculate the new y_offset for the second rectangle
            # This positions it immediately below the first rectangle
            second_rectangle_y_offset = y_offset + new_height

            # Draw the second blue rectangle below the first one
            # The width and height can be the same as the first rectangle
            cv2.rectangle(image, (x_offset, second_rectangle_y_offset), (x_offset + roi.shape[1], second_rectangle_y_offset + new_height), 
              (255, 0, 0), 2)

            # When you need to call OCR methods, use OCRUtils. For example:
    
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

# def main():
#     # Initialize any required variables or configurations

#     # Start camera feed and process frames
#     CameraUtils.generate_frames()

# Example usage of tag parsing
tag_code = "S-0002.C2.R4"
shelf_id, column, row = TagParser.parse_tag_code(tag_code)

# Constructing JSON data
json_data = TagParser.construct_json(shelf_id)

# URL of the backend endpoint
url = "https://192.168.1.2:7128/Interactions"
#url = "https://192.168.16.168:7128/Interactions"
headers = {"Content-Type": "application/json"}

# Making the POST request
response = TagParser.post_data(url, json_data)

response = requests.post(url, data=json_data,headers=headers, verify=False)

# Processing the response
if response.status_code == 200:  # Or another success code as per your API
    print("Success:", response.text)
else:
    print("Error:", response.status_code, response.text)

#parsed = response.text
#print(parsed)

@app.route('/')
def index():
    # Render template (make sure you have an 'index.html' template in 'templates' directory)
    return render_template('index.html')
   
@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag in 'index.html'
    #if the video stream is not working send a string response to the client
    #return Response(CameraUtils.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API endpoint to get the detected text
@app.route('/get_detected_text')
def get_detected_text():
    #return jsonify({"detected_text": detected_text, "detection_time": global_detection_time})
    global global_detection_time, last_sent_text
    # Check if there's new text to send
    if last_sent_text:
        response = {"detected_text": last_sent_text, "detection_time": global_detection_time}
        # Reset the last sent text to prevent repeated sending
        last_sent_text = None
    else:
        response = {"detected_text": "", "detection_time": ""}

    return jsonify(response)

def run_flask():
    # Set log level to WARNING to hide regular route access logs
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

if __name__ == '__main__':
    # Run Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    #CameraUtils.generate_frames()
    #main()

    print("Enter 'q' to quit:")
    while True:  # Main thread for user input
        if input() == 'q':
            # Use os._exit(0) to close the entire process including all threads
            os._exit(0)

 #usign the database class to connect to the database
    # if db.connect():q
    #     # Perform some database operation, for example, fetching data
    #     results = db.execute_query("SELECT * FROM Pallet")
    #     db.close()

    # Print results to the console
        #print("Database Results:", results)