from flask import Flask, render_template, Response
import cv2
import re
import time
import tesserocr
from PIL import Image
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

from pathlib import Path
tessdata_path = Path("/usr/share/tesseract-ocr/4.00/tessdata/")
tesApi = tesserocr.PyTessBaseAPI(path=str(tessdata_path), lang='eng')

# from ocr_utils import OCRUtils
#from camera_utils import CameraUtils

#from pallet_database  import Database

# Use GPIO numbers not pin numbers
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin (e.g., GPIO 17) to which the buzzer is connected
#buzzer_pin = 18
#GPIO.setup(buzzer_pin, GPIO.OUT)

# Function to buzz the buzzer
#def buzz(duration):
    #GPIO.output(buzzer_pin, True)
    #time.sleep(duration)
    #GPIO.output(buzzer_pin, False)

# Suppress only the single InsecureRequestWarning from urllib3 needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
CORS(app)

detected_text = None  # Global variable to store the detected text
shelf_tag =None # Global variable to store the shelf tag
pallet_tag = None # Global variable to store the pallet tag
last_sent_text = None
global_detection_time = None # Global variable for detection time
camera_is_active = False # Global flag to indicate the status of the camera


# Setting the path to the Tesseract executable
#pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# # function to preprocess the image for OCR
def preprocess_image_for_ocr(frame, gamma = 2):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Resize for consistency
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    return resized

# # function to perform OCR on the frame
def perform_ocr(roi, x_offset, y_offset, roi_position):
    global shelf_tag
    global pallet_tag

    preprocessed_frame = preprocess_image_for_ocr(roi)
    global detected_text, last_sent_text  # Use the global variable to store the detected text
    global global_detection_time  # Use the global variable to store the detection time
    
    # OCR Configuration: Using psm 6 for detecting a block of text
    custom_config = r'--oem 3 --psm 6'
    
    # Use pytesseract to get the bounding box coordinates and other data of each text segment
    #data = pytesseract.image_to_data(preprocessed_frame, config=custom_config, output_type=pytesseract.Output.DICT)
    eemaj = Image.fromarray(preprocessed_frame)
    tesApi.SetImage(eemaj)

    data = tesApi.GetUTF8Text()

    #print(data)

    full_text = ""  # String to accumulate detected text

    # Regular expression to match pattern 'P-A1.C1.R1'
    pattern = r'[A-Z]-[A-Z]\d{1,2}\.[C][1-9]\.[R][1-9]'

    # Padding or margin to apply around the text
    padding = 5  # Adjust the padding as needed

    return data, roi_position


    # for i in range(len(data['text'])):
            
    #         # Accumulate text
    #     full_text += data['text'][i] + " "
            
    # matches = re.findall(pattern, full_text)

    # if matches:
    #     detected_text = ' '.join(matches)
    #     if detected_text.find("S-") != -1:#if the detected text is a shelf tag
    #         shelf_tag = detected_text
    #     elif detected_text.find("P-") != -1:#if the detected text is a pallet tag
    #         pallet_tag = detected_text

    #     if detected_text != last_sent_text:
    #         last_sent_text = detected_text
    #         global_detection_time = datetime.now().strftime("%H:%M:%S")
    #         print(f"Detected Text ({roi_position}): {detected_text} at {global_detection_time}")
    #         #buzz(0.5)
    #         #do_some_bs()
    #     return detected_text, roi_position
    # else:
    #     return None, roi_position

def get_split_roi(frame, part):
    h, w = frame.shape[:2]
    roi_width = int(w)  # Full width
    roi_height = h // 2  # Half height

    if part == 'top':
        return frame[:roi_height, :], 0, 0
    elif part == 'bottom':
        return frame[roi_height:, :], 0, roi_height

    raise ValueError("Invalid part specified. Use 'top' or 'bottom'.")
         
# function to measure the frames per second
def measure_fps(camera,rawCapture, num_frames=60):
    start = time.time()
    for _ in range(num_frames):
        camera.capture(rawCapture, format="bgr", use_video_port=True)
        rawCapture.truncate(0)
    end = time.time()
    return num_frames / (end - start)

def is_non_empty_string(s):
    return bool(s and not s.isspace())

# Generate frame by frame from camera   
def generate_frames():  
    with PiCamera() as camera:

        #different resolutions
        camera.resolution = (640, 480)
        rawCapture = PiRGBArray(camera, size=(640, 480))
 
        # Allow the camera to warmup
        time.sleep(0.1)
        
        frame_count = 0
        ocr_interval = 5  # Perform OCR every 10 frames
        fps = measure_fps(camera,rawCapture)


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
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"OCR: {ocr_interval}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Split the frame into two ROIs
            top_roi, x_offset_top, y_offset_top = get_split_roi(image, 'top')
            bottom_roi, x_offset_bottom, y_offset_bottom = get_split_roi(image, 'bottom')

            # Perform OCR on top and bottom ROIs, passing 'top' and 'bottom' as roi_position
            top_text, top_position = perform_ocr(top_roi, x_offset_top, y_offset_top, "top")
            bottom_text, bottom_position = perform_ocr(bottom_roi, x_offset_bottom, y_offset_bottom, "bottom")

            pattern = re.compile(r'[SP]-[A-Z]\d\.[A-Z]\d\.[A-Z]\d')
            match = pattern.search(top_text)

            if match:
                extracted_code = match.group()
                print(f"Extracted code: {extracted_code} / Time: {datetime.now()}")

            # if is_non_empty_string(top_text):
            #     print(top_text)
            
            # if is_non_empty_string(bottom_text):
            #     print(bottom_text)

            # Padding values
            top_padding = 20
            bottom_padding = 20
            left_padding = 40
            right_padding = 40
            # Draw the blue ROI rectangle here

            # Adjusted rectangle for the top ROI
            cv2.rectangle(image, 
                        (x_offset_top + left_padding, y_offset_top + top_padding), 
                        (x_offset_top + top_roi.shape[1] - right_padding, y_offset_top + top_roi.shape[0] - bottom_padding), 
                        (255, 0, 0), 2)

            # Adjusted rectangle for the bottom ROI
            cv2.rectangle(image, 
                        (x_offset_bottom + left_padding, y_offset_bottom + top_padding), 
                        (x_offset_bottom + bottom_roi.shape[1] - right_padding, y_offset_bottom + bottom_roi.shape[0] - bottom_padding), 
                        (255, 0, 0), 2)
            # top ROI rectangle
            # cv2.rectangle(image, (x_offset_top, y_offset_top), (x_offset_top + top_roi.shape[1], y_offset_top + top_roi.shape[0]), (255, 0, 0), 2)
            # # bottom ROI rectangle
            # cv2.rectangle(image, (x_offset_bottom, y_offset_bottom), (x_offset_bottom + bottom_roi.shape[1], y_offset_bottom + bottom_roi.shape[0]), (255, 0, 0), 2)

            # to call OCR methods:
            #if frame_count % ocr_interval == 0:
                # Recheck this part if needed, since OCR is already performed above
                #top_text, top_position = perform_ocr(top_roi, x_offset_top, y_offset_top, "top")
                #bottom_text, bottom_position = perform_ocr(bottom_roi, x_offset_bottom, y_offset_bottom, "bottom")
            

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
#tag_code = "S-0002.C2.R4"
# pattern S-A5.C3.R6
# tag_code = "S-A5.C3.R6"
# shelf_id, column, row = TagParser.parse_tag_code(tag_code)

def do_some_bs():
    # Constructing JSON data
    json_data = TagParser.construct_json(shelf_id=shelf_tag, pallet_id=pallet_tag)

    # URL of the backend endpoint
    #url = "https://192.168.1.2:7128/Interactions"
    url = "https://192.168.16.222:7128/Interactions/TwoCodes"
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
    global global_detection_time, last_sent_text, roi_position
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
    