from flask import Flask, render_template, Response
import cv2
import tesserocr
from PIL import Image
import re
import time
from datetime import datetime ,timedelta
from picamera2 import Picamera2
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
import subprocess

from tag_parser import TagParser
# from ocr_utils import OCRUtils
#from camera_utils import CameraUtils
from pathlib import Path
# Import the face recognition function
from facehaarPicamFunc import perform_face_recognition

tessdata_path = Path("/usr/share/tesseract-ocr/5/tessdata/")
tesApi = tesserocr.PyTessBaseAPI(path=str(tessdata_path), lang='eng')

# Use GPIO numbers not pin numbers
#GPIO.setmode(GPIO.BCM)
import gpiod
from time import sleep
import os

LED_PIN = 23  # Use the appropriate GPIO pin number
chip = gpiod.Chip('gpiochip4')
led_line = chip.get_line(LED_PIN)
led_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)


# Function to buzz the buzzer
def buzz(duration):
    led_line.set_value(1)
    sleep(duration)
    led_line.set_value(0)
    sleep(duration)
    
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


# # function to preprocess the image for OCR
def preprocess_image_for_ocr(frame, gamma = 2):
    #print("\033[93m Func - process_rois\033[0m")
    # Display the original image
    # cv2.imshow('Original Image', frame)
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Resize for consistency
    #resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    #resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
    
    # Display the preprocessed image
    #cv2.imshow('Preprocessed Image', thresh)
    #cv2.waitKey(0)  # Wait for a key press to close the window
    #cv2.destroyAllWindows()

    return thresh

# function to perform OCR on the frame
def perform_ocr(roi, roi_position):
    #print("\033[93m Func - perform_ocr\033[0m")
    global shelf_tag
    global pallet_tag

    preprocessed_frame = preprocess_image_for_ocr(roi)
    global detected_text, last_sent_text  # Use the global variable to store the detected text
    global global_detection_time  # Use the global variable to store the detection time
    
    # Set the image to Tesseract API
    eemaj = Image.fromarray(preprocessed_frame)
    tesApi.SetImage(eemaj)

    #data = tesApi.GetUTF8Text()
    full_text = tesApi.GetUTF8Text()

    # Regular expression to match pattern 'P-A1.C1.R1'
    #pattern = r'[A-Z]-[A-Z]\d{1,2}\.[C][1-9]\.[R][1-9]'
    pattern = r'[SP]-[A-Z]\d\.[A-Z]\d\.[A-Z]\d'
            
    matches = re.findall(pattern, full_text)
    # Print the entire text if it matches the pattern
    if matches:
        detected_text = ' '.join(matches)
        if detected_text.find("S-") != -1:#if the detected text is a shelf tag
            shelf_tag = detected_text
        elif detected_text.find("P-") != -1:#if the detected text is a pallet tag
            pallet_tag = detected_text

        if detected_text != last_sent_text:
            last_sent_text = detected_text
            #global_detection_time = datetime.now().strftime("%H:%M:%S")
            global_detection_time = datetime.now()
            print(f"Detected Text ({roi_position}): {detected_text} at {global_detection_time}")
            time.sleep(1) # sleep for a second to before buzzing the buzzer 
            buzz(0.5)

        return detected_text, roi_position
    else:
        print(f"No match found in ({roi_position}) ROI")
        return None, roi_position

# function to split the frame into two ROIs
def get_split_roi(frame, part):
    #print("\033[93m Func -  get_split_roi\033[0m")
    h, w = frame.shape[:2]
    roi_width = int(w)  # Full width
    roi_height = h // 2  # Half height

    if part == 'top':
        return frame[:roi_height, :], 0, 0
    elif part == 'bottom':
        return frame[roi_height:, :], 0, roi_height

    raise ValueError("Invalid part specified. Use 'top' or 'bottom'.")
         
# Generate frame by frame from camera   
def generate_frames():
    #print("\033[93m 4. Func - generate_frames\033[0m")
    # Initialize Picamera2
    picam2 = Picamera2(0)

    # Configure the camera
    config = picam2.create_video_configuration(main={"size": (640, 480)}, buffer_count=4)
    picam2.configure(config)

    # Start the camera
    picam2.start()

    frame_count = 0
    ocr_interval = 5  # Perform OCR every 10 frames
    try:
        while True:
            # Capture an image
            image = picam2.capture_array()
          
            # Convert the image to a format OpenCV can use
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Set line length
            line_length = 30  # Length of the line (30 pixels)

            # Width and height of the frame
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            # Middle position for left and right lines (vertical)
            mid_vertical = frame_height // 2 # Position at half the height of the frame

            # Middle position for top and bottom lines (horizontal)
            mid_horizontal = frame_width // 2  # Position at half the width of the frame

            # Draw lines on each side of the frame
            # Left side (horizontal line at the middle)
            cv2.line(frame,  (0, mid_vertical), (line_length, mid_vertical),   (0, 255, 0), 2)
            # Right side (horizontal line at the middle)
            cv2.line(frame,  (frame_width - line_length, mid_vertical), (frame_width, mid_vertical),  (0, 255, 0), 2)    

            # Split the frame into two ROIs
            top_roi, x_offset_top, y_offset_top = get_split_roi(frame, 'top')
            bottom_roi, x_offset_bottom, y_offset_bottom = get_split_roi(frame, 'bottom')
            
            # control Padding values for the top ROI
            top_padding_topROI = 80
            bottom_padding_topROI = 20
            # control Padding values for the bottom ROI
            top_padding_bottomROI = 60
            bottom_padding_bottomROI = 50

            left_padding = 100
            right_padding = 130

            # Draw the blue ROI rectangle here
            # Adjusted rectangle for the top ROI
            cv2.rectangle(frame, 
                            (x_offset_top + left_padding, y_offset_top + top_padding_topROI), 
                            (x_offset_top + top_roi.shape[1] - right_padding, y_offset_top + top_roi.shape[0] - bottom_padding_topROI), 
                            (255, 0, 0), 2)

            # Adjusted rectangle for the bottom ROI
            cv2.rectangle(frame, 
                            (x_offset_bottom + left_padding, y_offset_bottom + top_padding_bottomROI), 
                            (x_offset_bottom + bottom_roi.shape[1] - right_padding, y_offset_bottom + bottom_roi.shape[0] - bottom_padding_bottomROI), 
                            (255, 0, 0), 2)
                
            # to call OCR methods:
            if frame_count % ocr_interval == 0:
                    
                    process_rois(top_roi, x_offset_top, y_offset_top, bottom_roi, x_offset_bottom, y_offset_bottom,
                                top_padding_topROI, bottom_padding_topROI, left_padding, right_padding,
                                top_padding_bottomROI, bottom_padding_bottomROI)

            # Display the frame locally
            #cv2.imshow('OCR Camera View', frame)

            # Convert the frame to bytes and yield it for the network stream
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
                break
    finally:
        # Stop the camera and close OpenCV windows
        picam2.stop()
        cv2.destroyAllWindows()
        

# function to process the ROIs for OCR and send the detected tags to the API
def process_rois(top_roi, x_offset_top, y_offset_top, bottom_roi, x_offset_bottom, y_offset_bottom,
                top_padding_topROI, bottom_padding_topROI, left_padding, right_padding,
                top_padding_bottomROI, bottom_padding_bottomROI):
    #print("\033[93m Func -  process_rois\033[0m")
    global shelf_tag
    global pallet_tag

    # Reset detected tags before processing
    shelf_tag = None
    pallet_tag = None

    # Extract the relevant portion of the top ROI based on the rectangle
    top_roi_processed = top_roi[top_padding_topROI : top_roi.shape[0] - bottom_padding_topROI,
                                left_padding : top_roi.shape[1] - right_padding]

    # Extract the relevant portion of the bottom ROI based on the rectangle
    bottom_roi_processed = bottom_roi[top_padding_bottomROI : bottom_roi.shape[0] - bottom_padding_bottomROI,
                                  left_padding : bottom_roi.shape[1] - right_padding]
    
    pallet_tag, _ = perform_ocr(top_roi_processed, "top")
    shelf_tag, _ = perform_ocr(bottom_roi_processed, "bottom")
    #time.sleep(5) # Sleep for 5 seconds to prevent repeated OCR

    # Check if both tags were detected and send them to the API
    if pallet_tag and shelf_tag:
        TagParser.send_tags(shelf_tag, pallet_tag)
        # Consider resetting the global tags if necessary


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

@app.route('/face_recognition_feed')
def face_recognition_feed():
    return Response(perform_face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

# function to run the Flask server
def run_flask():
    # Set log level to WARNING to hide regular route access logs
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    print("\033[93m 3. Func - Starting Flask server on http://0.0.0.0:5001 \033[0m")
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)

# Function to free up the specified port
def free_up_port(port):
    #print("1. Func - Freeing up port...", color="yellow")
    print("\033[93m 1. Func - Freeing up port...\033[0m")
    find_process = f"lsof -t -i:{port}"
    process = subprocess.run(find_process, shell=True, capture_output=True, text=True)
    process_id = process.stdout.strip()

    if process_id:
        print(f"Killing process {process_id} on port {port}")
        kill_process = f"sudo kill -9 {process_id}"
        subprocess.run(kill_process, shell=True)
    else:
        print(f"No process is using port {port}")

#Function to attempt releasing the camera
def release_camera():
    print("\033[93m 2. Func - Releasing camera resource... \033[0m")
    try:
        camera = Picamera2(0)
        camera.close()
        print("Camera resource released.")
    except:
        print("No camera resource to release or error in releasing.")

if __name__ == '__main__':
    # Free up port 5001 before starting the Flask server
    free_up_port(5001)
    # Try to release the camera resource
    release_camera()
    # Perform face recognition before starting OCR
    access_granted = perform_face_recognition()
    
    #flask_thread = threading.Thread(target=run_flask)
    #flask_thread.start()

    if access_granted:
        print("Access Granted. Proceeding with OCR...")
        # Run Flask in a separate thread
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.start()
    else:
        print("Access Denied.")

    print("Enter 'q' to quit:")
    while True:  # Main thread for user input
        if input() == 'q':
            # Use os._exit(0) to close the entire process including all threads
            os._exit(0)

 
# Generate frame by frame from camera   
# def generate_frames():  
#     # Initialize the USB camera
#     camera = None
#     for attempt in range(5):  # Try to open the camera 5 times
#         camera = cv2.VideoCapture(0)  # Attempt to access the first camera
#         if camera.isOpened():
#             print("Camera opened successfully.")
#             break  # If the camera is opened successfully, exit the loop
#         else:
#             print(f"Failed to open camera on attempt {attempt + 1}/5. Retrying...")
#             time.sleep(2)  # Wait for 2 seconds before retrying

#     if not camera or not camera.isOpened():
#         print("Failed to open the camera. Please check the device and try again.")
#         return  # Exit the function if the camera cannot be opened
    
#     # Set the resolution to 640x480
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     if not camera.isOpened():
#         raise RuntimeError('Could not start camera.')
 
#     # Allow the camera to warmup
#     time.sleep(0.1)
    
#     frame_count = 0
#     ocr_interval = 5  # Perform OCR every 10 frames
#     # Capture frames from the camera
#     while True:
#         # Capture frame-by-frame
#         ret, image = camera.read()
#         if not ret:
#             print("Failed to capture frame from OCR camera.")
#             break  # Break if the frame is not captured successfully

#         print(f"Captured frame {frame_count} from OCR camera.")
#         # Set line length
#         line_length = 30  # Length of the line (30 pixels)

#         # Width and height of the frame
#         frame_width = image.shape[1]
#         frame_height = image.shape[0]

#         # Middle position for left and right lines (vertical)
#         mid_vertical = frame_height // 2 # Position at half the height of the frame

#         # Middle position for top and bottom lines (horizontal)
#         mid_horizontal = frame_width // 2  # Position at half the width of the frame

#         # Draw lines on each side of the frame
#         # Left side (horizontal line at the middle)
#         cv2.line(image,  (0, mid_vertical), (line_length, mid_vertical),   (0, 255, 0), 2)
#         # Right side (horizontal line at the middle)
#         cv2.line(image,  (frame_width - line_length, mid_vertical), (frame_width, mid_vertical),  (0, 255, 0), 2)
        

#         # Split the frame into two ROIs
#         top_roi, x_offset_top, y_offset_top = get_split_roi(image, 'top')
#         bottom_roi, x_offset_bottom, y_offset_bottom = get_split_roi(image, 'bottom')
    
#         # control Padding values for the top ROI
#         top_padding_topROI = 80
#         bottom_padding_topROI = 20
#         # control Padding values for the bottom ROI
#         top_padding_bottomROI = 60
#         bottom_padding_bottomROI = 50

#         left_padding = 100
#         right_padding = 130

#         # Draw the blue ROI rectangle here
#         # Adjusted rectangle for the top ROI
#         cv2.rectangle(image, 
#                     (x_offset_top + left_padding, y_offset_top + top_padding_topROI), 
#                     (x_offset_top + top_roi.shape[1] - right_padding, y_offset_top + top_roi.shape[0] - bottom_padding_topROI), 
#                     (255, 0, 0), 2)

#         # Adjusted rectangle for the bottom ROI
#         cv2.rectangle(image, 
#                     (x_offset_bottom + left_padding, y_offset_bottom + top_padding_bottomROI), 
#                     (x_offset_bottom + bottom_roi.shape[1] - right_padding, y_offset_bottom + bottom_roi.shape[0] - bottom_padding_bottomROI), 
#                     (255, 0, 0), 2)
        
#         # to call OCR methods:
#         #if frame_count % ocr_interval == 0:
            
#             # process_rois(top_roi, x_offset_top, y_offset_top, bottom_roi, x_offset_bottom, y_offset_bottom,
#             #             top_padding_topROI, bottom_padding_topROI, left_padding, right_padding,
#             #             top_padding_bottomROI, bottom_padding_bottomROI)
        
#         if image is not None and image.size != 0:
#              print("Image is not empty")
#              print(image.size)
#              print(image.shape)
#              print(image)
#              cv2.imshow('OCR Camera View', image)
#         else:
#             print("Empty or invalid frame received.")
        
#         # Show the frame with OCR applied
#         #cv2.imshow('OCR Camera View', image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
#             break

#         # Convert the image to bytes and yield it
#         ret, buffer = cv2.imencode('.jpg', image)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
#         frame_count += 1
#      # When everything is done, release the capture
#     camera.release()
#     cv2.destroyAllWindows()
            
            # # Capture an image
            # image = picam2.capture_array()
            # # Check if the captured image is not empty
            # if image is not None:
            #     print("Captured image shape:", image.shape)
            # else:
            #     print("Empty frame captured")
            #     continue  # Skip processing empty frames
            # print(image)

            # # Convert the image to a format OpenCV can use
            # frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # # Check if the frame is not empty
            # if frame is not None:
            #     print("Converted frame shape:", frame.shape)
            # else:
            #     print("Empty frame after conversion")
            #     continue  # Skip processing empty frames
            # print(frame)
            # # Display the captured frame
            # # Display the frame locally
            # try:
            #     print("Before displaying frame")
            #     #cv2.imshow('OCR Camera View', frame)
            #     print("After displaying frame")
            #     cv2.waitKey(1)
            #     print("After waitKey")
            # except Exception as e:
            #     print("Error displaying frame:", e)
            #     continue # Continue to the next iteration of the loop