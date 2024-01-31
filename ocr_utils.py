# ocr_utils.py

import cv2
import pytesseract
from datetime import datetime, timedelta
import re

class OCRUtils:
    @staticmethod
    def preprocess_image_for_ocr(frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Resize for consistency
        resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        return resized
   
    # function to perform OCR on the frame
    @staticmethod
    def perform_ocr(frame,roi, x_offset, y_offset):
        preprocessed_frame = OCRUtils.preprocess_image_for_ocr(frame)
        global detected_text, last_sent_text  # Use the global variable to store the detected text
        global global_detection_time  # Use the global variable to store the detection time
        
        # OCR Configuration: Using psm 6 for detecting a block of text
        custom_config = r'--oem 3 --psm 6'
        
        # Use pytesseract to get the bounding box coordinates and other data of each text segment
        data = pytesseract.image_to_data(preprocessed_frame, config=custom_config, output_type=pytesseract.Output.DICT)

        full_text = ""  # String to accumulate detected text
        # Regular expression to match pattern 'A1.C1.R1'
        pattern = r'\b[A-Z]\d\.[C][1-9]\.[R][1-9]\b'

        # Regular expression to match pattern 'P-A1.C1.R1'
        #pattern = r'[A-Z]-[A-Z]\d{1,2}\.[C][1-9]\.[R][1-9]'


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

        return frame
    
    @staticmethod
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