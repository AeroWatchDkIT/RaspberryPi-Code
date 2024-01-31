# camera_utils.py

import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
from ocr_utils import OCRUtils  # Import OCRUtils class from ocr_utils.py

class CameraUtils:
    # function to measure the frames per second
    @staticmethod
    def measure_fps(camera,rawCapture, num_frames=60):
        start = time.time()
        for _ in range(num_frames):
            camera.capture(rawCapture, format="bgr", use_video_port=True)
            rawCapture.truncate(0)
        end = time.time()
        return num_frames / (end - start)

    # Generate frame by frame from camera
    @staticmethod   
    def generate_frames():  
        global camera_is_active
        with PiCamera() as camera:
            
            camera_is_active = True

            #different resolutions
            camera.resolution = (640, 480)
            rawCapture = PiRGBArray(camera, size=(640, 480))
            #camera.resolution = (1280, 720)
            #rawCapture = PiRGBArray(camera, size=(1280, 720)

            # Allow the camera to warmup
            time.sleep(0.1)
            
            frame_count = 0
            ocr_interval = 10  # Perform OCR every 10 frames
            fps = CameraUtils.measure_fps(camera,rawCapture)  # Measure FPS
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
                cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"OCR Interval: {ocr_interval}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw the blue ROI rectangle here
                # Get the ROI and its top-left corner coordinates
                roi, x_offset, y_offset = OCRUtils.get_centered_roi(image, scale=0.5)
                
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


                if frame_count % ocr_interval == 0:
                    image = OCRUtils.perform_ocr(image,roi,x_offset,y_offset)  # Apply OCR to every 30th frame
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
