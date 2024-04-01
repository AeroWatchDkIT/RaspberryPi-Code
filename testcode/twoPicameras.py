from picamera2 import Picamera2
import cv2

def display_dual_cameras():
    # Initialize the first camera
    picam1 = Picamera2(0)
    picam1.start()

    # Initialize the second camera
    picam2 = Picamera2(1)
    picam2.start()

    try:
        while True:
            # Capture frame from the first camera
            frame1 = picam1.capture_array()

            # Capture frame from the second camera
            frame2 = picam2.capture_array()

            # Display frame from the first camera
            cv2.imshow('Camera 1', frame1)

            # Display frame from the second camera
            cv2.imshow('Camera 2', frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release resources
        picam1.stop()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
     display_dual_cameras()

#TEST camera 1 - the big camera
# from picamera2 import Picamera2
# import cv2

# def display_dual_cameras():
#     # Initialize the first camera
#     picam1 = Picamera2(0)
#     picam1.start()

#     # Initialize the second camera
#     # picam2 = Picamera2(1)
#     # picam2.start()

#     try:
#         while True:
#             # Capture frame from the first camera
#             frame1 = picam1.capture_array()

#             # Capture frame from the second camera
#             #frame2 = picam2.capture_array()

#             # Display frame from the first camera
#             cv2.imshow('Camera 1', frame1)

#             # Display frame from the second camera
#             #cv2.imshow('Camera 2', frame2)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         # Release resources
#         picam1.stop()
#         #picam2.stop()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     display_dual_cameras()

#TEST camera 2 - the small camera
# from picamera2 import Picamera2
# import cv2

# def display_dual_cameras():
   
#     # Initialize the second camera
#     picam2 = Picamera2(1)
#     picam2.start()

#     try:
#         while True:
           
#             # Capture frame from the second camera
#             frame2 = picam2.capture_array()
#             cv2.imshow('Camera 2', frame2)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         # Release resources
#         picam2.stop()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     display_dual_cameras()

# from picamera2 import Picamera2, Preview
# from time import sleep
# picam0 = Picamera2(0)
# picam1 = Picamera2(1)
# picam0.start_preview(Preview.QTGL)
# picam1.start_preview(Preview.QTGL)
# picam0.start()
# picam1.start()
# sleep(10)
# picam0.capture_file("cam0.jpg")
# picam1.capture_file("cam1.jpg")
# picam0.stop()
# picam1.stop()
# picam0.stop_preview()
# picam1.stop_preview()