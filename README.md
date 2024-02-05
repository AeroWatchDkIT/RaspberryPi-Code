# OCR Camera System for Raspberry Pi

This project implements an Optical Character Recognition (OCR) system on the Raspberry Pi, designed to recognize and process text from a live camera feed. Utilizing the PiCamera and Tesseract OCR engine, this system can detect specific text patterns in real-time.

## Features

- **Real-Time Text Detection**: Detects and processes text from live camera feed in real-time.
- **Automatic Startup**: The system is configured to automatically start upon booting the Raspberry Pi, requiring no manual command to launch.
- **Easy Shutdown**: To safely turn off the device, simply press the on/off button. The system will handle safe shutdown procedures.
- **Tag Recognition**: Capable of recognizing predefined tag formats and performing actions based on the recognized tags.
- **Backend Integration**: Sends recognized tag information to a backend server for further processing or logging.
- **FPS Display**: Dynamically displays the Frames Per Second (FPS) on the camera feed for performance monitoring.

## Setup

To set up the OCR Camera System on your Raspberry Pi, follow these steps:

1. **Clone the repository**: Clone this repository to your Raspberry Pi.

    ```
    git clone https://github.com/AeroWatchDkIT/RaspberryPi-Code.git
    ```

2. **Install Dependencies**: Ensure you have Python 3 and the necessary libraries installed on your Raspberry Pi. This project requires [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), [PiCamera](https://picamera.readthedocs.io/en/release-1.13/), and OpenCV. 

    
    - **Tesseract OCR**: Used for optical character recognition.
        ```
        sudo apt-get update
        sudo apt-get install tesseract-ocr
        sudo apt-get install libtesseract-dev
        ```

    - **PiCamera**: Interface to the Raspberry Pi's camera module.
        ```
        sudo apt-get install python3-picamera
        ```

    - **OpenCV**: Open-source computer vision library.
        ```
        pip3 install opencv-python
        ```

    - **Flask**: Micro web framework for building web applications in Python.
        ```
        pip3 install Flask
        ```

    - **Requests**: Library for making HTTP requests in Python.
        ```
        pip3 install requests
		```
Adjust the above commands based on your specific dependencies. 
    

3. **Configure for auto-start**: To make the application run at boot, add it to the Raspberry Pi's `rc.local` file or create a systemd service.

4. **Adjust configurations**: Modify the `config.json` file (if present) to set up your specific parameters, like server URL and camera settings.

## Usage

Once set up, the OCR Camera System will start automatically when the Raspberry Pi boots up. Point the camera at the text you wish to recognize. The system will display the camera feed with the FPS and recognized text overlaid.

To safely shut down the system, press the on/off button on the Raspberry Pi. The application will handle any necessary cleanup before shutting down the device.

## Customization

You can customize the OCR patterns and actions taken upon recognition by modifying the `TagParser` and `OCRUtils` classes. For example, you can change the regular expression patterns to match different tag formats or modify the backend integration logic to suit your needs.




