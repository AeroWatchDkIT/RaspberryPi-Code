import unittest
import cv2
import numpy as np
import sys

sys.path.append('/path/to/your/project/directory')  # Adjust this path
import textOCROptimize

class TextOCROptimizeTest(unittest.TestCase):
    def setUp(self):
        # This method will be run before each test method
        self.test_image_path = '/home/teo/Desktop/CloneNewTwoRois/RaspberryPi-Code/operatorImages/1.jpg'  # Adjust the path to your test image

    def test_image_preprocessing(self):
        frame = cv2.imread(self.test_image_path)
        self.assertIsNotNone(frame, "Test image could not be loaded.")

        processed_image = textOCROptimize.preprocess_image_for_ocr(frame)

        # Example assertions for preprocess_image_for_ocr
        self.assertEqual(len(processed_image.shape), 2, "Processed image is not grayscale.")

    # def test_another_function(self):
    #     # Example of how to test another function from textOCROptimize
    #     result = textOCROptimize.another_function(some_input)
    #     self.assertEqual(some_expected_result, result, "another_function did not return expected result.")

if __name__ == '__main__':
    unittest.main()