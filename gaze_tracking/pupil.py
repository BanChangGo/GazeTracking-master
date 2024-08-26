import numpy as np
import cv2


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold, margin):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None
        self.margin = margin

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold, margin):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        # Apply margin adjustments if needed
        height, width = new_frame.shape[:2]
        new_frame = new_frame[margin:height-margin, margin:width-margin]

        return new_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass

    @staticmethod
    def debug_image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris and returns intermediate steps.
        
        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame
        
        Returns:
            A tuple containing the processed frames
        """
        processed_frames = []

        # Original frame
        processed_frames.append(eye_frame)

        # Bilateral Filter
        bilateral_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        processed_frames.append(bilateral_frame)

        # Erosion
        kernel = np.ones((3, 3), np.uint8)
        eroded_frame = cv2.erode(bilateral_frame, kernel, iterations=3)
        processed_frames.append(eroded_frame)

        # Thresholding
        threshold_frame = cv2.threshold(eroded_frame, threshold, 255, cv2.THRESH_BINARY)[1]
        processed_frames.append(threshold_frame)

        return processed_frames