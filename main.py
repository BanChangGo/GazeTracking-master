"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.eyemovement import EyeMovementTracker
from gaze_tracking.face_detector3 import FaceRecognizer

gaze = GazeTracking()
eye_tracker = EyeMovementTracker()
webcam = cv2.VideoCapture(0)
recognizer = FaceRecognizer()

while True:
    

    
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    # gaze_tracking?ùò refresh?ï®?àò?óê ?õπÏ∫°Ïóê?Ñú ?ùΩ??? frame?ùÑ ?Ñ£?ñ¥ ?ò∏Ï∂úÌï®
    # refresh?äî cv2 ?†Å?ö©?ïú frameÍ≥? faces Î≥??àòÎ•? Î∞òÌôò -> face detect?óê Î≥??àòÎ°úÏç® ?ì∏ ?àò ?ûà?ùå 
    frame, faces = gaze.refresh(frame)

    # ?îÑ?†à?ûÑ?óê cv2 ?†Å?ö©, ?ñºÍµ? ?ÉêÏß? 
    gaze._analyze()
    frame = gaze.annotated_frame()

    horizontal_ratio = gaze.horizontal_ratio()
    pupil_coords = gaze.pupil_left_coords()  

 
    eye_tracker.update(horizontal_ratio, pupil_coords)

    frame = gaze.annotated_frame()
    text_blink = ""
    text_left = ""
    text_right = ""
    text_center = ""
    text_ratio = ""

    print(f"is_blinking() returns: {gaze.is_blinking()}")
    print(f"is_right() returns: {gaze.is_right()}")


    if gaze.is_blinking is not None:
        text_blink = str(gaze.is_blinking())
    if gaze.is_right():
        text_right = "Looking right"
    if gaze.is_left():
        text_left = "Looking left"
    if gaze.is_center():
        text_center = "Looking center"
    text_ratio = str(gaze.horizontal_ratio())

    cv2.putText(frame, text_blink, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.putText(frame, text_left, (20, 270), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 0), 2)
    cv2.putText(frame, text_ratio, (200, 270), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.putText(frame, text_right, (400, 270), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 2)
    cv2.putText(frame, text_center, (180, 90), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
    
webcam.release()
cv2.destroyAllWindows()
