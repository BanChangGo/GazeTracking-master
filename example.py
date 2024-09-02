"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.eyemovement import EyeMovementTracker

gaze = GazeTracking()
eye_tracker = EyeMovementTracker()
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    horizontal_ratio = gaze.horizontal_ratio()
    pupil_coords = gaze.pupil_left_coords()  


    ##¾ó±¼ Å½ÁöµÇ¸é ½ÇÇàµÇ°Ô
    eye_tracker.update(horizontal_ratio, pupil_coords)

    frame = gaze.annotated_frame()
    text_blink = ""
    text_left = ""
    text_right = ""
    text_center = ""
    text_ratio = ""

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
