import cv2
import dlib
import numpy as np

from gaze_tracking import GazeTracking
from gaze_tracking.eyemovement import EyeMovementTracker
from gaze_tracking.face_detector3_2 import FaceRecognizer

gaze = GazeTracking()
eye_tracker = EyeMovementTracker()
recognizer = FaceRecognizer()


def entry_mode():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            # ?��기에 ?��?�� 추적 로직 추�??

        original_frame = frame.copy()

        gaze.refresh(frame)
        horizontal_ratio = gaze.horizontal_ratio()
        pupil_coords = gaze.pupil_left_coords()

        detect = eye_tracker.update(horizontal_ratio, pupil_coords)
        if detect:
            recognizer.recognize_faces(original_frame)  # ?���?? ?��?�� ?��?�� ?���??

        # annotated_frame = gaze.annotated_frame()

        # text_blink = ""
        # text_left = ""
        # text_right = ""
        # text_center = ""
        # text_ratio = ""        

        # if gaze.is_blinking is not None:
        #     text_blink = str(gaze.is_blinking())
        # if gaze.is_right():
        #     text_right = "Looking right"
        # if gaze.is_left():
        #     text_left = "Looking left"
        # if gaze.is_center():
        #     text_center = "Looking center"
        # text_ratio = str(gaze.horizontal_ratio())
        
        # cv2.putText(annotated_frame, text_blink, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        # cv2.putText(annotated_frame, text_left, (20, 270), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 0), 2)
        # cv2.putText(annotated_frame, text_ratio, (200, 270), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        # cv2.putText(annotated_frame, text_right, (400, 270), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255), 2)
        # cv2.putText(annotated_frame, text_center, (180, 90), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        
        
        # cv2.imshow("Face Entry Mode", annotated_frame)

        # cv2.imshow("Face Entry Mode2", original_frame)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# 메인 함구
def main():
    while True:
        #print("1: 학습 모드, 2: 진입 모드, q: 종료")
        choice = False
        '''
            GPIO input => Ture
        ''' 
        entry_mode()

        if choice == True:
            print("학습 모드...")
            recognizer.recognize_faces()
        '''
        elif choice == '2':
            print("진입 모드...")
        
        elif choice == 'q':
            break
        else:
            print("?잘못된 선택.")
        '''
if __name__ == "__main__":
    main()
