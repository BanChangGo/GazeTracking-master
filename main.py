import cv2
import dlib
import numpy as np
import threading
from time import sleep

from gaze_tracking import GazeTracking
from gaze_tracking.eyemovement import EyeMovementTracker
from gaze_tracking.face_detector import FaceRecognizer
from gaze_tracking.sol_control import RelayController

gaze = GazeTracking()
eye_tracker = EyeMovementTracker()
recognizer = FaceRecognizer()
solanoid = RelayController()

choice = False

def input_listener():
    global choice
    while True:
        key = input("If you want to enter learner mode push l: \n")
        if key.lower() == 'l':
            choice = True
        elif key.lower() == 'q':
            break


def make_frame(cap):   
    ret, frame = cap.read()
    if not ret:
        return None
    return frame.copy()


def process_mode(frame):
    copy_frame = frame.copy()
    gaze.refresh(frame)
    horizontal_ratio = gaze.horizontal_ratio()
    pupil_coords = gaze.pupil_left_coords()

    '''
    
    --Start--Showing the points and the values of features

    '''
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

    '''
    --End--of Showing
    
    '''


    '''
    #Detecting the Eye movement (From left to right)
    '''
    detect = eye_tracker.update(horizontal_ratio, pupil_coords)

    
    '''
    #If Detect the movement successfully than Inspect the User (is authorized or not)
    '''
    if detect:
        ##print("detect to recognize")
        if(recognizer.recognize_faces(copy_frame) == "open"):
            print("OPEN in main")
            solanoid.relay_controller.on()
            sleep(2)
            solanoid.relay_controller.off()
            sleep(2)
                

        
                                            

   

    

def main():
    global choice
    cap = cv2.VideoCapture(0)

    # Threading that receive the choice Input( by GPIO later )
    input_thread = threading.Thread(target=input_listener)
    input_thread.daemon = True
    input_thread.start()

    while True:
        frame = make_frame(cap)
        if frame is None:  # handling the expectioin 
            break
        

        #By the value of choice, start the Learing Mode
        if choice:
            print("Lerning Mode")
            recognizer.learn_face(frame)
            choice = False
        else:
            process_mode(frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
