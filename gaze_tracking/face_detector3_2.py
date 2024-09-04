import os
import cv2
import dlib
import numpy as np
import pickle

class FaceRecognizer:
    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
        self.predictor = dlib.shape_predictor(model_path)

        cwd2 = os.path.abspath(os.path.dirname(__file__))
        model_path2 = os.path.abspath(os.path.join(cwd2, "dlib_face_recognition_resnet_model_v1.dat"))
        self.recognizer = dlib.face_recognition_model_v1(model_path2)
        
        
        self.face_encodings = []
        self.face_names = []
        self.load_faces()
        #self.face_names 由ъ뒪?듃?쓽 ?쁽?옱 湲몄씠?뿉 1?쓣 ?뜑?븯?뿬 ?떎?쓬?뿉 ?궗?슜?븷 ID瑜? ?꽕?젙.
        #由ъ뒪?듃?뿉?꽌 ?뼹援? ?씠由꾩씠 ?젣嫄? ?릺?뿀?쓣 ?븣, ?떎?쓬 ID瑜? ?삱諛붾Ⅴ寃? ?꽕?젙.
        self.next_id = len(self.face_names) + 1

    # ?씠?쟾?뿉 ????옣?맂 ?뼹援? ?씤肄붾뵫怨? ?씠由꾩쓣 ?뙆?씪?뿉?꽌 遺덈윭?샂.
    def load_faces(self):
        """Load previously saved face encodings and names."""
        if os.path.exists('face_data.pkl'):
            with open('face_data.pkl', 'rb') as f:
                self.face_encodings, self.face_names = pickle.load(f)
    # ?쁽?옱?쓽 ?뼹援? ?씤肄붾뵫怨? ?씠由꾩쓣 ?뙆?씪?뿉 ????옣.
    def save_faces(self):
        """Save face encodings and names to a file."""
        with open('face_data.pkl', 'wb') as f:
            pickle.dump((self.face_encodings, self.face_names), f)

    def learn_face(self,image):
        print("learn face function start")

        """Learn a new face from an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            print("No face detected for learning.")
            return

        for face in faces:
            shape = self.predictor(gray, face)
            encoding = np.array(self.recognizer.compute_face_descriptor(image, shape))
            
            self.face_encodings.append(encoding)
            self.face_names.append(f"User_{self.next_id}")# User_1, User_2 allocating automatically
            self.next_id += 1 
            self.save_faces()

            # 학습 중인 얼굴을 시각적으로 표시
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(image, f"Learning: User_{self.next_id - 1}", 
                    (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 이미지 창을 업데이트하고 2초 동안 대기
            cv2.imshow("Learning Face", image)
            cv2.waitKey(2000)  # 2초 동안 표시

        cv2.destroyWindow("Learning Face")  # 작업이 끝나면 창을 닫음

    def recognize_faces(self,frame = None):
        """Recognize faces from the webcam feed."""
        if frame is not None:
    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                shape = self.predictor(gray, face)
                encoding = np.array(self.recognizer.compute_face_descriptor(frame, shape))

                if len(self.face_encodings) > 0:
                    distances = np.linalg.norm(self.face_encodings - encoding, axis=1)
                    min_distance_index = np.argmin(distances)
                       
                    if distances[min_distance_index] < 0.6:
                        name = self.face_names[min_distance_index]
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                        cv2.imshow("Face Recognition", frame)
                        cv2.waitKey(2000)
                        print("OPEN!")
                        return
                        #exit()

                            #face_recognized = True
                    else:
                        print("You are thief!")
                        return
                        #exit(1)
                            # print("Unknown ?뼹援? 諛쒓껄. ?븰?뒿?븷源뚯슂? (y/n)")
                            # user_input = input()
                            # if user_input.lower() == 'y':
                                # self.learn_faces(face)
                        
                else:
                    name = "Can't recognize face"
                    print("Before register -> again eyetracking")
                    return
                    

                                
        cap = cv2.VideoCapture(0)
        #face_recognized = False

        while True:
            ret, frame = cap.read()#frame
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                shape = self.predictor(gray, face)
                encoding = np.array(self.recognizer.compute_face_descriptor(frame, shape))



                if len(self.face_encodings) > 0:
                    distances = np.linalg.norm(self.face_encodings - encoding, axis=1)
                    min_distance_index = np.argmin(distances)
                    
                    if distances[min_distance_index] < 0.6:
                        name = self.face_names[min_distance_index]
                        #face_recognized = True
                    else:
                        name = "Unknown"
                        
                else:
                    name = "Can't recognize face"

                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            

            cv2.imshow("Face Recognition", frame)

            '''
            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.learn_face(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''

        cap.release()
        cv2.destroyAllWindows()
