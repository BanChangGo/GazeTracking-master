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
        #self.face_names λ¦¬μ€?Έ? ??¬ κΈΈμ΄? 1? ???¬ ?€?? ?¬?©?  IDλ₯? ?€? .
        #λ¦¬μ€?Έ?? ?Όκ΅? ?΄λ¦μ΄ ? κ±? ??? ?, ?€? IDλ₯? ?¬λ°λ₯΄κ²? ?€? .
        self.next_id = len(self.face_names) + 1

    # ?΄? ? ????₯? ?Όκ΅? ?Έμ½λ©κ³? ?΄λ¦μ ??Ό?? λΆλ¬?΄.
    def load_faces(self):
        """Load previously saved face encodings and names."""
        if os.path.exists('face_data.pkl'):
            with open('face_data.pkl', 'rb') as f:
                self.face_encodings, self.face_names = pickle.load(f)
    # ??¬? ?Όκ΅? ?Έμ½λ©κ³? ?΄λ¦μ ??Ό? ????₯.
    def save_faces(self):
        """Save face encodings and names to a file."""
        with open('face_data.pkl', 'wb') as f:
            pickle.dump((self.face_encodings, self.face_names), f)

    def learn_face(self,image):

        """Learn a new face from an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            print("No face detected for learning.")
            return

        for face in faces:
            shape = self.predictor(gray, face)
            encoding = np.array(self.recognizer.compute_face_descriptor(image, shape))

            # ?΄λ―? ??΅? ?Όκ΅΄μΈμ§? ??Έ
            # if any(np.allclose(encoding, enc, atol=1e-4) for enc in self.face_encodings):
            #     print("?΄λ―? ????₯? ?Όκ΅? ?°?΄?°???€.")
            #     return
            # for i, enc in enumerate(self.face_encodings):
            #     if np.allclose(encoding, enc, atol=1e-4):
            #         print()
            
            self.face_encodings.append(encoding)
            self.face_names.append(f"User_{self.next_id}")# User_1, User_2 allocating automatically
            self.next_id += 1 
            self.save_faces()

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
                        #κ±°λ¦¬ κΈ°μ?? 0.6 -> 0.5 μ‘°μ .
                    if distances[min_distance_index] < 0.6:
                        name = self.face_names[min_distance_index]
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                        cv2.imshow("Face Recognition", frame)
                        cv2.waitKey(2000)
                        print("OPEN!")
                        exit()

                            #face_recognized = True
                    else:
                        print("You are thief!")
                        exit(1)
                            # print("Unknown ?Όκ΅? λ°κ²¬. ??΅? κΉμ? (y/n)")
                            # user_input = input()
                            # if user_input.lower() == 'y':
                                # self.learn_faces(face)
                else:
                    name = "Can't recognize face"

                    
                
                # if face_recognized:
                #     cv2.putText(frame, "Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #     cv2.imshow("Face Recognition", frame)
                #     cv2.waitKey(2000)  # 2μ΄? ?? λ³΄μ¬μ€?
                #     break
        cap = cv2.VideoCapture(0)
        #face_recognized = False

        while True:
            ret, frame = cap.read()#frame? λ°μ?¨?€.
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
                    #κ±°λ¦¬ κΈ°μ?? 0.6 -> 0.5 μ‘°μ .
                    if distances[min_distance_index] < 0.6:
                        name = self.face_names[min_distance_index]
                        #face_recognized = True
                    else:
                        name = "Unknown"
                        # print("Unknown ?Όκ΅? λ°κ²¬. ??΅? κΉμ? (y/n)")
                        # user_input = input()
                        # if user_input.lower() == 'y':
                            # self.learn_faces(face)
                else:
                    name = "Can't recognize face"

                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # if face_recognized:
            #     cv2.putText(frame, "Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #     cv2.imshow("Face Recognition", frame)
            #     cv2.waitKey(2000)  # 2μ΄? ?? λ³΄μ¬μ€?
            #     break

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.learn_face(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
