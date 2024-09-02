import os
import cv2
import dlib
import numpy as np

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

    def learn_face(self, name, image):
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
            self.face_names.append(name)
        print(f"Learned face: {name}")

    def recognize_faces(self):
        """Recognize faces from the webcam feed."""
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
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
                    else:
                        name = "Unknown"
                else:
                    name = "Can't recognize face"

                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                name = input("Enter name for the captured face: ")
                self.learn_face(name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
