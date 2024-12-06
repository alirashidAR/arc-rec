import cv2
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

class FaceAuthSystem:
    def __init__(self, data_dir="face_data"):
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "users.json")

        self.model_path = "deploy.prototxt"
        self.weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.face_net = cv2.dnn.readNetFromCaffe(self.model_path, self.weights_path)
        
        self.embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")

        os.makedirs(data_dir, exist_ok=True)
        self.users = {}
        self.load_data()

    def load_data(self):
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
        except Exception as e:
            print(f"Error loading data: {e}")
            self.users = {}

    def save_data(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)

    def detect_face(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                     mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x_max, y_max) = box.astype("int")
                faces.append((x, y, x_max - x, y_max - y))
        return faces

    def get_face_embedding(self, face_img):
        face_blob = cv2.dnn.blobFromImage(face_img, scalefactor=1.0/255, size=(96, 96),
                                          mean=(0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(face_blob)
        return self.embedder.forward()

    def register_user(self, username):
        if username in self.users:
            print(f"User {username} already exists!")
            return False

        print("Starting camera for face capture...")

        cap = cv2.VideoCapture(0)
        embeddings = []
        captured = 0

        while captured < 5:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            faces = self.detect_face(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Register User - Press SPACE to capture', frame)

            key = cv2.waitKey(1)
            if key == 32:  # SPACE key
                if len(faces) != 1:
                    print("Please ensure exactly one face is visible")
                    continue
                
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                embedding = self.get_face_embedding(face_img)
                embeddings.append(embedding.flatten())
                captured += 1
                print(f"Captured {captured}/5 images")
            
            elif key == 27:  # ESC key
                print("Registration cancelled")
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured < 5:
            print("Not enough images captured. Registration failed!")
            return False

        average_embedding = np.mean(embeddings, axis=0)
        self.users[username] = {
            "registered_date": datetime.now().isoformat(),
            "last_login": None,
            "embedding": average_embedding.tolist()
        }
        self.save_data()
        print(f"User {username} registered successfully!")

    def authenticate(self):
        if not self.users:
            print("No users registered yet!")
            return

        print("Starting camera for authentication...")
        cap = cv2.VideoCapture(0)

        while True:
            embeddings = []
            captured = 0

            while captured < 5:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                faces = self.detect_face(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_img = frame[y:y+h, x:x+w]
                    embedding = self.get_face_embedding(face_img)
                    embeddings.append(embedding.flatten())
                    captured += 1
                    print(f"Captured {captured}/5 frames")

                cv2.imshow('Authenticating...', frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    print("Authentication cancelled")
                    break

            if captured < 5:
                print("Not enough frames captured. Authentication failed!")
                cap.release()
                cv2.destroyAllWindows()
                return

            # Average embedding of captured frames
            average_embedding = np.mean(embeddings, axis=0)
            best_match = None
            highest_similarity = -1

            for username, user_data in self.users.items():
                user_embedding = np.array(user_data["embedding"])
                similarity = cosine_similarity([average_embedding], [user_embedding])[0][0]
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = username

            if highest_similarity > 0.80:
                print(f"Authentication successful! Welcome, {best_match}")
                self.users[best_match]["last_login"] = datetime.now().isoformat()
                self.save_data()
                cap.release()
                cv2.destroyAllWindows()
                break  # Exit the loop after successful authentication
            else:
                print("Authentication failed! Retrying...")
                cv2.imshow("Retrying Authentication", frame)
                cv2.waitKey(1)  # Allow frame to display briefly before retrying



def main():
    auth_system = FaceAuthSystem()

    while True:
        print("\nFace Authentication System")
        print("1. Register new user")
        print("2. Authenticate user")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            username = input("Enter username to register: ")
            auth_system.register_user(username)
        elif choice == "2":
            auth_system.authenticate()
        elif choice == "3":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
