from deepface import DeepFace
import cv2
import tkinter as tk
import os

# Create a directory to store registered user images if it doesn't exist
os.makedirs("registered_users", exist_ok=True)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_and_register():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Register Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            print("Pressed 'c'")  # Debugging line to check if the key is being captured
            if len(faces) == 0:
                print("No face detected. Try again.")
            else:
                user_name = input("Enter Name for Registration: ").strip()
                if user_name == "":
                    print("Error: Please enter a valid name.")
                else:
                    # Register each detected face with the same name
                    for (x, y, w, h) in faces:
                        face = frame[y:y + h, x:x + w]
                        file_path = os.path.join("registered_users", f"{user_name}_{x}_{y}.png")  # Save multiple faces with unique names
                        cv2.imwrite(file_path, face)
                        print(f"User {user_name} registered successfully at position ({x}, {y})!")

                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()

def verify_user():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Extract the face and save it
            captured_face_path = "captured_face.png"
            cv2.imwrite(captured_face_path, frame[y:y + h, x:x + w])

            try:
                result = DeepFace.find(img_path=captured_face_path, db_path="registered_users", enforce_detection=False)
                if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                    matched_user = os.path.basename(result[0].iloc[0]['identity']).split('.')[0]
                    print(f"User verified: {matched_user}")

                    # Display name on top of bounding box
                    cv2.putText(frame, matched_user, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                else:
                    print("Error: User not recognized.")
            except Exception as e:
                print(f"Error: {str(e)}")

            # Remove the captured face to clean up
            os.remove(captured_face_path)
            continue

        cv2.imshow("Verify User", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Menu
while True:
    print("\nFace Recognition System")
    print("1. Register User")
    print("2. Verify User")
    print("3. Exit")
    choice = input("Enter your choice: ").strip()

    if choice == '1':
        capture_and_register()
    elif choice == '2':
        verify_user()
    elif choice == '3':
        print("Exiting the system. Goodbye!")
        break
    else:
        print("Invalid choice. Please try again.")
