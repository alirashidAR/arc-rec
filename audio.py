import speech_recognition as sr
import os
from deepface import DeepFace
import cv2

# Directory where registered user images are stored
REGISTERED_USERS_DIR = "registered_users"

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to verify the user based on face
def verify_user_with_face():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Extract the face and save it temporarily
            captured_face_path = "captured_face.png"
            cv2.imwrite(captured_face_path, frame[y:y + h, x:x + w])

            try:
                result = DeepFace.find(img_path=captured_face_path, db_path=REGISTERED_USERS_DIR, enforce_detection=False)
                if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                    matched_user = os.path.basename(result[0].iloc[0]['identity']).split('.')[0]
                    print(f"User verified: {matched_user}")
                    
                    # User is verified, now let's check for "START" via audio
                    if verify_audio_command():
                        print("Audio Command: START detected. Access granted.")
                    else:
                        print("Audio Command: 'START' not detected. Try again.")
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

# Function to verify the "START" audio command
def verify_audio_command():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please say 'START' to proceed...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Capture audio

    try:
        # Recognize the speech
        command = recognizer.recognize_google(audio).lower()
        print(f"Detected speech: {command}")
        
        if "start" in command:
            return True  # Detected the word "START"
        else:
            return False  # Did not detect the word "START"

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
        return False
    except sr.RequestError as e:
        print(f"Error with the speech recognition service; {e}")
        return False

if __name__ == "__main__":
    verify_user_with_face()
