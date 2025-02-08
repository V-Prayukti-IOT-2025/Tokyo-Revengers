import cv2 as cv
import numpy as np
import face_recognition as fr
import pyttsx3
import pytesseract
import speech_recognition as sr

# Set Tesseract-OCR Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Text-to-Speech Engine
n = pyttsx3.init('sapi5')
n.setProperty('voice', n.getProperty('voices')[0].id)
n.setProperty('rate', 150)

def speak(audio):
    """Speak the given text using TTS"""
    n.say(audio)
    n.runAndWait()

def takeCommand():
    """Take voice input from the user"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}")
        return query.lower()
    except Exception as e:
        print("Could not recognize speech:", e)
        speak("Sorry, I didn't understand that.")
        return None

def facedetection():
    """Live face detection and recognition"""
    cap = cv.VideoCapture(0)
    announced_names = set()  # Store names to avoid repeated announcements

    try:
        # Load known faces
        nans_img = fr.load_image_file("Narayanan.jpg")
        nans_face_encoding = fr.face_encodings(nans_img)[0]

        tony_img = fr.load_image_file("tony.jpg")
        tony_face_encoding = fr.face_encodings(tony_img)[0]

        known_face_encodings = [nans_face_encoding, tony_face_encoding]
        known_face_names = ["Nans", "Tony"]
    except Exception as e:
        print("Error loading face encodings:", e)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            if name not in announced_names:
                speak(f"{name} is right before you")
                announced_names.add(name)

            # Draw rectangle around face
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv.imshow("Face Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to quit
            break

    cap.release()
    cv.destroyAllWindows()

def object_detection():
    """Live object detection using MobileNet SSD"""
    cap = cv.VideoCapture(0)
    announced_objects = set()  # Store objects to avoid repeated announcements

    # Load pre-trained MobileNet SSD model and class labels
    proto = "deploy.prototxt"
    model = "mobilenet_iter_73000.caffemodel"
    net = cv.dnn.readNetFromCaffe(proto, model)

    class_names = {
    0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
    15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train',
    20: 'tvmonitor'}
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                class_id = int(detections[0, 0, i, 1])
                if class_id in class_names:
                    label = class_names[class_id]

                    if label not in announced_objects:
                        speak(f"{label} detected")
                        announced_objects.add(label)

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv.putText(frame, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv.imshow("Object Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to quit
            break

    cap.release()
    cv.destroyAllWindows()

def live_text_recognition():
    """Live text detection from camera"""
    cap = cv.VideoCapture(0)
    announced_texts = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()

        if text and text not in announced_texts:
            print("Extracted Text:", text)
            speak("Text detected: " + text)
            announced_texts.add(text)

        cv.putText(frame, text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.imshow("Live Text Recognition", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to quit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    speak("Hello! Say 'face detection', 'object detection', or 'text recognition' to start.")

    while True:
        command = takeCommand()

        if command:
            if "face detection" in command or "face" in command:
                speak("Starting face detection...")
                facedetection()

            elif "object detection" in command or "object" in command:
                speak("Starting object detection...")
                object_detection()

            elif "text recognition" in command or "text" in command:
                speak("Starting text recognition...")
                live_text_recognition()

            elif "exit" in command or "stop" in command:
                speak("Goodbye!")
                break
