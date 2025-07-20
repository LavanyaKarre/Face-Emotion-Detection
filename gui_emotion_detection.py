import cv2
import tkinter as tk
from tkinter import simpledialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import threading

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load emotion recognition model
model = load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Create main window
root = tk.Tk()
root.title("Facial Expression Detection")
root.geometry("800x800")
root.configure(bg="#f0f0f0")  # Light gray background

# Global variables
user_name = ""
stop_detection = False  # Flag to stop detection

# Function to get the user's name
def get_name():
    global user_name
    user_name = simpledialog.askstring("Enter Name", "What is your name?")
    if user_name:
        welcome_label.config(text=f"Hello, {user_name}! Click 'Check Emotion' to analyze your facial expression.")

# Function to check facial expression
def check_emotion():
    global stop_detection
    stop_detection = False
    cap = cv2.VideoCapture(0)

    while not stop_detection:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0  # Normalize

            prediction = model.predict(roi_gray)
            emotion = emotion_labels[np.argmax(prediction)]

            # Set different colors based on detected emotion
            color_map = {
                "Happy": (0, 255, 0),  # Green
                "Sad": (255, 0, 0),  # Blue
                "Angry": (0, 0, 255),  # Red
                "Neutral": (128, 128, 128),  # Gray
                "Fear": (255, 255, 0),  # Cyan
                "Surprise": (255, 165, 0),  # Orange
                "Disgust": (128, 0, 128)  # Purple
            }
            color = color_map.get(emotion, (255, 255, 255))  # Default White

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display emotion text
            emotion_text = f"{user_name} is feeling {emotion}" if user_name else f"Detected Emotion: {emotion}"
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Facial Emotion Detection", frame)

        # Press 'q' to exit detection manually
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to stop emotion detection
def stop_emotion_detection():
    global stop_detection
    stop_detection = True  # Set flag to stop detection

# UI Elements
title_label = Label(root, text="Facial Expression Detection", font=("Arial", 20, "bold"), bg="#4CAF50", fg="white", padx=10, pady=10)
title_label.pack(fill="both")

welcome_label = Label(root, text="Welcome to Facial Expression Detection! Enter your name to begin.", font=("Arial", 12), bg="#f0f0f0")
welcome_label.pack(pady=20)

name_button = Button(root, text="Enter Name", font=("Arial", 14), bg="#2196F3", fg="white", padx=10, pady=5, command=get_name)
name_button.pack(pady=10)

emotion_button = Button(root, text="Check Emotion", font=("Arial", 14), bg="#FF9800", fg="white", padx=10, pady=5, command=lambda: threading.Thread(target=check_emotion).start())
emotion_button.pack(pady=10)

exit_button = Button(root, text="Stop Detection", font=("Arial", 14), bg="#F44336", fg="white", padx=10, pady=5, command=stop_emotion_detection)
exit_button.pack(pady=10)

# Run the application
root.mainloop()
