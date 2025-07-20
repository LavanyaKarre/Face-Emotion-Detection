import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import os

# Load the trained emotion model
model = load_model("emotion_model.h5")

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion labels (Adjust based on model training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to process all images in a folder
def process_folder():
    folder_path = filedialog.askdirectory()  # Select folder
    
    if not folder_path:
        return

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)

        # Read image
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray_image[y:y+h, x:x+w]  # Extract face region
            face = cv2.resize(face, (48, 48))  # Resize to model input size
            face = np.expand_dims(face, axis=-1)  # Add channel dimension
            face = np.expand_dims(face, axis=0)  # Add batch dimension
            face = face / 255.0  # Normalize pixel values

            # Predict emotion
            prediction = model.predict(face)
            emotion_index = np.argmax(prediction)  # Get the predicted label index
            emotion = emotion_labels[emotion_index]  # Get the emotion name
            accuracy = np.max(prediction) * 100  # Get confidence score

            # Print results in terminal
            print(f"Image: {file_name} -> Emotion: {emotion} (Confidence: {accuracy:.2f}%)")

            # Draw bounding box around the face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Centering text inside the bounding box
            text = f"{emotion} ({accuracy:.2f}%)"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x + (w - text_size[0]) // 2  # Center the text
            text_y = y + h + 30  # Position below the face, adjust as needed

            # Draw text background for better visibility
            cv2.rectangle(image, (text_x - 5, text_y - 25), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Resize for better visibility
        large_image = cv2.resize(image, (800, 600))  # Resize to 800x600

        # Display the image with detected emotion
        cv2.imshow("Emotion Detection", large_image)
        cv2.waitKey(3000)  # Display each image for 3 seconds
        cv2.destroyAllWindows()

# GUI Setup
root = tk.Tk()
root.title("Face Emotion Detection - Folder Processing")

# UI Elements
btn_select_folder = tk.Button(root, text="Select Folder & Process Images", command=process_folder, font=("Arial", 12))
btn_select_folder.pack(pady=20)

root.mainloop()