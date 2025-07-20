# Facial Emotion Detection using Deep Learning 🎭

This project detects facial emotions such as **happy, sad, angry, surprise, neutral, etc.**, from images or videos using a Convolutional Neural Network (CNN). It uses TensorFlow/Keras, OpenCV, and other supporting libraries for emotion classification.

---

## 🔧 Requirements

Ensure you have Python **3.8 to 3.10** (recommended). Avoid Python 3.13 as some libraries may not support it yet.

### 📦 Install Dependencies

```bash
pip install tensorflow keras opencv-python numpy matplotlib pillow scikit-learn
📁 Project Structure
graphql
Copy
Edit
facial_emotion_detection/
│
├── train_model.py             # Train CNN model on FER2013 dataset
├── model.h5                   # Trained model file (if available)
├── detect_emotion.py          # Real-time or image-based emotion detection
├── emotion_utils.py           # Utility functions (preprocessing, predictions)
├── haarcascade_frontalface.xml  # OpenCV's face detector XML
├── requirements.txt           # All dependencies
├── README.md                  # This file
🚀 How to Run
1️⃣ Train the Model (Optional - Pretrained model can be used)
bash
Copy
Edit
python train_model.py
This will generate a model.h5 file containing the trained CNN model.

2️⃣ Run Emotion Detection
bash
Copy
Edit
python detect_emotion.py
Choose one of the following modes inside the script:

📷 Detect emotion from an image

🎥 Detect emotion in real-time using your webcam

📚 Dataset Used
This project is typically trained on the FER-2013 dataset:

Available via Kaggle - FER2013 Facial Expression Dataset

Place the dataset CSV in the same directory or update train_model.py accordingly.

🧠 Model Architecture
Convolutional layers

MaxPooling

Dropout

Flatten

Dense layers with softmax activation

📊 Accuracy
Achieved ~65-75% accuracy on validation set (depends on dataset split and model architecture).

🛠️ Built With
Python 🐍

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Scikit-learn

Pillow# Face-Emotion-Detection
