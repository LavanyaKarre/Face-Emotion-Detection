import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model("emotion_model.h5")

# Define validation dataset path
val_dir = "validation/test"  # Change this path as per your folder structure
img_size = (48, 48)

# Class labels (ensure they match training classes)
class_labels = sorted(os.listdir(val_dir))  # Automatically fetch folder names

# ImageDataGenerator for validation (rescaling images)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=1,  # Evaluate one image at a time
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False
)

# Evaluate model on validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"\nModel Accuracy on Validation Set: {val_accuracy * 100:.2f}%")

# Track actual vs predicted labels
true_labels = val_generator.classes
predicted_labels = []

# Predict each image in the validation dataset
for i in range(len(val_generator.filenames)):
    img_path = os.path.join(val_dir, val_generator.filenames[i])  # Full path
    img = image.load_img(img_path, target_size=img_size, color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for model input

    # Get model prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Get class index

    # Store predicted label
    predicted_labels.append(predicted_class)

    # Print result for each image
    actual_class = true_labels[i]
    print(f"File: {val_generator.filenames[i]} | Actual: {class_labels[actual_class]} | Predicted: {class_labels[predicted_class]}")

# Convert true and predicted labels to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Compute validation accuracy manually
correct_predictions = np.sum(true_labels == predicted_labels)
total_images = len(true_labels)
validation_accuracy = (correct_predictions / total_images) * 100

print(f"\nOverall Validation Accuracy: {validation_accuracy:.2f}%")