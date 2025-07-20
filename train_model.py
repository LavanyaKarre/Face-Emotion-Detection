import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess the dataset
train_data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = train_data_gen.flow_from_directory(
    "dataset/", 
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

val_generator = train_data_gen.flow_from_directory(
    "dataset/", 
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# Build CNN model
model = Sequential([
    Conv2D(64, (3,3), activation="relu", input_shape=(48, 48, 1)),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")  # 7 classes for emotions
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=50)

# Save model
model.save("emotion_model.h5")
print("Model saved successfully as emotion_model.h5")
