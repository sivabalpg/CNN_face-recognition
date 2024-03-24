import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to load images and labels
def load_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (128, 128))  # Resize to a consistent size
        label = int(os.path.split(image_path)[-1].split(".")[1])
        
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images_and_labels('dataset')

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preprocess the data
X_train = X_train.reshape(-1, 128, 128, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 128, 128, 1).astype('float32') / 255.0

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('face_recognition_model.h5')
