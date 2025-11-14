import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set dataset paths
train_dir = r"C:\Users\shrav\Downloads\fer 2013\train"
val_dir = r"C:\Users\shrav\Downloads\fer 2013\test"

# Data Augmentation and Generators
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=32, class_mode='categorical')
val_generator = datagen.flow_from_directory(val_dir, target_size=(48, 48), batch_size=32, class_mode='categorical')

# Get the number of classes dynamically
num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} classes: {train_generator.class_indices}")

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Adjusted to match dataset classes
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=15)

# Save Model
model.save("model/model.h5")
print("Model training complete and saved as model.h5")

