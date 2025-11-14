import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load the trained model
model_path = r"C:\Users\shrav\Downloads\real time music\model\model.h5"

if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
else:
    print(f"❌q Error: Model file not found at {model_path}")
    exit()

# ✅ Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def predict_emotion(face_image):
    """
    Function to predict emotion from a face image.
    """
    # ✅ Ensure the image is grayscale and convert to RGB
    face_image = cv2.resize(face_image, (48, 48))  
    face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

    face_image = face_image / 255.0  # Normalize pixel values
    face_image = face_image.reshape(1, 48, 48, 3)  # Reshape for model input

    # ✅ Predict emotion
    predictions = model.predict(face_image)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]

# ✅ Start Video Captureq
cap = cv2.VideoCapture(0)  # Open the webcam

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("❌ Error: Failed to capture image.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Extract the face
        predicted_emotion = predict_emotion(face)  # Predict emotion
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)  # Display the frame

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()








