import os
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Load environment variables from .env file
load_dotenv()

# Get Spotify API credentials
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# Initialize Spotify
if SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
    sp = Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET))
    print("✅ Spotify initialized successfully!")
else:
    sp = None
    print("⚠️ Warning: Spotify credentials not found. Song recommendations will be disabled.")

# Emotion to genre mapping
emotion_song_map = {
    'Happy': 'happy',
    'Sad': 'sad',
    'Angry': 'rock',
    'Disgust': 'rock',
    'Fear': 'ambient',
    'Neutral': 'chill',
    'Surprise': 'party'
}

# Emotion labels based on model output
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load emotion detection model
model_path = os.path.join(os.getcwd(), "model", "model.h5")
if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Emotion detection model loaded successfully!")
else:
    print(f"❌ Error: Model file not found at {model_path}")
    exit()

# Load Haar Cascade for face detection
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(haar_cascade_path):
    print(f"❌ Error: Haar cascade file not found at {haar_cascade_path}")
    exit()
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
print("✅ Face detector initialized successfully!")

# Initialize Flask app
app = Flask(__name__)

# Serve index.html
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Helper: Predict emotion
def predict_emotion(face_image_gray):
    try:
        face_resized = cv2.resize(face_image_gray, (48, 48))

        if model.input_shape[-1] == 3:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            face_normalized = face_rgb / 255.0
            face_input = face_normalized.reshape(1, 48, 48, 3)
        else:
            face_normalized = face_resized / 255.0
            face_input = face_normalized.reshape(1, 48, 48, 1)

        predictions = model.predict(face_input)
        emotion_index = np.argmax(predictions[0])
        return emotion_labels[emotion_index]
    except Exception as e:
        print(f"❌ Error during emotion prediction: {e}")
        return None

# Endpoint: Scan emotion
@app.route("/scan-emotion", methods=["GET"])
def scan_emotion():
    cap = None
    face_gray = None
    detected_emotion = None
    recommendations = []
    MAX_ATTEMPTS = 10
    DELAY_BETWEEN_ATTEMPTS = 0.1

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return jsonify({"error": "Could not open webcam"}), 500

        print("Attempting to capture and detect face...")
        for attempt in range(MAX_ATTEMPTS):
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Attempt {attempt + 1}: Failed to capture image.")
                time.sleep(DELAY_BETWEEN_ATTEMPTS)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

            if len(faces) > 0:
                print(f"✅ Face detected on attempt {attempt + 1}!")
                x, y, w, h = faces[0]
                face_gray = gray[y:y + h, x:x + w]
                break
            else:
                print(f"Attempt {attempt + 1}: No face detected.")
                time.sleep(DELAY_BETWEEN_ATTEMPTS)

        if face_gray is not None:
            detected_emotion = predict_emotion(face_gray)
            if detected_emotion:
                recommendations = get_song_recommendations(detected_emotion)
            else:
                return jsonify({"error": "Face detected, but emotion prediction failed"}), 500
        else:
            return jsonify({"message": "No face detected after multiple attempts"}), 200

        return jsonify({
            "emotion": detected_emotion,
            "songs": recommendations
        })

    except Exception as e:
        print(f"❌ Error in /scan-emotion endpoint: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500
    finally:
        if cap and cap.isOpened():
            cap.release()
            print("✅ Webcam released.")

# Spotify recommendations
def get_song_recommendations(emotion):
    if not sp:
        return []
    genre = emotion_song_map.get(emotion, 'pop')
    try:
        results = sp.search(q=f'genre:{genre}', type='track', limit=5)
        return [{'name': track['name'], 'artist': track['artists'][0]['name']}
                for track in results['tracks']['items']]
    except Exception as e:
        print(f"❌ Error fetching Spotify recommendations: {e}") 
        return []

# Run Flask server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
