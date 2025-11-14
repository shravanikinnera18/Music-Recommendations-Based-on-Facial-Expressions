import os
import cv2
import numpy as np
from flask import Flask, jsonify
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Get Spotify API credentials from environment variables
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# ✅ Initialize Spotify
sp = Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# ✅ Emotion-to-genre mapping
emotion_song_map = {
    'Happy': 'happy',
    'Sad': 'sad',
    'Angry': 'rock',
    'Neutral': 'chill',
    'Surprise': 'party'
}

# ✅ Load the trained emotion detection model
model = load_model(r"C:\Users\shrav\Downloads\real time music\model\model.h5")

# ✅ Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Real-Time Music Recommendation API!"})

@app.route("/detect-emotion", methods=["POST"])
def detect_emotion():
    try:
        # ✅ Capture real-time image from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({"error": "Failed to capture image"}), 500

        # ✅ Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ✅ Detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 400

        # ✅ Process first detected face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48)) / 255.0  # Normalize
        face_input = np.expand_dims(face_resized, axis=(0, -1))

        # ✅ Predict emotion
        emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        predictions = model.predict(face_input)
        detected_emotion = emotion_labels[np.argmax(predictions)]

        print(f"Detected Emotion: {detected_emotion}")  # Debugging

        # ✅ Get song recommendations based on detected emotion
        recommendations = get_song_recommendations(detected_emotion)

        return jsonify({
            "emotion": detected_emotion,
            "songs": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_song_recommendations(emotion):
    genre = emotion_song_map.get(emotion, 'pop')
    results = sp.search(q=f'genre:{genre}', type='track', limit=5)

    return [{'name': track['name'], 'artist': track['artists'][0]['name']} for track in results['tracks']['items']]

if __name__ == "__main__":
    app.run(debug=True, port=5001)







