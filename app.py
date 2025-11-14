import os
import cv2
import numpy as np
from flask import Flask, jsonify
from tensorflow.keras.models import load_model
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# ✅ Initialize Spotipy
sp = Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# ✅ Emotion-to-genre mapping
emotion_song_map = {
    'Happy': 'happy',
    'Sad': 'sad',
    'Angry': 'rock',
    'Neutral': 'chill',
    'Surprise': 'party'
}

# ✅ Load emotion detection model
model_path = r"C:\Users\shrav\Downloads\real time music\model\model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Error: Model file not found at {model_path}")
    exit()

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ✅ Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Real-Time Music Recommendation API!"})

@app.route("/detect-emotion", methods=["POST"])
def detect_emotion():
    """
    Captures image from webcam, detects emotion, and recommends songs.
    """
    try:
        # ✅ Capture real-time image from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({"error": "Failed to capture image"}), 500

        # ✅ Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 400

        # ✅ Process the first detected face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))  # Resize to match model input

        # ✅ Convert to RGB, normalize & reshape
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB) / 255.0
        face_input = np.expand_dims(face_rgb, axis=0)

        # ✅ Predict emotion
        predictions = model.predict(face_input)
        detected_emotion = emotion_labels[np.argmax(predictions)]

        # ✅ Get song recommendations
        recommendations = get_song_recommendations(detected_emotion)

        return jsonify({
            "emotion": detected_emotion,
            "songs": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_song_recommendations(emotion):
    """
    Fetch song recommendations based on detected emotion.
    """
    genre = emotion_song_map.get(emotion, 'pop')
    results = sp.search(q=f'genre:{genre}', type='track', limit=5)

    return [{'name': track['name'], 'artist': track['artists'][0]['name']} for track in results['tracks']['items']]

if __name__ == "__main__":
    app.run(debug=True, port=5001)
