# Music-Recommendations-Based-on-Facial-Expressions
       A deep learning–based system that detects human emotions from facial expressions and recommends songs using the Spotify API.

📌 Overview

This project uses a CNN model trained on the FER-2013 dataset to detect emotions like:
Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
Detected emotions are then mapped to music recommendations using the Spotify Developer API.

📝 Project Steps
🔹 STEP 1 — Dataset (FER-2013)

Total images: 35,887

Training dataset: 28,709

Validation/Testing dataset: 7,178

Image resolution: 48×48 pixels, grayscale

Emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

🔹 STEP 2 — Training the Model (train.py)

Loaded training and testing datasets

Used a CNN architecture

Trained for 15 epochs

Saved the trained model as model.h5

🔹 STEP 3 — Real-Time Emotion Detection (emotion.py)

Loaded the trained model model.h5

Used OpenCV’s haarcascade_frontalface_default.xml for face detection

Performs real-time emotion prediction (happy, sad, angry, etc.)

🔹 STEP 4 — Spotify API Integration (spotify.py)

Used Spotify for Developers (Client ID & Secret)

Maps detected emotions to music categories

Fetches song recommendations based on detected emotion

Extracts song details such as track name, artist, preview URL, and album cover

🔧 Technologies Used

Python

TensorFlow / Keras

OpenCV

CNN (Convolutional Neural Network)

Spotify API

FER-2013 Dataset

▶️ How to Run

Clone the repository

Install dependencies

Run emotion.py

Allow webcam access for real-time emotion detection

Spotify recommendations will be displayed based on detected emotion

📂 Repository Structure (Recommended)
/music-recommendation
│── train.py
│── emotion.py
│── spotify.py
│── model.h5
│── haarcascade_frontalface_default.xml
│── README.md
│── requirements.txt

🚀 Future Enhancements

Deploy using Flask / Streamlit

Add more emotion categories

Add UI for user interaction

Improve recommendation accuracy

👤 Author

Shravani Kinnera
