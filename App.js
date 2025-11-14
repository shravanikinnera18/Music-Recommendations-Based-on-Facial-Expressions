import React, { useState } from 'react';

function App() {
  const [emotion, setEmotion] = useState('');
  const [songs, setSongs] = useState([]);

  const detectEmotion = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/detect-emotion', {
        method: 'POST',
      });
      const data = await response.json();
      if (response.ok) {
        setEmotion(data.emotion);
        setSongs(data.songs);
      } else {
        alert(data.error || 'Something went wrong!');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('There was an error detecting emotion');
    }
  };

  return (
    <div className="App">
      <h1>Emotion-Based Music Recommendation</h1>
      <button onClick={detectEmotion}>Detect Emotion</button>
      <div>
        <h2>Detected Emotion: {emotion}</h2>
        <ul>
          {songs.map((song, index) => (
            <li key={index}>
              <strong>{song.name}</strong> by {song.artist}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;

