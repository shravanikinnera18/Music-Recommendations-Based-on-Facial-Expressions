import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Your Spotify API credentials
client_id = "faab16f59588492797cd960f2a080992"
client_secret = "d337497aded14f4cb5461da6184247f7"

# Set up Spotify client with client credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Emotion to genre mapping
emotion_song_map = {
    'Happy': 'pop',  # 'pop' genre for happiness
    'Sad': 'sad',    # 'sad' genre for sadness
    'Angry': 'rock', # 'rock' genre for anger
    'Neutral': 'chill', # 'chill' for neutral emotions
    'Surprise': 'party' # 'party' for surprise
}

def get_song_recommendations(emotion):
    """
    Fetch song recommendations based on the emotion provided.
    """
    # Get the genre from emotion
    genre = emotion_song_map.get(emotion, 'pop')

    try:
        # Search for tracks by genre
        results = sp.search(q=f'genre:{genre}', type='track', limit=5)
        
        print(results)

        # If no tracks are found, return a message
        if not results['tracks']['items']:
            return [{"error": "No songs found for the genre"}]

        # Return the song name and artist
        return [{'name': track['name'], 'artist': track['artists'][0]['name']} for track in results['tracks']['items']]
    
    except Exception as e:
        # Return an error message if the API call fails
        return [{"error": str(e)}]

# Test: Get recommendations for a sample emotion
if __name__ == "__main__":
    emotion = 'Happy'
    recommendations = get_song_recommendations(emotion)
    print(f"Recommendations for {emotion}: {recommendations}")
