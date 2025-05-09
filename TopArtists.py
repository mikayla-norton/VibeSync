import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

# Set up the Spotify API client with your app credentials
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="bbbd08ee2e0c4aa4be3fd103e0c5b343",
                                               client_secret="fb7060c737bf4093aed4009d6d5e8a08",
                                               redirect_uri="http://localhost:8888/callback",
                                               scope=["user-library-read", "user-top-read", "playlist-read-private"]))

# Fetch user's top artists
def get_top_artists(limit=10):
    top_artists = sp.current_user_top_artists(limit=limit, time_range='long_term')  # Can be 'short_term', 'medium_term', or 'long_term'
    return pd.DataFrame([{
        'artist_name': artist['name'],
        'genres': artist['genres'],
        'popularity': artist['popularity'],
    } for artist in top_artists['items']])

# Example: Get user's top artists
user_top_artists = get_top_artists(limit=5)
print(user_top_artists)