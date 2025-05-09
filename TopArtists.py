import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import streamlit as st

# Access the Spotify credentials securely from Streamlit secrets
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
redirect_uri = "https://mikaylanorton-vibesync.streamlit.app/"

# Set up the Spotify API client with the credentials from secrets
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
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