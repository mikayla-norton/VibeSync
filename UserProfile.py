import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Access the Spotify credentials securely from Streamlit secrets
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
redirect_uri = "https://mikaylanorton-vibesync.streamlit.app/callback"

# Set up the Spotify API client with the credentials from secrets
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=["user-library-read", "user-top-read", "playlist-read-private"]))


# Fetch user's top artists
def get_top_artists(limit=10):
    top_artists = sp.current_user_top_artists(limit=limit, time_range='medium_term')  # Can be 'short_term', 'medium_term', or 'long_term'
    return pd.DataFrame([{
        'artist_name': artist['name'],
        'genres': artist['genres'],
        'popularity': artist['popularity'],
    } for artist in top_artists['items']])

# Fetch artist's genre information
def get_artist_genre(artist_name):
    result = sp.search(q=artist_name, type='artist', limit=1)
    artist = result['artists']['items'][0]
    return artist['genres']  # Return the list of genres associated with the artist

# Build user profile based on genres of top artists
def build_user_profile(user_top_artists):
    tfidf = TfidfVectorizer(stop_words='english')
    genres_list = [' '.join(artist['genres']) for artist in user_top_artists]
    genre_matrix = tfidf.fit_transform(genres_list)
    user_profile = np.mean(genre_matrix.toarray(), axis=0)
    return user_profile, tfidf

# Calculate similarity between user profile and artist genres
def calculate_compatibility(user_profile, artist_name, tfidf):
    artist_genres = get_artist_genre(artist_name)
    artist_vector = tfidf.transform([' '.join(artist_genres)]).toarray()
    similarity_score = cosine_similarity(user_profile.reshape(1, -1), artist_vector)[0][0]
    return similarity_score

# Main function to execute the app
def main():
    # Fetch user's top artists and build user profile
    user_top_artists = get_top_artists(limit=10)
    print("User's Top 10 Artists:")
    print(user_top_artists[['artist_name', 'genres']].head(10))

    user_profile, tfidf = build_user_profile(user_top_artists.to_dict(orient='records'))
    
    # Input artist names (comma-separated list)
    artist_names = input("Enter a list of artist names (comma-separated): ").split(',')
    artist_names = [artist.strip() for artist in artist_names]
    
    for artist_name in artist_names:
        print(f"\nResults for {artist_name}:")
        
        # Calculate compatibility score
        compatibility_score = calculate_compatibility(user_profile, artist_name, tfidf)
        print(f"Compatibility score: {compatibility_score:.2f}")
                
        # Print associated genres
        genres = get_artist_genre(artist_name)
        print(f"Genres: {', '.join(genres)}")

if __name__ == "__main__":
    main()
