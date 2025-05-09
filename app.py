import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlencode

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
    top_artists = sp.current_user_top_artists(limit=limit, time_range='long_term')  # Can be 'short_term', 'medium_term', or 'long_term'
    return pd.DataFrame([{
        'artist_name': artist['name'],
        'genres': artist['genres'],
        'popularity': artist['popularity'],
        'image_url': artist['images'][0]['url'] if artist['images'] else None  # Get the artist's image
    } for artist in top_artists['items']])

# Fetch artist's genre and image information
def get_artist_genre(artist_name):
    # Search for the artist by name, and limit the results to 1.
    result = sp.search(q=f"artist:{artist_name}", type='artist', limit=5)
    
    # Check if we have any artists in the results
    if result['artists']['items']:
        # Sort results by popularity or pick the first result
        artist = result['artists']['items'][0]
        
        # Get genres and images, check for missing data
        genres = artist['genres'] if 'genres' in artist else []
        image_url = artist['images'][0]['url'] if artist['images'] else None
        
        # If genres are missing or empty, print a warning
        if not genres:
            print(f"Warning: No genres available for artist '{artist_name}'.")
            genres = ["Genres data not available"]  # Default message if no genres
        
        # If image URL is missing, print a warning
        if not image_url:
            print(f"Warning: No image available for artist '{artist_name}'.")
        
        return genres, image_url  # Return genres and image URL
    else:
        print(f"Warning: No artist found for '{artist_name}'.")
        return [], None  # If no artist found, return empty genres and no image




# Build user profile based on genres of top artists
def build_user_profile(user_top_artists):
    tfidf = TfidfVectorizer(stop_words='english')
    genres_list = [' '.join(artist['genres']) for artist in user_top_artists]
    genre_matrix = tfidf.fit_transform(genres_list)
    user_profile = np.mean(genre_matrix.toarray(), axis=0)
    return user_profile, tfidf

# Calculate similarity between user profile and artist genres
def calculate_compatibility(user_profile, artist_name, tfidf):
    artist_genres, _ = get_artist_genre(artist_name)
    artist_vector = tfidf.transform([' '.join(artist_genres)]).toarray()
    similarity_score = cosine_similarity(user_profile.reshape(1, -1), artist_vector)[0][0]
    return similarity_score

# Streamlit app layout and flow
def main():
    # App title and description
    st.title("Spotify Artist Compatibility Analysis")
    st.markdown("## Welcome to the Artist Compatibility Analyzer!")
    st.markdown("Log in to your Spotify account and analyze your top artists or check compatibility with new ones!")

    # Sidebar styling and flow
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an option", ("Login and View Top Artists", "Artist Compatibility Analysis"))

    # First button: OAuth and display top 10 artists
    if choice == "Login and View Top Artists":
        if st.button("Login with Spotify"):
            st.write("Logging in...")

            # URL for Spotify OAuth authorization request
            auth_url = sp.auth_manager.get_authorize_url()
            st.markdown(f"[Click here to login with Spotify]({auth_url})")

            st.write("After logging in, please return to the app.")

            # Display a message to let users know they're in the login process
            st.write("You will be redirected to Spotify for authentication. Once authenticated, you will be able to see your top artists.")

            user_top_artists = get_top_artists(limit=30)
            st.write("### Your Top 10 Artists:")
            st.write(user_top_artists[['artist_name', 'genres']].head(10))
            
            # Display artist images
            for index, row in user_top_artists.iterrows():
                st.write(f"**{row['artist_name']}**")
                st.write(f"Genres: {', '.join(row['genres'])}")
                if row['image_url']:
                    st.image(row['image_url'], width=100)
                st.markdown("---")

    # Second button: input artist names and analyze compatibility
    if choice == "Artist Compatibility Analysis":
        st.markdown("### Enter Artist Names")
        artist_input = st.text_area("Enter a list of artist names (comma-separated):")
        if st.button("Analyze Compatibility"):
            if artist_input:
                artist_names = [artist.strip() for artist in artist_input.split(",")]
                user_top_artists = get_top_artists(limit=10)
                user_profile, tfidf = build_user_profile(user_top_artists.to_dict(orient='records'))

                for artist_name in artist_names:
                    st.write(f"\n### Results for {artist_name}:")
                    
                    # Calculate compatibility score
                    compatibility_score = calculate_compatibility(user_profile, artist_name, tfidf)
                    st.write(f"Compatibility score: {compatibility_score:.2f}")

                    # Display genres and artist image
                    genres, image_url = get_artist_genre(artist_name)
                    st.write(f"Genres: {', '.join(genres)}")
                    if image_url:
                        st.image(image_url, width=100)

            else:
                st.error("Please enter at least one artist name.")

# Run the app
if __name__ == "__main__":
    main()
