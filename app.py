import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlencode
import warnings

warnings.filterwarnings("ignore")
st.set_option('logger.level', 'error')

# Access the Spotify credentials securely from Streamlit secrets
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
redirect_uri = "https://mikaylanorton-vibesync.streamlit.app/callback"

# SpotifyOAuth setup
sp_oauth = SpotifyOAuth(client_id=client_id,
                         client_secret=client_secret,
                         redirect_uri=redirect_uri,
                         scope=["user-library-read", "user-top-read", "playlist-read-private"])

# Fetch user's top artists
def get_top_artists():
    # Ensure authentication is handled
    token_info = sp_oauth.get_access_token(st.experimental_get_query_params().get("code", [None])[0])
    if token_info:
        sp = spotipy.Spotify(auth=token_info["access_token"])
        top_artists = sp.current_user_top_artists(limit=10, time_range='long_term')
        return pd.DataFrame([{
            'artist_name': artist['name'],
            'genres': artist['genres'],
            'popularity': artist['popularity'],
            'image_url': artist['images'][0]['url'] if artist['images'] else None
        } for artist in top_artists['items']])
    else:
        return None


# Fetch artist's genre and image information
def get_artist_genre(artist_name):
    result = sp.search(q=f"artist:{artist_name}", type='artist', limit=5)
    if result['artists']['items']:
        artist = result['artists']['items'][0]
        genres = artist['genres'] if 'genres' in artist else []
        image_url = artist['images'][0]['url'] if artist['images'] else None
        return genres, image_url
    else:
        return [], None


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
    st.title("Spotify Artist Compatibility Analysis")
    st.markdown("## Welcome to the Artist Compatibility Analyzer!")
    st.markdown("Log in to your Spotify account and analyze your top artists or check compatibility with new ones!")

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an option", ("Login and View Top Artists", "Artist Compatibility Analysis"))

    # First button: OAuth and display top 10 artists
    if choice == "Login and View Top Artists":
        # If we're already authenticated
        query_params = st.experimental_get_query_params()
        if "code" in query_params:
            st.write("Successfully logged in!")

            user_top_artists = get_top_artists()
            if user_top_artists is not None:
                st.write("### Your Top 10 Artists:")
                st.write(user_top_artists[['artist_name', 'genres']])

                # Display artist images
                for index, row in user_top_artists.iterrows():
                    st.write(f"**{row['artist_name']}**")
                    st.write(f"Genres: {', '.join(row['genres'])}")
                    if row['image_url']:
                        st.image(row['image_url'], width=100)
                    st.markdown("---")
            else:
                st.error("There was an issue with your authentication. Please try again.")

        else:
            # Show login button
            auth_url = sp_oauth.get_authorize_url()
            st.markdown(f"[Click here to login with Spotify]({auth_url})")
            st.write("After logging in, please return to the app.")

    # Second button: input artist names and analyze compatibility
    if choice == "Artist Compatibility Analysis":
        artist_input = st.text_area("Enter a list of artist names (comma-separated):")
        if st.button("Analyze Compatibility"):
            if artist_input:
                artist_names = [artist.strip() for artist in artist_input.split(",")]
                user_top_artists = get_top_artists()
                user_profile, tfidf = build_user_profile(user_top_artists.to_dict(orient='records'))

                for artist_name in artist_names:
                    st.write(f"\n### Results for {artist_name}:")
                    
                    compatibility_score = calculate_compatibility(user_profile, artist_name, tfidf)
                    st.write(f"Compatibility score: {compatibility_score:.2f}")

                    genres, image_url = get_artist_genre(artist_name)
                    st.write(f"Genres: {', '.join(genres)}")
                    if image_url:
                        st.image(image_url, width=100)

            else:
                st.error("Please enter at least one artist name.")


if __name__ == "__main__":
    main()
