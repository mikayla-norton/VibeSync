import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlencode

# Access the Spotify credentials securely from Streamlit secrets
client_id     = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
redirect_uri  = "https://mikaylanorton-vibesync.streamlit.app/callback"

# SpotifyOAuth setup
sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=["user-library-read","user-top-read","playlist-read-private"]
)

# Fetch user's top artists
def get_top_artists():
    code = st.query_params.get("code", [None])[0]
    token_info = sp_oauth.get_access_token(code)
    if token_info:
        sp = spotipy.Spotify(auth=token_info["access_token"])
        top_artists = sp.current_user_top_artists(limit=10, time_range='long_term')
        return pd.DataFrame([{
            'artist_name': artist['name'],
            'genres'     : artist['genres'],
            'popularity' : artist['popularity'],
            'image_url'  : artist['images'][0]['url'] if artist['images'] else None
        } for artist in top_artists['items']])
    return None

# Fetch artist's genre and image information
def get_artist_genre(artist_name):
    code = st.query_params.get("code", [None])[0]
    token_info = sp_oauth.get_access_token(code)
    if token_info:
        sp = spotipy.Spotify(auth=token_info["access_token"])
        result = sp.search(q=f"artist:{artist_name}", type='artist', limit=5)
        if result['artists']['items']:
            artist    = result['artists']['items'][0]
            genres    = artist.get('genres', [])
            image_url = artist['images'][0]['url'] if artist['images'] else None
            return genres, image_url
    return [], None

# Build user profile based on genres of top artists
def build_user_profile(user_top_artists):
    tfidf       = TfidfVectorizer(stop_words='english')
    genres_list = [' '.join(a['genres']) for a in user_top_artists]
    matrix      = tfidf.fit_transform(genres_list)
    profile     = np.mean(matrix.toarray(), axis=0)
    return profile, tfidf

# Calculate similarity between user profile and artist genres
def calculate_compatibility(user_profile, artist_name, tfidf):
    artist_genres, _ = get_artist_genre(artist_name)
    artist_vec       = tfidf.transform([' '.join(artist_genres)]).toarray()
    score            = cosine_similarity(user_profile.reshape(1,-1), artist_vec)[0][0]
    return score

# Streamlit app layout and flow
def main():
    st.title("Spotify Artist Compatibility Analysis")
    st.markdown("## Welcome to the Artist Compatibility Analyzer!")
    st.markdown("Log in to your Spotify account and analyze your top artists or check compatibility with new ones!")

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an option",
                              ("Login and View Top Artists", "Artist Compatibility Analysis"))

    params = st.query_params  # <- no parentheses

    if choice == "Login and View Top Artists":
        if "code" in params:
            st.success("Successfully logged in!")
            df = get_top_artists()
            if df is not None:
                st.write("### Your Top 10 Artists:")
                st.write(df[['artist_name','genres']])
                for _, row in df.iterrows():
                    st.write(f"**{row['artist_name']}**")
                    st.write(f"Genres: {', '.join(row['genres'])}")
                    if row['image_url']:
                        st.image(row['image_url'], width=100)
                    st.markdown("---")
            else:
                st.error("Authentication failed. Try again.")
        else:
            auth_url = sp_oauth.get_authorize_url()
            st.markdown(f"[Login with Spotify]({auth_url})")
            st.info("After logging in, return here to see your top artists.")

    if choice == "Artist Compatibility Analysis":
        artist_input = st.text_area("Enter artist names (comma-separated):")
        if st.button("Analyze Compatibility"):
            if not artist_input:
                st.error("Please enter at least one artist name.")
                return

            user_df = get_top_artists()
            if user_df is None:
                st.error("You must log in first!")
                return

            profile, tfidf = build_user_profile(user_df.to_dict(orient='records'))
            for artist in [a.strip() for a in artist_input.split(",")]:
                st.write(f"### {artist}")
                score = calculate_compatibility(profile, artist, tfidf)
                st.write(f"Compatibility score: {score:.2f}")
                genres, img = get_artist_genre(artist)
                st.write(f"Genres: {', '.join(genres)}")
                if img:
                    st.image(img, width=100)

if __name__ == "__main__":
    main()
