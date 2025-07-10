import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Access Spotify credentials from Streamlit secrets
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
redirect_uri = "https://mikaylanorton-vibesync.streamlit.app/callback"

# Define SpotifyOAuth without global cache
def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=["user-library-read", "user-top-read", "playlist-read-private"]
    )

# Fetch user's top artists
def get_top_artists(sp, limit=10):
    top_artists = sp.current_user_top_artists(limit=limit, time_range='long_term')
    return pd.DataFrame([{
        'artist_name': artist['name'],
        'genres': artist['genres'],
        'popularity': artist['popularity'],
        'image_url': artist['images'][0]['url'] if artist['images'] else None
    } for artist in top_artists['items']])

# Fetch artist's genre and image information
def get_artist_genre(sp, artist_name):
    result = sp.search(q=f"artist:{artist_name}", type='artist', limit=5)
    if result['artists']['items']:
        artist = result['artists']['items'][0]
        genres = artist['genres'] if 'genres' in artist else []
        image_url = artist['images'][0]['url'] if artist['images'] else None
        if not genres:
            genres = ["Genres data not available"]
        return genres, image_url
    else:
        return [], None

# Build user profile
def build_user_profile(user_top_artists):
    tfidf = TfidfVectorizer(stop_words='english')
    genres_list = [' '.join(artist['genres']) for artist in user_top_artists]
    genre_matrix = tfidf.fit_transform(genres_list)
    user_profile = np.mean(genre_matrix.toarray(), axis=0)
    return user_profile, tfidf

# Calculate similarity
def calculate_compatibility(user_profile, artist_name, tfidf, sp):
    artist_genres, _ = get_artist_genre(sp, artist_name)
    artist_vector = tfidf.transform([' '.join(artist_genres)]).toarray()
    similarity_score = cosine_similarity(user_profile.reshape(1, -1), artist_vector)[0][0]
    return similarity_score

# Main app
def main():
    st.title("VibeSync: Spotify Artist Compatibility Analysis")
    st.markdown("## Welcome to the Artist Compatibility Analyzer!")

    st.set_page_config(page_title='VibeSync', 
                                    initial_sidebar_state='expanded', 
                                    page_icon=':musical_note:', 
                                    layout='wide') 
    # Handle Spotify OAuth
    query_params = st.query_params
    code = query_params.get("code", [None])[0]

    if "sp" not in st.session_state:
        if code:
            oauth = get_spotify_oauth()
            token_info = oauth.get_access_token(code)
            sp = spotipy.Spotify(auth=token_info['access_token'])
            st.session_state.sp = sp
            st.success("Logged in successfully!")
        else:
            auth_url = get_spotify_oauth().get_authorize_url()
            st.markdown(f"[Click here to log in with Spotify]({auth_url})")
            st.stop()

    sp = st.session_state.sp  # Use session-specific Spotify client

    # Sidebar navigation
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an option", 
                              ("Login and View Top Artists", "Artist Compatibility Analysis"), 
                              key="navigation_choice")

    # View top artists
    if choice == "Login and View Top Artists":
        try:
            user_top_artists = get_top_artists(sp, limit=30)
            st.write("### Your Top 10 Artists:")
            st.write(user_top_artists[['artist_name', 'genres']].head(10))

            for index, row in user_top_artists.iterrows():
                st.write(f"**{row['artist_name']}**")
                st.write(f"Genres: {', '.join(row['genres'])}")
                if row['image_url']:
                    st.image(row['image_url'], width=100)
                st.markdown("---")
        except spotipy.SpotifyException:
            st.error("Failed to retrieve data. Please ensure you're logged in with Spotify.")

    # Compatibility analysis
    elif choice == "Artist Compatibility Analysis":
        st.markdown("### Enter Artist Names")
        artist_input = st.text_area("Enter a list of artist names (comma-separated):")
        if st.button("Analyze Compatibility"):
            if artist_input:
                artist_names = [artist.strip() for artist in artist_input.split(",")]
                try:
                    user_top_artists = get_top_artists(sp, limit=10)
                    user_profile, tfidf = build_user_profile(user_top_artists.to_dict(orient='records'))

                    for artist_name in artist_names:
                        st.write(f"\n### Results for {artist_name}:")
                        compatibility_score = calculate_compatibility(user_profile, artist_name, tfidf, sp)
                        st.write(f"Compatibility score: {compatibility_score:.2f}")

                        genres, image_url = get_artist_genre(sp, artist_name)
                        st.write(f"Genres: {', '.join(genres)}")
                        if image_url:
                            st.image(image_url, width=100)
                        st.markdown("---")
                except spotipy.SpotifyException:
                    st.error("Failed to fetch data. Please log in again.")
            else:
                st.error("Please enter at least one artist name.")

    # Optional logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Log out"):
        for key in ["sp"]:
            st.session_state.pop(key, None)
        st.rerun()

# Run the app
if __name__ == "__main__":
    main()
