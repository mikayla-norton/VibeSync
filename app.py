import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Configuration ------------------ #
SPOTIFY_SCOPES = "user-top-read,user-library-read"
REDIRECT_URI = "https://mikaylanorton-vibesync.streamlit.app/callback"

# ------------------ Spotify Authentication ------------------ #
def authenticate_spotify():
    oauth = SpotifyOAuth(
        client_id=st.secrets["spotify"]["client_id"],
        client_secret=st.secrets["spotify"]["client_secret"],
        redirect_uri=REDIRECT_URI,
        scope=SPOTIFY_SCOPES,
        cache_path=".spotify_cache"
    )
    token_info = oauth.get_cached_token()

    if not token_info:
        query_params = st.query_params
        code = query_params.get("code")

        if code:
            try:
                token_info = oauth.get_access_token(code, as_dict=False)
                st.query_params.clear()
            except spotipy.SpotifyOauthError:
                st.error("Authentication failed. Please retry.")
                return None
        else:
            auth_url = oauth.get_authorize_url()
            st.markdown(f"[🎧 Authenticate with Spotify]({auth_url})")
            st.stop()

    return spotipy.Spotify(auth=token_info) if token_info else None


# ------------------ Data Fetching ------------------ #
def fetch_top_artists(sp, limit=20):
    results = sp.current_user_top_artists(limit=limit, time_range='long_term')
    artists = []
    for item in results['items']:
        artists.append({
            'name': item['name'],
            'genres': item['genres'],
            'popularity': item['popularity']
        })
    return pd.DataFrame(artists)

# ------------------ Profile & Compatibility ------------------ #
def build_genre_profile(artists_df):
    genres = [genre for genres_list in artists_df.genres for genre in genres_list]
    vectorizer = TfidfVectorizer()
    genre_matrix = vectorizer.fit_transform(genres)
    genre_scores = np.asarray(genre_matrix.sum(axis=0)).ravel()

    return vectorizer, genre_scores

def calculate_compatibility(sp, user_vectorizer, user_genre_scores, artist_names):
    compat_results = []
    for name in artist_names:
        search_result = sp.search(q=f"artist:{name}", type="artist", limit=1)
        if search_result['artists']['items']:
            artist = search_result['artists']['items'][0]
            artist_genres = artist['genres']
            if artist_genres:
                artist_vec = np.asarray(user_vectorizer.transform(artist_genres).sum(axis=0)).ravel()   
                score = cosine_similarity([user_genre_scores], artist_vec)
                compat_results.append({'artist': name, 'compatibility': score[0][0]})
            else:
                compat_results.append({'artist': name, 'compatibility': 0.0})
        else:
            compat_results.append({'artist': name, 'compatibility': None})

    return pd.DataFrame(compat_results).sort_values(by='compatibility', ascending=False)

# ------------------ Streamlit UI ------------------ #
def main():
    st.set_page_config(
        page_title='VibeSync',
        layout='wide',
        initial_sidebar_state='expanded',
        page_icon=':musical_note:'
    )
    st.title("🎶 VibeSync")
    st.subheader("Spotify Artist Compatibility Analyzer")

    sp = authenticate_spotify()

    if sp:
        st.sidebar.success("Spotify Connected!")
        if st.sidebar.button("Logout"):
            import os
            os.remove(".spotify_cache")
            st.rerun()

        artists_df = fetch_top_artists(sp)
        st.write("### Your Top Artists:")
        st.dataframe(artists_df[['name', 'genres', 'popularity']], use_container_width=True)

        st.write("---")
        st.write("### Check Compatibility with New Artists")
        artist_input = st.text_area("Enter artist names (comma-separated):")

        if st.button("Analyze Compatibility", key="analyze_btn"):
            if artist_input.strip():
                user_vectorizer, user_genre_scores = build_genre_profile(artists_df)
                new_artist_list = [name.strip() for name in artist_input.split(",")]
                compat_df = calculate_compatibility(sp, user_vectorizer, user_genre_scores, new_artist_list)

                st.write("#### Compatibility Scores")
                st.dataframe(compat_df, use_container_width=True)
            else:
                st.warning("Please enter at least one artist name.")


if __name__ == "__main__":
    main()
