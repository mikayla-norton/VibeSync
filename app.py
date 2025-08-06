import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyOauthError
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Access Spotify credentials from Streamlit secrets
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
redirect_uri = "https://mikaylanorton-vibesync.streamlit.app/callback"

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
        'popularity': artist['popularity']
    } for artist in top_artists['items']])

# Build user profile vector from top artists
def build_user_profile(artists, top_k=20):
    all_genres = [g for art in artists for g in art['genres']]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(all_genres)
    genre_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    genre_index = tfidf.get_feature_names_out()
    user_profile = pd.DataFrame({
        'genre': genre_index,
        'score': genre_scores
    }).sort_values(by='score', ascending=False).head(top_k)
    return user_profile, tfidf

# Compatibility: cosine similarity between user profile and new artists
def analyze_compatibility(sp, user_profile, tfidf, new_artists, limit=10):
    # fetch genres for each new artist
    data = []
    for name in new_artists:
        results = sp.search(q=f"artist:{name}", type="artist", limit=1)
        items = results.get('artists', {}).get('items', [])
        if items:
            art = items[0]
            art_genres = art['genres']
            art_vec = tfidf.transform(art_genres).toarray().sum(axis=0)
            score = cosine_similarity(
                user_profile['score'].values.reshape(1, -1),
                art_vec.reshape(1, -1)
            )[0, 0]
            data.append({'artist': name, 'compatibility': score})
    return pd.DataFrame(data).sort_values(by='compatibility', ascending=False).head(limit)

def main():
    st.set_page_config(
        page_title='VibeSync',
        layout='wide',
        initial_sidebar_state='expanded',
        page_icon=':musical_note:'
    )
    st.title("VibeSync: Spotify Artist Compatibility Analysis")
    st.markdown("## Welcome to the Artist Compatibility Analyzer!")

    # Handle Spotify OAuth
    query_params = st.query_params
    code = query_params.get("code", [None])[0]

    if "sp" not in st.session_state:
        oauth = get_spotify_oauth()
        if code:
            try:
                token_info = oauth.get_access_token(code)
                sp = spotipy.Spotify(auth=token_info['access_token'])
                st.session_state.sp = sp
                st.success("Logged in successfully!")
                # Clear query params so a page reload won't reuse the same code
                st.experimental_set_query_params()
            except SpotifyOauthError as e:
                st.error(f"OAuth failed: {e}. Please log in again.")
                auth_url = oauth.get_authorize_url()
                st.markdown(f"[▶️ Log in with Spotify]({auth_url})")
                st.stop()
        else:
            auth_url = oauth.get_authorize_url()
            st.markdown(f"[▶️ Log in with Spotify]({auth_url})")
            st.stop()

    sp = st.session_state.sp

    # Sidebar navigation
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio(
        "Choose an option",
        ("Login and View Top Artists", "Artist Compatibility Analysis")
    )

    if choice == "Login and View Top Artists":
        try:
            user_top_artists = get_top_artists(sp, limit=10)
            st.write("### Your Top 10 Artists:")
            st.write(user_top_artists[['artist_name', 'genres']])
        except spotipy.SpotifyException:
            st.error("Failed to retrieve top artists. Please log in again.")
    elif choice == "Artist Compatibility Analysis":
        st.markdown("### Enter Artist Names")
        artist_input = st.text_area("Enter a list of artist names (comma-separated):")
        if st.button("Analyze Compatibility"):
            if artist_input.strip():
                artist_names = [a.strip() for a in artist_input.split(",")]
                try:
                    user_top_artists = get_top_artists(sp, limit=10)
                    user_profile, tfidf = build_user_profile(
                        user_top_artists.to_dict(orient='records')
                    )
                    results = analyze_compatibility(sp, user_profile, tfidf, artist_names)
                    st.write("### Compatibility Results:")
                    st.write(results)
                except spotipy.SpotifyException:
                    st.error("Failed to fetch data. Please log in again.")
            else:
                st.error("Please enter at least one artist name.")

    # Optional logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Log out"):
        st.session_state.pop("sp", None)
        st.rerun()

if __name__ == "__main__":
    main()
