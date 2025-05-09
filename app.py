import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyOauthError
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

def exchange_code_for_token():
    """Helper: only exchange if we actually have a code."""
    code = st.query_params.get("code", [None])[0]
    if not code:
        return None
    try:
        return sp_oauth.get_access_token(code)
    except SpotifyOauthError as e:
        st.error("⚠️ Authentication failed: please log in again.")
        return None

# Fetch user's top artists
def get_top_artists():
    token_info = exchange_code_for_token()
    if not token_info:
        return None
    sp = spotipy.Spotify(auth=token_info["access_token"])
    items = sp.current_user_top_artists(limit=10, time_range='long_term')['items']
    return pd.DataFrame([{
        'artist_name': artist['name'],
        'genres'     : artist['genres'],
        'popularity' : artist['popularity'],
        'image_url'  : artist['images'][0]['url'] if artist['images'] else None
    } for artist in items])

# Fetch artist's genre and image information
def get_artist_genre(artist_name):
    token_info = exchange_code_for_token()
    if not token_info:
        return [], None
    sp = spotipy.Spotify(auth=token_info["access_token"])
    result = sp.search(q=f"artist:{artist_name}", type='artist', limit=1)
    items = result.get('artists', {}).get('items', [])
    if not items:
        return [], None
    artist = items[0]
    return artist.get('genres', []), (artist['images'][0]['url'] if artist.get('images') else None)



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

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an option",
                              ("Login and View Top Artists", "Artist Compatibility Analysis"))

    params = st.query_params

    if choice == "Login and View Top Artists":
        if "code" in params:
            st.success("Logged in!")
            df = get_top_artists()
            if df is None:
                st.error("Could not fetch your top artists. Try logging in again.")
            else:
                st.write("### Your Top 10 Artists:")
                st.write(df[['artist_name','genres']])
                for _, row in df.iterrows():
                    st.write(f"**{row['artist_name']}**")
                    st.write(f"Genres: {', '.join(row['genres'])}")
                    if row['image_url']:
                        st.image(row['image_url'], width=100)
                    st.markdown("---")
        else:
            auth_url = sp_oauth.get_authorize_url()
            st.markdown(f"[Login with Spotify]({auth_url})")

    if choice == "Artist Compatibility Analysis":
        artist_input = st.text_area("Enter a list of artist names (comma-separated):")
        if st.button("Analyze Compatibility"):
            if not artist_input:
                return st.error("Please enter at least one artist name.")
            df = get_top_artists()
            if df is None:
                return st.error("You need to log in first (switch to Login and View Top Artists).")
            user_profile, tfidf = build_user_profile(df.to_dict(orient='records'))
            for artist in [a.strip() for a in artist_input.split(",")]:
                st.write(f"### {artist}")
                score = calculate_compatibility(user_profile, artist, tfidf)
                st.write(f"Compatibility score: {score:.2f}")
                genres, img = get_artist_genre(artist)
                st.write(f"Genres: {', '.join(genres)}")
                if img:
                    st.image(img, width=100)

if __name__ == "__main__":
    main()