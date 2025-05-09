import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyOauthError
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── STREAMLIT / SPOTIFY SETUP ────────────────────────────────────────────────

client_id     = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
redirect_uri  = "https://mikaylanorton-vibesync.streamlit.app/callback"

sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=[
        "user-library-read",
        "user-top-read",
        "playlist-read-private"
    ]
)

def exchange_code_for_token():
    """Safely exchange the ?code=… from the URL for an access token."""
    code = st.query_params.get("code", [None])[0]
    if not code:
        return None

    try:
        return sp_oauth.get_access_token(code)
    except SpotifyOauthError:
        st.error("⚠️ Authentication failed. Please click Login again.")
        return None

# ─── DATA-FETCHING FUNCTIONS ─────────────────────────────────────────────────

def get_top_artists():
    token_info = exchange_code_for_token()
    if not token_info:
        return None

    sp = spotipy.Spotify(auth=token_info["access_token"])
    items = sp.current_user_top_artists(limit=10, time_range="long_term")["items"]

    return pd.DataFrame([{
        "artist_name": artist["name"],
        "genres"     : artist["genres"],
        "popularity" : artist["popularity"],
        "image_url"  : artist["images"][0]["url"] if artist["images"] else None
    } for artist in items])


def get_artist_genre(artist_name):
    token_info = exchange_code_for_token()
    if not token_info:
        return [], None

    sp = spotipy.Spotify(auth=token_info["access_token"])
    result = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    items = result.get("artists", {}).get("items", [])
    if not items:
        return [], None

    artist    = items[0]
    genres    = artist.get("genres", [])
    image_url = artist["images"][0]["url"] if artist.get("images") else None
    return genres, image_url

# ─── PROFILE & SIMILARITY ─────────────────────────────────────────────────────

def build_user_profile(user_top_artists):
    tfidf       = TfidfVectorizer(stop_words="english")
    genres_list = [" ".join(a["genres"]) for a in user_top_artists]
    matrix      = tfidf.fit_transform(genres_list)
    profile     = np.mean(matrix.toarray(), axis=0)
    return profile, tfidf

def calculate_compatibility(user_profile, artist_name, tfidf):
    artist_genres, _ = get_artist_genre(artist_name)
    artist_vec       = tfidf.transform([" ".join(artist_genres)]).toarray()
    score            = cosine_similarity(user_profile.reshape(1, -1), artist_vec)[0][0]
    return score

# ─── STREAMLIT UI ────────────────────────────────────────────────────────────

def main():
    st.title("🎵 Spotify Artist Compatibility Analysis")
    st.markdown("Analyze your top artists or check compatibility with new ones!")

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio(
        "Choose an option",
        ("Login and View Top Artists", "Artist Compatibility Analysis")
    )

    # ─ Login & View Top Artists ───────────────────────────────────────────────
    if choice == "Login and View Top Artists":
        # If they already have a code=… in the URL, we've got auth
        if "code" in st.query_params:
            st.success("✅ Logged in with Spotify!")
            df = get_top_artists()

            if df is None:
                st.error("Could not fetch your top artists. Try logging in again.")
                return

            st.write("### Your Top 10 Artists:")
            st.write(df[["artist_name", "genres"]])

            for _, row in df.iterrows():
                st.write(f"**{row['artist_name']}**")
                st.write(f"Genres: {', '.join(row['genres'])}")
                if row["image_url"]:
                    st.image(row["image_url"], width=100)
                st.markdown("---")

        else:
            st.info("You need to log in with Spotify to continue.")
            if st.button("🔑 Log in with Spotify"):
                auth_url = sp_oauth.get_authorize_url()
                st.experimental_set_query_params()  # clear old params
                st.write(f"👉 [Click here to authenticate]({auth_url})")

    # ─ Artist Compatibility Analysis ──────────────────────────────────────────
    elif choice == "Artist Compatibility Analysis":
        artist_input = st.text_area("Enter artist names (comma-separated):")

        if st.button("Analyze Compatibility"):
            if not artist_input:
                st.error("Please enter at least one artist name.")
                return

            df = get_top_artists()
            if df is None:
                st.error("You must log in first (go to the Login tab).")
                return

            user_profile, tfidf = build_user_profile(df.to_dict(orient="records"))
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
