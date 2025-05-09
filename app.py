import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyOauthError
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── STREAMLIT / SPOTIFY SETUP ────────────────────────────────────────────────
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlencode

# Access Spotify credentials from Streamlit secrets
client_id = st.secrets["spotify"]["client_id"]
client_secret = st.secrets["spotify"]["client_secret"]
redirect_uri = "https://mikaylanorton-vibesync.streamlit.app/callback"

<<<<<<< Updated upstream
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
=======
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
>>>>>>> Stashed changes
    result = sp.search(q=f"artist:{artist_name}", type='artist', limit=5)
    if result['artists']['items']:
        artist = result['artists']['items'][0]
        genres = artist['genres'] if 'genres' in artist else []
        image_url = artist['images'][0]['url'] if artist['images'] else None
<<<<<<< Updated upstream
=======
        if not genres:
            genres = ["Genres data not available"]
>>>>>>> Stashed changes
        return genres, image_url
    else:
        return [], None

<<<<<<< Updated upstream

# Build user profile based on genres of top artists
=======
# Build user profile
>>>>>>> Stashed changes
def build_user_profile(user_top_artists):
    tfidf = TfidfVectorizer(stop_words='english')
    genres_list = [' '.join(artist['genres']) for artist in user_top_artists]
    genre_matrix = tfidf.fit_transform(genres_list)
    user_profile = np.mean(genre_matrix.toarray(), axis=0)
    return user_profile, tfidf

<<<<<<< Updated upstream

# Calculate similarity between user profile and artist genres
def calculate_compatibility(user_profile, artist_name, tfidf):
    artist_genres, _ = get_artist_genre(artist_name)
=======
# Calculate similarity
def calculate_compatibility(user_profile, artist_name, tfidf, sp):
    artist_genres, _ = get_artist_genre(sp, artist_name)
>>>>>>> Stashed changes
    artist_vector = tfidf.transform([' '.join(artist_genres)]).toarray()
    similarity_score = cosine_similarity(user_profile.reshape(1, -1), artist_vector)[0][0]
    return similarity_score

<<<<<<< Updated upstream

# Streamlit app layout and flow
=======
# Main app
>>>>>>> Stashed changes
def main():
    st.title("Spotify Artist Compatibility Analysis")
    st.markdown("## Welcome to the Artist Compatibility Analyzer!")

<<<<<<< Updated upstream
=======
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
>>>>>>> Stashed changes
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an option", 
                              ("Login and View Top Artists", "Artist Compatibility Analysis"), 
                              key="navigation_choice")

    # View top artists
    if choice == "Login and View Top Artists":
<<<<<<< Updated upstream
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
=======
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
>>>>>>> Stashed changes
        artist_input = st.text_area("Enter a list of artist names (comma-separated):")
        if st.button("Analyze Compatibility"):
            if artist_input:
                artist_names = [artist.strip() for artist in artist_input.split(",")]
<<<<<<< Updated upstream
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
=======
                try:
                    user_top_artists = get_top_artists(sp, limit=10)
                    user_profile, tfidf = build_user_profile(user_top_artists.to_dict(orient='records'))

                    for artist_name in artist_names:
                        st.write(f"\n### Results for {artist_name}:")
                        compatibility_score = calculate_compatibility(user_profile, artist_name, tfidf, sp)
                        st.write(f"Compatibility score: {compatibility_score:.2f}")
>>>>>>> Stashed changes

                        genres, image_url = get_artist_genre(sp, artist_name)
                        st.write(f"Genres: {', '.join(genres)}")
                        if image_url:
                            st.image(image_url, width=100)
                        st.markdown("---")
                except spotipy.SpotifyException:
                    st.error("Failed to fetch data. Please log in again.")
            else:
                st.error("Please enter at least one artist name.")

<<<<<<< Updated upstream

if __name__ == "__main__":
    main()
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


=======
    # Optional logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Log out"):
        for key in ["sp"]:
            st.session_state.pop(key, None)
        st.rerun()

# Run the app
>>>>>>> Stashed changes
if __name__ == "__main__":
    main()
