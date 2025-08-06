# VibeSync

A portfolio project showcasing a personalized music profile and compatibility insights powered by the Spotify API and Streamlit. This private reference directs viewers to a live Streamlit demonstration of the application.

---

## Overview

VibeSync connects to your Spotify account to:  
- Retrieve your listening history and audio features.  
- Analyze and visualize your unique music profile.  
- Compare your profile against curated playlists and other user archetypes.  
- Generate personalized recommendations and compatibility scores.

---

## Live Demo

Access the interactive Streamlit application to explore your music profile and compatibility insights:

[VibeSync Streamlit App](https://mikaylanorton-vibesync.streamlit.app/)

---

## Architecture

The pipeline consists of:

1. **Data Retrieval:**  
   - OAuth authentication with Spotify  
   - Fetch user’s top tracks, artists, and audio features using Spotipy  

2. **Profile Generation:**  
   - Aggregate audio features (danceability, energy, valence, etc.)  
   - Create radar charts and statistical summaries  

3. **Compatibility Analysis:**  
   - Compute similarity metrics between user profile and preset archetype profiles  
   - Rank recommended playlists and tracks  

4. **Visualization:**  
   - Streamlit UI renders interactive charts, tables, and recommendations  
   - Multiple pages for Profile, Compatibility, and Recommendations  

---

## Key Features

- **User Audio Profile:** Radar charts of top audio features.  
- **Listening Habits:** Bar charts of top tracks and artists.  
- **Compatibility Scores:** Numerical similarity against genre and mood archetypes.  
- **Recommendations:** Curated track suggestions based on profile distance metrics.  

---

## Project Structure

```
VibeSync/
├── streamlit_app/             # Streamlit interface code and multipage scripts
├── src/                       # Core modules: data fetching, analysis, utilities
├── notebooks/                 # Jupyter notebooks for exploratory analysis
├── data/                      # Sample JSON or CSV exports for testing
├── assets/                    # Static images, logos, and icons
├── .gitignore                 # Git ignore rules for temporary and large files
├── requirements.txt           # List of Python dependencies
└── README.md                  # Project overview and information
```


---

## Technologies

- **Frameworks:** Streamlit, Spotipy  
- **Languages:** Python 3.10+  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn  

---

For questions or feedback, please contact Mikayla Norton at [mikayla.e.norton@gmail.com](mailto:mikayla.e.norton@gmail.com).  
