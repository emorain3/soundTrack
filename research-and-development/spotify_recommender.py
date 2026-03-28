import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import kagglehub
import os

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎵",
    layout="wide",
)

# ─── Custom CSS  (light theme) ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
}

.stApp {
    background: #f5f4f9;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e0dff0;
}

.stRadio label, .stSelectbox label, .stSlider label,
.stTextInput label, div[data-testid="stMarkdownContainer"] p {
    color: #1a1a2e !important;
    font-size: 14px;
}

.stTextInput input {
    background: #f5f4f9;
    border: 1px solid #c8c6e0;
    border-radius: 8px;
    color: #1a1a2e;
    font-family: 'DM Sans', sans-serif;
}
.stTextInput input:focus {
    border-color: #7c3aed;
    box-shadow: 0 0 0 2px #7c3aed22;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #f5f4f9;
    border: 1px solid #c8c6e0;
    border-radius: 8px;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 15px;
    padding: 0.65rem 2rem;
    width: 100%;
    letter-spacing: 0.02em;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    color: #1a1a2e;
}

hr { border-color: #e0dff0; }

.rec-card {
    background: #ffffff;
    border: 1px solid #e0dff0;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.rec-card:hover {
    border-color: #7c3aed;
    box-shadow: 0 2px 12px #7c3aed18;
}

.rank-badge {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #7c3aed;
    min-width: 34px;
    text-align: center;
}

.track-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 15px;
    color: #1a1a2e;
    margin: 0 0 3px 0;
}

.track-meta {
    font-size: 13px;
    color: #6b6b8a;
    margin: 0;
}

.sim-bar-wrap {
    margin-left: auto;
    text-align: right;
    min-width: 70px;
}

.sim-score {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 17px;
    color: #7c3aed;
}

.sim-label {
    font-size: 11px;
    color: #9999b8;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.genre-pill {
    display: inline-block;
    background: #ede9fe;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 500;
    color: #5b21b6;
    margin-left: 7px;
    vertical-align: middle;
}

.explicit-tag {
    display: inline-block;
    background: #fee2e2;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 10px;
    font-weight: 700;
    color: #991b1b;
    margin-right: 5px;
    vertical-align: middle;
    letter-spacing: 0.05em;
}

.query-card {
    background: #ede9fe;
    border: 1px solid #c4b5fd;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 18px;
}

.query-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}

.query-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 16px;
    color: #1a1a2e;
}

.query-sub {
    font-size: 13px;
    color: #6b6b8a;
    margin-top: 2px;
}

.mode-header {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: #9999b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 12px 0 4px 0;
}

.landing-card {
    background: #ffffff;
    border: 1px solid #e0dff0;
    border-radius: 14px;
    padding: 24px 28px;
}

.landing-card h3 { margin-top: 0; font-size: 18px; }

.results-header {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #6b6b8a;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f5f4f9; }
::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    csv_path = os.path.join(path, "dataset.csv")
    df = pd.read_csv(csv_path, encoding="latin-1", engine="python", on_bad_lines="skip")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    df.dropna(inplace=True)
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset="track_name", keep="first")
    df = df.reset_index(drop=True)
    return df


@st.cache_resource(show_spinner="Building recommendation engine...")
def build_engine(_df):
    features = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]
    X = _df[features].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # Use 2000 neighbors so filtering never exhausts the pool
    knn = NearestNeighbors(n_neighbors=2000, metric="cosine", algorithm="brute")
    knn.fit(X_scaled)
    return scaler, X_scaled, knn, features


df = load_data()
scaler, X_scaled, knn, FEATURES = build_engine(df)
GENRES = sorted(df["track_genre"].unique().tolist())


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Find your sound")
    st.markdown("---")

    mode = st.radio("Search mode", ["By a song I like", "By mood / vibe"])

    st.markdown("---")
    st.markdown('<p class="mode-header">Genre</p>', unsafe_allow_html=True)
    selected_genre = st.selectbox("Genre", ["Any genre"] + GENRES,
                                  label_visibility="collapsed")

    st.markdown('<p class="mode-header">Recommendations to show</p>',
                unsafe_allow_html=True)
    n_results = st.slider("N", min_value=3, max_value=20, value=10,
                          label_visibility="collapsed")

    st.markdown('<p class="mode-header">Content</p>', unsafe_allow_html=True)
    explicit_pref = st.radio("Content", ["All tracks", "Clean only", "Explicit only"],
                             label_visibility="collapsed")

    st.markdown("---")

    if mode == "By a song I like":
        st.markdown('<p class="mode-header">Seed song</p>', unsafe_allow_html=True)
        song_input   = st.text_input("Song name", placeholder="e.g. Blinding Lights",
                                     label_visibility="collapsed")
        artist_input = st.text_input("Artist (optional)", placeholder="e.g. The Weeknd",
                                     label_visibility="collapsed")
    else:
        st.markdown('<p class="mode-header">Energy</p>', unsafe_allow_html=True)
        energy_val = st.slider("Energy", 0.0, 1.0, 0.7, 0.01,
                               label_visibility="collapsed")

        st.markdown('<p class="mode-header">Danceability</p>', unsafe_allow_html=True)
        dance_val  = st.slider("Danceability", 0.0, 1.0, 0.6, 0.01,
                               label_visibility="collapsed")

        st.markdown('<p class="mode-header">Valence (happiness)</p>',
                    unsafe_allow_html=True)
        valence_val = st.slider("Valence", 0.0, 1.0, 0.5, 0.01,
                                label_visibility="collapsed")

        st.markdown('<p class="mode-header">Acousticness</p>', unsafe_allow_html=True)
        acoustic_val = st.slider("Acousticness", 0.0, 1.0, 0.1, 0.01,
                                 label_visibility="collapsed")

        st.markdown('<p class="mode-header">Instrumentalness</p>',
                    unsafe_allow_html=True)
        instrumental = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01,
                                 label_visibility="collapsed")

    search_btn = st.button("Get recommendations")


# ─── Main area ────────────────────────────────────────────────────────────────
st.markdown("# Spotify Recommender")
st.markdown("Discover tracks that match your taste — powered by audio feature similarity.")
st.markdown("---")

if search_btn:
    query_vector = None
    error_msg    = None
    seed_idx     = None

    # ── Build query vector ────────────────────────────────────────────────────
    if mode == "By a song I like":
        if not song_input.strip():
            error_msg = "Please enter a song name."
        else:
            mask = df["track_name"].str.lower() == song_input.strip().lower()
            if artist_input.strip():
                mask &= df["artists"].str.lower().str.contains(
                    artist_input.strip().lower(), na=False
                )
            matches = df[mask]
            if matches.empty:
                error_msg = (
                    f"Could not find **{song_input}**. "
                    "Try a different spelling or leave the artist field blank."
                )
            else:
                seed_idx     = matches["popularity"].idxmax()
                query_vector = X_scaled[seed_idx].reshape(1, -1)
                row = df.iloc[seed_idx]
                st.markdown(
                    f'<div class="query-card">'
                    f'<div class="query-label">Seed track</div>'
                    f'<div class="query-title">{row["track_name"]}'
                    f'<span class="genre-pill">{row["track_genre"]}</span></div>'
                    f'<div class="query-sub">{row["artists"]} &nbsp;·&nbsp; '
                    f'Popularity {int(row["popularity"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        med = df[FEATURES].median()
        mood_raw = np.array([[
            dance_val, energy_val, med["loudness"], med["speechiness"],
            acoustic_val, instrumental, med["liveness"], valence_val, med["tempo"]
        ]])
        query_vector = scaler.transform(mood_raw)
        st.markdown(
            f'<div class="query-card">'
            f'<div class="query-label">Your vibe</div>'
            f'<div class="query-title">'
            f'Energy {energy_val:.2f} &nbsp;·&nbsp; '
            f'Dance {dance_val:.2f} &nbsp;·&nbsp; '
            f'Valence {valence_val:.2f} &nbsp;·&nbsp; '
            f'Acoustic {acoustic_val:.2f}'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # ── KNN + filter ──────────────────────────────────────────────────────────
    if error_msg:
        st.error(error_msg)

    elif query_vector is not None:
        distances, indices = knn.kneighbors(query_vector)
        rec_indices   = indices[0]
        rec_distances = distances[0]

        # Remove the seed song itself
        if seed_idx is not None:
            keep          = rec_indices != seed_idx
            rec_indices   = rec_indices[keep]
            rec_distances = rec_distances[keep]

        results = df.iloc[rec_indices].copy()
        results["similarity"] = 1 - rec_distances

        # Genre filter
        if selected_genre != "Any genre":
            results = results[results["track_genre"] == selected_genre]

        # Explicit filter
        if explicit_pref == "Clean only":
            results = results[results["explicit"] == False]
        elif explicit_pref == "Explicit only":
            results = results[results["explicit"] == True]

        # Take exactly N results
        results = results.head(n_results)

        if results.empty:
            st.warning(
                "No tracks matched your filters. "
                "Try **Any genre** or a different content filter."
            )
        else:
            st.markdown(
                f'<p class="results-header">{len(results)} recommendations</p>',
                unsafe_allow_html=True
            )

            for rank, (_, row) in enumerate(results.iterrows(), 1):
                sim_pct      = int(row["similarity"] * 100)
                explicit_tag = '<span class="explicit-tag">E</span>' \
                               if row["explicit"] else ""
                st.markdown(
                    f'<div class="rec-card">'
                    f'  <div class="rank-badge">{rank:02d}</div>'
                    f'  <div style="flex:1; min-width:0;">'
                    f'    <p class="track-title">{explicit_tag}{row["track_name"]}'
                    f'      <span class="genre-pill">{row["track_genre"]}</span></p>'
                    f'    <p class="track-meta">{row["artists"]}'
                    f'      &nbsp;·&nbsp; Popularity {int(row["popularity"])}</p>'
                    f'  </div>'
                    f'  <div class="sim-bar-wrap">'
                    f'    <div class="sim-score">{sim_pct}%</div>'
                    f'    <div class="sim-label">match</div>'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with st.expander("Audio feature comparison"):
                show_cols = ["track_name", "danceability", "energy",
                             "valence", "acousticness", "tempo", "popularity"]
                st.dataframe(
                    results[show_cols].reset_index(drop=True).round(3),
                    use_container_width=True,
                    hide_index=True,
                )

# ── Landing state ─────────────────────────────────────────────────────────────
else:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            '<div class="landing-card"><h3>🎵 By a song you love</h3>'
            '<p style="color:#6b6b8a;font-size:14px;line-height:1.7">'
            'Type in any track name and the engine finds songs with a similar '
            'audio fingerprint — same energy, tempo, mood, and texture.</p></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="landing-card"><h3>🎚️ By mood / vibe</h3>'
            '<p style="color:#6b6b8a;font-size:14px;line-height:1.7">'
            'Use the sliders to describe exactly how you want to feel. '
            'High energy + high danceability = party mode. '
            'Low energy + high acousticness = late-night focus.</p></div>',
            unsafe_allow_html=True
        )
        