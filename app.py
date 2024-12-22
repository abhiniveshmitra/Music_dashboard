import streamlit as st
import pandas as pd
import subprocess
import sys

# --------------------
# Ensure SpaCy and Model are Installed
# --------------------
def install_spacy():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        st.warning("SpaCy not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "spacy"])
        import spacy
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        st.success("SpaCy installed and model downloaded successfully!")
    except OSError:
        st.warning("Downloading SpaCy model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        st.success("SpaCy model downloaded successfully!")

install_spacy()

from loader import load_data
from sentiment_analysis import search_sentiment_analysis, analyze_sentiment
from artist_comparison import compare_artists

# --------------------
# Page Configuration
# --------------------
st.set_page_config(layout="wide", page_title="Rock Lyrics Dashboard", page_icon="🎸")

# --------------------
# Data Loading
# --------------------
data = load_data()

# --------------------
# Header Section – Introduction
# --------------------
st.markdown(
    """
    <style>
    .big-font {
        font-size:40px !important;
        text-align: center;
        font-weight: bold;
        color: #ff6347;
    }
    .subheader {
        font-size:22px !important;
        text-align: center;
        font-weight: bold;
        color: #1e90ff;
    }
    .highlight {
        font-size:18px;
        font-style: italic;
        text-align: center;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-font">🎸 The Evolution of Rock Music (1950 - 2000)</p>', unsafe_allow_html=True)
st.markdown('<p class="highlight">Explore how rock music evolved over the decades – from lyrical sentiment to artist comparisons.</p>', unsafe_allow_html=True)

st.markdown("---")

# --------------------
# Sidebar – Filters
# --------------------
st.sidebar.header("🎚️ Filters")
data['decade'] = data['year'].apply(lambda x: (x // 10) * 10)

available_decades = sorted(data['decade'].unique())
selected_decades = st.sidebar.multiselect("Filter by Decades", available_decades, default=available_decades)

artist_options = data['artist'].unique()
selected_artists = st.sidebar.multiselect("Select Two Artists to Compare", artist_options, max_selections=2, default=artist_options[:2])

word_count_filter = st.sidebar.slider("Max Word Count", min_value=50, max_value=600, value=600)

# --------------------
# Filter Data
# --------------------
filtered_data = data[data['decade'].isin(selected_decades)]
filtered_data = filtered_data[filtered_data['lyric_length'] <= word_count_filter]

if selected_artists and len(selected_artists) == 2:
    filtered_data = filtered_data[filtered_data['artist'].isin(selected_artists)]

    if 'sentiment' not in filtered_data.columns:
        filtered_data = analyze_sentiment(filtered_data)

else:
    st.sidebar.error("Please select exactly **two artists** for comparison.")

# --------------------
# Title and Main Section
# --------------------
st.markdown('<p class="subheader">🎵 Rock Music Through the Years</p>', unsafe_allow_html=True)
st.write("Discover the number of rock songs released over the years and the artists who defined this era.")

st.subheader("📅 Number of Rock Songs by Year")
yearly_counts = filtered_data.groupby('year').size()
yearly_counts_df = pd.DataFrame({'Year': yearly_counts.index, 'Count': yearly_counts.values})

st.bar_chart(yearly_counts_df.set_index('Year'), use_container_width=True)

st.subheader("🔥 Most Popular Artists by Cumulative Listens (Millions)")
top_popular_artists = (
    data.groupby('artist')['views'].sum()
    .sort_values(ascending=False)
    .head(10)
    .apply(lambda x: round(x / 1_000_000, 2))  # Convert to millions
)

top_popular_df = pd.DataFrame({'Artist': top_popular_artists.index, 'Views (M)': top_popular_artists.values})
st.bar_chart(top_popular_df.set_index('Artist'), use_container_width=True)

if len(selected_artists) == 2:
    st.subheader("🎤 Compare Two Rock Legends")
    st.markdown("""
    Select two artists from the sidebar to compare their lyrical diversity, most popular songs, and emotional tone.  
    This section dives deep into how two artists' styles contrast across different time periods.
    """)
    compare_artists(filtered_data)
