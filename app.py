import streamlit as st
import pandas as pd
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
# Sidebar – Filters
# --------------------
st.sidebar.header("Filters")
data['decade'] = data['year'].apply(lambda x: (x // 10) * 10)

# Decade Filter
available_decades = sorted(data['decade'].unique())
selected_decades = st.sidebar.multiselect("Select Decades", available_decades, default=available_decades)

# Artist Selection – Two Artists Only
artist_options = data['artist'].unique()
selected_artists = st.sidebar.multiselect("Select Two Artists", artist_options, max_selections=2, default=artist_options[:2])

word_count_filter = st.sidebar.slider("Max Word Count", min_value=50, max_value=600, value=600)

# Apply Filters
filtered_data = data[data['decade'].isin(selected_decades)]
filtered_data = filtered_data[filtered_data['lyric_length'] <= word_count_filter]

if selected_artists and len(selected_artists) == 2:
    filtered_data = filtered_data[filtered_data['artist'].isin(selected_artists)]

    # Ensure Sentiment is Applied
    if 'sentiment' not in filtered_data.columns:
        filtered_data = analyze_sentiment(filtered_data)

    # Perform Artist Comparison
    compare_artists(filtered_data)
else:
    st.sidebar.error("Please select exactly **two artists** for comparison.")
