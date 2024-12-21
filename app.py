import streamlit as st
import pandas as pd
from loader import load_data
from sentiment_analysis import search_sentiment_analysis
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
decades = st.sidebar.multiselect("Select Decades", data['year'].unique().tolist(), default=data['year'].unique())
selected_artist = st.sidebar.multiselect("Select Artists", data['artist'].unique().tolist())
word_count_filter = st.sidebar.slider("Max Word Count", min_value=50, max_value=600, value=600)

# Apply Filters
filtered_data = data[data['year'].isin(decades)]
filtered_data = filtered_data[filtered_data['lyric_length'] <= word_count_filter]

if selected_artist:
    filtered_data = filtered_data[filtered_data['artist'].isin(selected_artist)]

# --------------------
# Title and Main Section
# --------------------
st.title("🎸 Rock Lyrics Analysis Dashboard")
st.write("Explore rock music from 1950 to 2000 through sentiment analysis, word clouds, and artist comparisons.")

# --------------------
# Visualization 1 – Yearly Song Distribution
# --------------------
st.subheader("🎵 Number of Rock Songs by Year")
yearly_counts = filtered_data.groupby('year').size()
st.bar_chart(yearly_counts)

# --------------------
# Visualization 2 – Top Artists
# --------------------
st.subheader("🎤 Top 10 Artists by Number of Songs")
top_artists = filtered_data['artist'].value_counts().head(10)
st.bar_chart(top_artists)

# --------------------
# Sentiment Analysis
# --------------------
search_sentiment_analysis(filtered_data)

# --------------------
# Artist Comparison
# --------------------
compare_artists(filtered_data)

# --------------------
# Data Exploration
# --------------------
st.subheader("🗂 Explore the Data")
st.dataframe(filtered_data)
