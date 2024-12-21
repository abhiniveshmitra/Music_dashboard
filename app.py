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
# Header Section – Introduction
# --------------------
st.title("🎸 The Evolution of Rock Music (1950 - 2000)")
st.markdown("""
#### **Welcome to the Rock Lyrics Analysis Dashboard!**  
Explore the golden era of rock from **1950 to 2000**.  
This dashboard breaks down rock music through **sentiment analysis**, **word frequency**, and **artist comparisons**.  
Uncover trends in lyrics, identify the most popular artists, and visualize how the emotional tone of rock has evolved over the decades.  
""")

st.markdown("---")

# --------------------
# Sidebar – Filters
# --------------------
st.sidebar.header("🎚️ Filters")
data['decade'] = data['year'].apply(lambda x: (x // 10) * 10)

# Decade Filter
available_decades = sorted(data['decade'].unique())
selected_decades = st.sidebar.multiselect("Filter by Decades", available_decades, default=available_decades)

# Artist Selection – Two Artists Only
artist_options = data['artist'].unique()
selected_artists = st.sidebar.multiselect("Select Two Artists to Compare", artist_options, max_selections=2, default=artist_options[:2])

# Word Count Filter
word_count_filter = st.sidebar.slider("Max Word Count", min_value=50, max_value=600, value=600)

# --------------------
# Filter Data
# --------------------
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

# --------------------
# Title and Main Section
# --------------------
st.header("🎵 Rock Music Through the Years")
st.write("Discover the number of rock songs released over the years and the artists who defined this era.")

# --------------------
# Visualization 1 – Yearly Song Distribution
# --------------------
st.subheader("📅 Number of Rock Songs by Year")
st.markdown("This graph shows the distribution of rock songs released each year. Use the filters on the left to narrow down by decade or artist.")

yearly_counts = filtered_data.groupby('year').size()
st.bar_chart(yearly_counts)

# --------------------
# Visualization 2 – Top Artists
# --------------------
st.subheader("🎸 Top 10 Artists by Number of Songs")
st.markdown("Here are the top 10 artists with the highest number of rock songs in the filtered data.")

top_artists = filtered_data['artist'].value_counts().head(10)
st.bar_chart(top_artists)

# --------------------
# Sentiment Analysis Section
# --------------------
st.subheader("🎭 Sentiment Analysis of Rock Lyrics")
st.markdown("Analyze the emotional tone of rock lyrics over the years. This section reveals whether the music was uplifting, dark, or neutral.")

search_sentiment_analysis(filtered_data)

# --------------------
# Artist Comparison Section
# --------------------
st.subheader("🎤 Compare Two Rock Legends")
st.markdown("""
Select two artists from the sidebar to compare their lyrical diversity, most popular songs, and emotional tone.  
This section dives deep into how two artists' styles contrast across different time periods.
""")

if len(selected_artists) == 2:
    compare_artists(filtered_data)
