import streamlit as st
from data_loader import load_data
from sentiment_analysis import (
    plot_complexity_vs_sentiment,
    analyze_sentiment,
    calculate_lyric_complexity
)
from artist_comparison import compare_artists

# Page Config
st.set_page_config(layout="wide", page_title="Rock Lyrics Dashboard", page_icon="🎸")

data = load_data()

st.title("🎸 Rock Lyrics Analysis Dashboard")
st.write("Explore rock music from 1950 to 2000 through lyrics analysis, sentiment, and complexity.")

# Sidebar Filters
if 'year' in data.columns:
    decades = st.sidebar.multiselect("Select Decades", data['year'].unique().tolist(), default=data['year'].unique())
else:
    st.sidebar.warning("Year column missing.")

filtered_data = data[data['year'].isin(decades)] if 'year' in data.columns else data

# 🛠️ Apply Sentiment and Complexity Analysis Before Plotting
filtered_data = analyze_sentiment(filtered_data)
filtered_data = calculate_lyric_complexity(filtered_data)

# Visualization Buttons
if st.button("Analyze Lyric Complexity"):
    plot_complexity_vs_sentiment(filtered_data)

if st.button("Compare Artists"):
    compare_artists(filtered_data)
