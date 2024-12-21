import streamlit as st
import pandas as pd
from textblob import TextBlob

@st.cache_data
def analyze_sentiment(data):
    data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

def search_sentiment_analysis(data):
    st.subheader("🎼 Sentiment Analysis of Song or Artist")
    
    # Dropdown for Artist or Song Analysis
    option = st.selectbox("Choose to Analyze by", ["Select", "Artist", "Song"])
    
    if option == "Select":
        st.info("Select Artist or Song to analyze sentiment.")
        return

    if option == "Artist":
        artist = st.selectbox("Select Artist", data['artist'].unique(), key="artist_sent")
        filtered = data[data['artist'] == artist]
    else:
        song = st.selectbox("Select Song", data['title'].unique(), key="song_sent")
        filtered = data[data['title'] == song]

    if filtered.empty:
        st.warning("No data found for selection.")
        return

    # Ensure Sentiment Column Exists
    if 'sentiment' not in filtered.columns:
        filtered = analyze_sentiment(filtered)

    # Display Results
    st.write(f"**Sentiment Distribution for {artist if option == 'Artist' else song}:**")
    st.bar_chart(filtered['sentiment'])

    # Sentiment Meaning
    st.markdown("**Sentiment Interpretation:**")
    st.write("- **Positive (0.1 to 1.0):** Joyful, upbeat songs.")
    st.write("- **Neutral (-0.1 to 0.1):** Calm, factual, or ambiguous.")
    st.write("- **Negative (-1.0 to -0.1):** Sad, dark, or aggressive tones.")
