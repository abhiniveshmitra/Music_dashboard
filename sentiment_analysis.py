import streamlit as st
import pandas as pd
from textblob import TextBlob

@st.cache_data
def analyze_sentiment(data):
    # Perform sentiment analysis
    data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

def get_top_songs_by_sentiment(data, artist=None):
    if artist:
        artist_data = data[data['artist'] == artist]
    else:
        artist_data = data

    # Sort by sentiment (highest to lowest) and get top 3
    top_songs = artist_data.sort_values(by='sentiment', ascending=False).head(3)[['title', 'artist', 'sentiment']]
    return top_songs

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

    # Display Sentiment Analysis
    st.write(f"**Sentiment Distribution for {artist if option == 'Artist' else song}:**")
    st.line_chart(filtered[['year', 'sentiment']].set_index('year'))

    # 🌟 Display Top 3 Songs by Sentiment
    st.subheader("🔥 Top 3 Songs by Sentiment")
    top_songs = get_top_songs_by_sentiment(filtered, artist if option == "Artist" else None)
    st.table(top_songs)

    # Sentiment Meaning
    st.markdown("**Sentiment Interpretation:**")
    st.write("- **Positive (0.1 to 1.0):** Joyful, upbeat songs.")
    st.write("- **Neutral (-0.1 to 0.1):** Calm, factual, or ambiguous.")
    st.write("- **Negative (-1.0 to -0.1):** Sad, dark, or aggressive tones.")
