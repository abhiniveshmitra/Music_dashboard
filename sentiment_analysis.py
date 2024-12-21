import pandas as pd
from textblob import TextBlob
import streamlit as st
import matplotlib.pyplot as plt

# Sentiment Labeling Function
def label_sentiment(score):
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral"

@st.cache_data
def analyze_sentiment(data):
    data['year'] = data['year'].astype(str).str.replace(',', '')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    
    if 'sentiment' not in data.columns:
        st.info("Running sentiment analysis on all lyrics...")
        data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        data['sentiment_label'] = data['sentiment'].apply(label_sentiment)
        st.success("Sentiment analysis complete.")
    
    return data

# 🎵 Get Top 3 Songs by Sentiment for Each Artist
def get_top_songs_by_sentiment(data, artist):
    artist_data = data[data['artist'] == artist]
    top_songs = artist_data.sort_values(by='sentiment', ascending=False).head(3)
    return top_songs[['title', 'sentiment', 'sentiment_label']]
