import streamlit as st
import pandas as pd
from textblob import TextBlob

@st.cache_data
def analyze_sentiment(data):
    # Sentiment Analysis
    data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

def get_top_songs_by_sentiment(data, top_n=3):
    # Top Positive and Negative Songs
    top_positive = data.sort_values(by='sentiment', ascending=False).head(top_n)
    top_negative = data.sort_values(by='sentiment', ascending=True).head(top_n)
    return top_positive, top_negative

def search_sentiment_analysis(data):
    st.subheader("🎭 Sentiment Analysis of Artists")
    
    # Select Artist
    artist_choice = st.selectbox("Select Artist", data['artist'].unique())
    filtered = data[data['artist'] == artist_choice]
    filtered = analyze_sentiment(filtered)
    
    avg_sentiment = filtered['sentiment'].mean()
    
    st.write(f"**Average Sentiment for {artist_choice}:** {'😄 Positive' if avg_sentiment > 0 else '😞 Negative' if avg_sentiment < 0 else '😐 Neutral'}")

    # Top Songs by Sentiment
    top_pos, top_neg = get_top_songs_by_sentiment(filtered)
    
    st.write("**Top Positive Songs:**")
    st.dataframe(top_pos[['title', 'sentiment']])
    
    st.write("**Top Negative Songs:**")
    st.dataframe(top_neg[['title', 'sentiment']])
