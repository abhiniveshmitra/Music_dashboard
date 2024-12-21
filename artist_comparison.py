import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment

@st.cache_data
def generate_wordcloud(data, artist):
    # Filter lyrics for selected artist
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    
    if artist_lyrics:
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(artist_lyrics)
        st.image(wordcloud.to_array())
    else:
        st.warning(f"No lyrics available for {artist}")

def compare_artists(data):
    st.subheader("🎤 Compare Two Artists")
    
    # Ensure sentiment analysis has been performed
    if 'sentiment' not in data.columns:
        st.warning("Running sentiment analysis on the dataset...")
        data = analyze_sentiment(data)
    
    # Artist Selection
    artists = data['artist'].unique()
    artist1 = st.selectbox("Select First Artist", artists, key='artist1')
    artist2 = st.selectbox("Select Second Artist", artists, key='artist2')

    if artist1 and artist2:
        comparison_data = data[data['artist'].isin([artist1, artist2])]
        
        # Average Sentiment Comparison
        st.write("**📊 Sentiment Comparison**")
        avg_sentiment = comparison_data.groupby('artist')['sentiment'].mean()
        st.bar_chart(avg_sentiment)

        # Top 3 Songs by Sentiment for Each Artist
        st.write(f"**🔥 Top 3 Songs by Sentiment for {artist1}**")
        top_songs_1 = get_top_songs_by_sentiment(data, artist1)
        st.table(top_songs_1)

        st.write(f"**🔥 Top 3 Songs by Sentiment for {artist2}**")
        top_songs_2 = get_top_songs_by_sentiment(data, artist2)
        st.table(top_songs_2)

        # 🎨 Word Clouds for Both Artists
        st.write(f"**☁️ Word Cloud for {artist1}**")
        generate_wordcloud(data, artist1)

        st.write(f"**☁️ Word Cloud for {artist2}**")
        generate_wordcloud(data, artist2)
