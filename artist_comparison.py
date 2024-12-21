import streamlit as st
import pandas as pd
from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment

def compare_artists(data):
    st.subheader("🎼 Compare Artists")

    # Select Two Artists for Comparison
    artist_options = data['artist'].unique()
    artist1 = st.selectbox("Select First Artist", artist_options)
    artist2 = st.selectbox("Select Second Artist", artist_options, index=1)
    
    comparison_data = data[(data['artist'] == artist1) | (data['artist'] == artist2)]
    comparison_data = analyze_sentiment(comparison_data)
    
    # Average Sentiment
    avg_sentiment = comparison_data.groupby('artist')['sentiment'].mean()
    st.bar_chart(avg_sentiment)

    # Top Songs by Sentiment
    for artist in [artist1, artist2]:
        st.write(f"**Top 3 Songs by Sentiment – {artist}:**")
        artist_data = comparison_data[comparison_data['artist'] == artist]
        top_pos, top_neg = get_top_songs_by_sentiment(artist_data)
        st.dataframe(top_pos[['title', 'sentiment']])
