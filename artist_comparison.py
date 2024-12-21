import streamlit as st
import pandas as pd

def compare_artists(data):
    st.subheader("🎤 Compare Artists")

    # Dropdowns to Select Artists
    artist_list = data['artist'].unique()
    artist1 = st.selectbox("Select First Artist", artist_list, key="artist1")
    artist2 = st.selectbox("Select Second Artist", artist_list, key="artist2")

    comparison_data = data[data['artist'].isin([artist1, artist2])]

    # Calculate Insights (Sentiment, Popularity, etc.)
    avg_sentiment = comparison_data.groupby('artist')['sentiment'].mean()
    avg_length = comparison_data.groupby('artist')['lyric_length'].mean()
    most_viewed = comparison_data.groupby('artist')['views'].max()

    # Show Results
    st.subheader("🎧 Artist Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Average Sentiment**")
        st.dataframe(avg_sentiment.rename("Avg Sentiment").to_frame())

    with col2:
        st.write(f"**Average Lyric Length**")
        st.dataframe(avg_length.rename("Avg Words/Song").to_frame())

    st.subheader("🔥 Most Popular Songs by Views")
    st.dataframe(most_viewed.rename("Max Views").to_frame())
