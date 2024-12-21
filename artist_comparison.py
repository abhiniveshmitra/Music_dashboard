import streamlit as st
import pandas as pd
from sentiment_analysis import get_top_songs_by_sentiment, analyze_sentiment

def compare_artists(data):
    st.subheader("🎸 Compare Artists (Popularity, Debut, Sentiment)")

    # Artist Selection with Placeholder
    artist_list = data['artist'].unique()
    artist1 = st.selectbox("Select First Artist", ["Select Artist"] + list(artist_list), key="artist1")
    artist2 = st.selectbox("Select Second Artist", ["Select Artist"] + list(artist_list), key="artist2")

    if artist1 == "Select Artist" or artist2 == "Select Artist":
        st.warning("Please select two artists for comparison.")
        return

    comparison_data = data[data['artist'].isin([artist1, artist2])]

    # Ensure Sentiment Analysis
    if 'sentiment' not in comparison_data.columns:
        comparison_data = analyze_sentiment(comparison_data)

    avg_sentiment = comparison_data.groupby('artist')['sentiment'].mean()
    avg_length = comparison_data.groupby('artist')['lyric_length'].mean()
    most_viewed = comparison_data.groupby('artist')['views'].max()
    debut_year = comparison_data.groupby('artist')['year'].min()

    # Display Insights
    st.subheader("🎧 Artist Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Average Sentiment**")
        st.dataframe(avg_sentiment.rename("Avg Sentiment").to_frame())

        st.write("**Most Popular Song**")
        st.dataframe(most_viewed.rename("Top Views").to_frame())

    with col2:
        st.write("**Debut Year**")
        st.dataframe(debut_year.rename("Debut Year").to_frame())

        st.write("**Average Lyric Length**")
        st.dataframe(avg_length.rename("Avg Words/Song").to_frame())

    # 🌟 Top 3 Sentiment Songs
    st.subheader("🔥 Top 3 Songs by Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Top 3 Songs for {artist1}:**")
        top_songs_1 = get_top_songs_by_sentiment(comparison_data, artist1)
        st.dataframe(top_songs_1)

    with col2:
        st.write(f"**Top 3 Songs for {artist2}:**")
        top_songs_2 = get_top_songs_by_sentiment(comparison_data, artist2)
        st.dataframe(top_songs_2)
