import streamlit as st
import pandas as pd
from sentiment_analysis import get_top_songs_by_sentiment

def compare_artists(data):
    st.subheader("🎸 Compare Artists (Debut, Popularity, Themes)")

    artist_list = data['artist'].unique()
    artist1 = st.selectbox("Select First Artist", artist_list, key="artist1")
    artist2 = st.selectbox("Select Second Artist", artist_list, key="artist2")

    comparison_data = data[data['artist'].isin([artist1, artist2])]

    # Key Metrics
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
        st.write(f"**Top 3 Songs by Sentiment for {artist1}:**")
        top_songs_1 = get_top_songs_by_sentiment(data, artist1)
        st.dataframe(top_songs_1)

    with col2:
        st.write(f"**Top 3 Songs by Sentiment for {artist2}:**")
        top_songs_2 = get_top_songs_by_sentiment(data, artist2)
        st.dataframe(top_songs_2)
