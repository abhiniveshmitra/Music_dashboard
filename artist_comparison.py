import streamlit as st
import pandas as pd

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
    
    # Debut Year
    debut_year = comparison_data.groupby('artist')['year'].min()

    # Top Song by Views
    top_song = comparison_data.loc[
        comparison_data.groupby('artist')['views'].idxmax(), ['artist', 'title', 'views']
    ]

    # Display Results
    st.subheader("🎧 Artist Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Average Sentiment**")
        st.dataframe(avg_sentiment.rename("Avg Sentiment").to_frame())

        st.write("**Most Popular Song**")
        st.dataframe(top_song)

    with col2:
        st.write("**Debut Year**")
        st.dataframe(debut_year.rename("Debut Year").to_frame())

        st.write("**Average Lyric Length**")
        st.dataframe(avg_length.rename("Avg Words/Song").to_frame())
