import streamlit as st
from textblob import TextBlob
import pandas as pd

@st.cache_data
def analyze_sentiment(data):
    # Apply sentiment analysis to the lyrics column
    data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

@st.cache_data
def search_sentiment_analysis(data):
    st.subheader("🔍 Sentiment Analysis for a Specific Artist or Song")

    # Choose to search by artist or song
    search_type = st.radio("Search by:", ['Artist', 'Song'])
    
    if search_type == 'Artist':
        artist_list = data['artist'].unique()
        artist_choice = st.selectbox("Select Artist", artist_list)
        filtered_data = data[data['artist'] == artist_choice]
    else:
        song_list = data['title'].unique()
        song_choice = st.selectbox("Select Song", song_list)
        filtered_data = data[data['title'] == song_choice]
    
    if filtered_data.empty:
        st.warning("No data available for the selection.")
    else:
        # Perform sentiment analysis
        if 'sentiment' not in filtered_data.columns:
            st.warning("Running sentiment analysis...")
            filtered_data = analyze_sentiment(filtered_data)

        # Display Results
        st.write("### Sentiment Breakdown")
        st.dataframe(filtered_data[['title', 'artist', 'sentiment']])

        # Highlight Top 3 Positive and Negative Songs
        st.write("#### 🎵 Top 3 Positive Songs")
        top_positive = filtered_data.nlargest(3, 'sentiment')[['title', 'sentiment']]
        st.table(top_positive)

        st.write("#### 😢 Top 3 Negative Songs")
        top_negative = filtered_data.nsmallest(3, 'sentiment')[['title', 'sentiment']]
        st.table(top_negative)
