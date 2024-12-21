import streamlit as st
from textblob import TextBlob
import pandas as pd

@st.cache_data
def analyze_sentiment(data):
    # Apply sentiment analysis if not already calculated
    if 'sentiment' not in data.columns:
        data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

def get_top_songs_by_sentiment(data, artist_name=None):
    # Filter by artist if specified
    if artist_name:
        filtered_data = data[data['artist'] == artist_name]
    else:
        filtered_data = data
    
    if filtered_data.empty:
        return pd.DataFrame(columns=['title', 'artist', 'sentiment'])
    
    # Ensure sentiment analysis is applied
    filtered_data = analyze_sentiment(filtered_data)

    # Get top 3 positive and top 3 negative songs
    top_positive = filtered_data.nlargest(3, 'sentiment')[['title', 'artist', 'sentiment']]
    top_negative = filtered_data.nsmallest(3, 'sentiment')[['title', 'artist', 'sentiment']]
    
    return top_positive, top_negative

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
        filtered_data = analyze_sentiment(filtered_data)

        # Display Results
        st.write("### Sentiment Breakdown")
        st.dataframe(filtered_data[['title', 'artist', 'sentiment']])

        # Top 3 Positive/Negative Songs for Selected Artist
        st.write("#### 🎵 Top 3 Positive and Negative Songs")
        top_positive, top_negative = get_top_songs_by_sentiment(data, artist_choice)

        st.write("#### 🎵 Top 3 Positive Songs")
        st.table(top_positive)

        st.write("#### 😢 Top 3 Negative Songs")
        st.table(top_negative)
