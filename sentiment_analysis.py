import streamlit as st
from textblob import TextBlob
import pandas as pd

@st.cache_data
def analyze_sentiment(data):
    if 'sentiment' not in data.columns:
        data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

def get_top_songs_by_sentiment(data, artist_name=None, top_n=3):
    if artist_name:
        filtered_data = data[data['artist'] == artist_name]
    else:
        filtered_data = data

    if filtered_data.empty:
        return pd.DataFrame(columns=['title', 'artist', 'sentiment']), pd.DataFrame(columns=['title', 'artist', 'sentiment'])

    filtered_data = analyze_sentiment(filtered_data)

    top_positive = filtered_data.sort_values(by='sentiment', ascending=False).head(top_n)
    top_negative = filtered_data.sort_values(by='sentiment', ascending=True).head(top_n)

    return top_positive, top_negative

def search_sentiment_analysis(data):
    st.subheader("🔍 Sentiment Analysis by Artist")
    
    # Select Artist for Sentiment Analysis
    artist_list = data['artist'].unique()
    artist_choice = st.selectbox("Select Artist", artist_list, key="artist_search_sent")

    # Filter Data for Selected Artist
    filtered_data = data[data['artist'] == artist_choice]

    if filtered_data.empty:
        st.warning(f"No data available for {artist_choice}.")
        return
    
    # Apply Sentiment Analysis if Missing
    if 'sentiment' not in filtered_data.columns:
        filtered_data = analyze_sentiment(filtered_data)

    # Average Sentiment Display
    avg_sentiment = filtered_data['sentiment'].mean()
    st.write(f"**Average Sentiment for {artist_choice}:** {avg_sentiment:.2f}")
    explain_sentiment(avg_sentiment)

    # Display Top Positive and Negative Songs
    st.write("### 🎵 Top 3 Positive and Negative Songs")
    top_positive, top_negative = get_top_songs_by_sentiment(filtered_data, artist_choice)

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 🎵 Top 3 Positive Songs")
        st.table(top_positive[['title', 'sentiment']])

    with col2:
        st.write("#### 😢 Top 3 Negative Songs")
        st.table(top_negative[['title', 'sentiment']])

def explain_sentiment(sentiment_score):
    if sentiment_score > 0.5:
        st.success("😊 Highly Positive – Lyrics are uplifting and joyful.")
    elif sentiment_score > 0:
        st.info("🙂 Positive – A generally optimistic tone.")
    elif sentiment_score < -0.5:
        st.error("😞 Highly Negative – Lyrics convey sadness or frustration.")
    else:
        st.warning("😐 Neutral – Mixed or balanced sentiment.")
