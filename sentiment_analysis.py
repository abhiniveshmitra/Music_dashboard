import streamlit as st
from textblob import TextBlob
import pandas as pd

@st.cache_data
def analyze_sentiment(data):
    if 'sentiment' not in data.columns:
        data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

def get_top_songs_by_sentiment(data, artist_name=None):
    if artist_name:
        filtered_data = data[data['artist'] == artist_name]
    else:
        filtered_data = data
    
    if filtered_data.empty:
        return pd.DataFrame(columns=['title', 'artist', 'sentiment'])
    
    filtered_data = analyze_sentiment(filtered_data)

    top_positive = filtered_data.nlargest(3, 'sentiment')[['title', 'artist', 'sentiment']]
    top_negative = filtered_data.nsmallest(3, 'sentiment')[['title', 'artist', 'sentiment']]
    
    return top_positive, top_negative

@st.cache_data
def search_sentiment_analysis(data):
    st.subheader("🔍 Sentiment Analysis for an Artist")

    # Dropdown for Artist Selection Only
    artist_list = data['artist'].unique()
    artist_choice = st.selectbox("Select Artist", artist_list)
    
    filtered_data = data[data['artist'] == artist_choice]
    
    if filtered_data.empty:
        st.warning("No data available for this artist.")
    else:
        filtered_data = analyze_sentiment(filtered_data)

        # Display Sentiment Summary
        avg_sentiment = filtered_data['sentiment'].mean()
        st.write(f"**Average Sentiment for {artist_choice}:** {avg_sentiment:.2f}")
        explain_sentiment(avg_sentiment)

        # Show Top 3 Positive/Negative Songs
        st.write("### 🎵 Top 3 Positive and Negative Songs")
        top_positive, top_negative = get_top_songs_by_sentiment(data, artist_choice)

        st.write("#### 🎵 Top 3 Positive Songs")
        st.table(top_positive)

        st.write("#### 😢 Top 3 Negative Songs")
        st.table(top_negative)

def explain_sentiment(sentiment_score):
    if sentiment_score > 0.5:
        st.success("😊 Highly Positive – This artist’s lyrics are generally uplifting and joyful.")
    elif sentiment_score > 0:
        st.info("🙂 Positive – A slightly optimistic tone overall.")
    elif sentiment_score < -0.5:
        st.error("😞 Highly Negative – This artist’s lyrics often convey sadness or frustration.")
    else:
        st.warning("😐 Neutral – Lyrics show a mix of positive and negative sentiment.")
