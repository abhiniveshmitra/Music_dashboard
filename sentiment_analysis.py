from textblob import TextBlob
import streamlit as st

@st.cache_data
def analyze_sentiment(data):
    if 'sentiment' not in data.columns:
        data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return data

def plot_sentiment_trend(data):
    data = analyze_sentiment(data)
    st.subheader("📊 Sentiment Analysis Over Time")
    if 'year' in data.columns:
        sentiment_by_year = data.groupby('year')['sentiment'].mean()
        st.line_chart(sentiment_by_year)
