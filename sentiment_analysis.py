import pandas as pd
from textblob import TextBlob
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def analyze_sentiment(data):
    data['year'] = data['year'].astype(str).str.replace(',', '')
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    
    if 'sentiment' not in data.columns:
        st.info("Running sentiment analysis...")
        data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        st.success("Sentiment analysis complete.")
    
    return data

def plot_sentiment_trend(data):
    data = analyze_sentiment(data)
    st.subheader("📊 Sentiment Analysis Over Time")
    
    if 'year' in data.columns:
        sentiment_by_year = data.groupby('year')['sentiment'].mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sentiment_by_year.index, sentiment_by_year.values, marker='o', color='tab:blue')
        ax.set_title("Average Sentiment of Rock Lyrics by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Sentiment")
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.grid(True)
        st.pyplot(fig)
