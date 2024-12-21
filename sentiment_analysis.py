import pandas as pd
from textblob import TextBlob
import streamlit as st
import matplotlib.pyplot as plt

def label_sentiment(score):
    if score > 0.2:
        return "Positive"
    elif score < -0.2:
        return "Negative"
    else:
        return "Neutral"

@st.cache_data
def analyze_sentiment(data):
    # Run Sentiment Analysis Only if Needed
    if 'sentiment' not in data.columns:
        st.info("Running sentiment analysis on search results...")
        data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        data['sentiment_label'] = data['sentiment'].apply(label_sentiment)
        st.success("Sentiment analysis complete.")
    return data

def search_sentiment_analysis(data):
    st.subheader("🎤 Sentiment Analysis for Artist or Song")
    
    search_query = st.text_input("Enter Artist or Song Title", "")

    if search_query:
        result = data[
            (data['artist'].str.contains(search_query, case=False, na=False)) |
            (data['title'].str.contains(search_query, case=False, na=False))
        ]
        
        if not result.empty:
            # Perform Sentiment Analysis Only on Search Results
            result = analyze_sentiment(result)
            
            st.subheader(f"🎵 Sentiment for {search_query}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(result['title'], result['sentiment'], color='tab:blue')
            ax.set_xlabel("Song Title")
            ax.set_ylabel("Sentiment Score")
            ax.set_title(f"Sentiment Analysis for '{search_query}'")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.warning(f"No results found for '{search_query}'. Try another search.")
