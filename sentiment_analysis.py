
import pandas as pd
from textblob import TextBlob
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def analyze_sentiment(data):
    # Clean and prepare data
    data['year'] = data['year'].astype(str).str.replace(',', '')  # Remove commas from year
    data['year'] = pd.to_numeric(data['year'], errors='coerce')  # Convert to numeric
    
    # Run Sentiment Analysis if not already present
    if 'sentiment' not in data.columns:
        st.info("Running sentiment analysis (this may take a few minutes)...")
        data['sentiment'] = data['lyrics'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        st.success("Sentiment analysis complete.")
    
    return data


def plot_sentiment_trend(data):
    # Perform Sentiment Analysis
    data = analyze_sentiment(data)
    
    # Title and Explanation
    st.subheader("📊 Sentiment Analysis of Rock Songs Over Time")
    st.write("This chart displays the average sentiment of rock lyrics from 1950 to 2000. "
             "Positive values indicate positive sentiment, while negative values reflect negative sentiment.")
    
    # Plot Sentiment Trend
    if 'year' in data.columns:
        sentiment_by_year = data.groupby('year')['sentiment'].mean()

        # Create Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sentiment_by_year.index, sentiment_by_year.values, marker='o', color='tab:blue')

        # Labels and Titles
        ax.set_title("Average Sentiment of Rock Lyrics by Year", fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Average Sentiment Score", fontsize=12)
        
        # Add Grid and Horizontal Line for Neutral Sentiment
        ax.axhline(0, color='red', linestyle='--', linewidth=1)  # Neutral line at y=0
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Display Plot
        st.pyplot(fig)
    else:
        st.warning("Year column missing. Unable to plot sentiment trend.")
