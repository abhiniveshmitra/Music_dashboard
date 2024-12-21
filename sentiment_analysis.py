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

def calculate_lyric_complexity(data):
    data['lyric_length'] = data['lyrics'].apply(lambda x: len(str(x).split()))
    data['unique_words'] = data['lyrics'].apply(lambda x: len(set(str(x).split())))
    data['lexical_diversity'] = data['unique_words'] / data['lyric_length']
    return data

def plot_complexity_vs_sentiment(data):
    data = calculate_lyric_complexity(data)
    
    st.subheader("📈 Lyric Complexity vs Sentiment")
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(data['lexical_diversity'], data['sentiment'], c=data['year'], cmap='viridis', alpha=0.7)
    ax.set_xlabel("Lexical Diversity (Unique Words / Total Words)")
    ax.set_ylabel("Sentiment Score")
    ax.set_title("Lexical Diversity vs Sentiment")
    plt.colorbar(scatter, ax=ax, label="Year")
    st.pyplot(fig)
