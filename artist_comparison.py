import streamlit as st
import pandas as pd
from collections import Counter
from sentiment_analysis import analyze_sentiment  # Import sentiment analysis function

stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every", "little"
])

def compare_artists(data):
    st.title("🎸 Compare Two Rock Artists")

    artist1 = st.selectbox("Select First Artist", data['artist'].unique())
    artist2 = st.selectbox("Select Second Artist", data['artist'].unique(), index=1)

    comparison_data = data[(data['artist'] == artist1) | (data['artist'] == artist2)]

    # **Apply Sentiment Analysis if Missing:**
    if 'sentiment' not in comparison_data.columns:
        comparison_data = analyze_sentiment(comparison_data)

    st.markdown("### Sentiment Analysis and Comparison")

    # Sentiment Comparison Over Time
    try:
        sentiment_by_year = comparison_data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
        artist_years = comparison_data['year'].unique()

        st.markdown("#### 📈 Sentiment Over Time")
        st.line_chart(sentiment_by_year.loc[artist_years], color=["#32CD32", "#FF6347"])

    except KeyError as e:
        st.error(f"Sentiment data not available: {e}")
        return

    # Most Popular Songs
    st.markdown("#### 🔥 Most Popular Songs (Top 3 by Views)")
    popular_songs = comparison_data.sort_values(by='views', ascending=False).groupby('artist').head(3)
    st.dataframe(popular_songs[['title', 'artist', 'views']])

    # Positive/Negative Songs
    st.markdown("#### 🎵 Top Positive and Negative Songs")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Most Positive Songs by {artist1}**")
        pos1, neg1 = get_top_songs_by_sentiment(comparison_data, artist1)
        st.table(pos1)

    with col2:
        st.markdown(f"**Most Negative Songs by {artist2}**")
        pos2, neg2 = get_top_songs_by_sentiment(comparison_data, artist2)
        st.table(neg2)

    # Lexical Complexity
    st.markdown("#### 📚 Lexical Complexity Analysis")
    comparison_data['lexical_diversity'] = comparison_data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0
    )
    lexical_comparison = comparison_data.groupby('artist')['lexical_diversity'].mean()
    st.bar_chart(lexical_comparison)
