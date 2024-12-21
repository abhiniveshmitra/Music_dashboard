from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment
import streamlit as st
import pandas as pd
from collections import Counter

# Stopwords List
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every", "little"
])


# Frequent Words Analysis (1-based indexing)
def get_most_frequent_words(data, artist, top_n=10):
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()
    filtered_words = [word.lower() for word in words if len(word) > 4 and word.lower() not in stop_words]
    most_common = Counter(filtered_words).most_common(top_n)
    df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
    df.index = df.index + 1  # 1-based indexing
    return df


# Artist Comparison Function
def compare_artists(data):
    st.title("🎸 Compare Two Rock Artists")

    col1, col2 = st.columns(2)

    with col1:
        artist1 = st.selectbox("Select First Artist", data['artist'].unique())

    with col2:
        artist2 = st.selectbox("Select Second Artist", data['artist'].unique(), index=1)

    # Filter Data for Comparison
    comparison_data = data[(data['artist'] == artist1) | (data['artist'] == artist2)]

    # Ensure Sentiment is Calculated
    if 'sentiment' not in comparison_data.columns:
        comparison_data = analyze_sentiment(comparison_data)

    # Display Popular Songs
    st.markdown("### 🔥 Most Popular Songs (Top 3 by Views)")
    popular_songs = comparison_data.sort_values(by='views', ascending=False).groupby('artist').head(3)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Most Popular Songs by {artist1}**")
        st.table(popular_songs[popular_songs['artist'] == artist1][['title', 'views']])

    with col2:
        st.markdown(f"**Most Popular Songs by {artist2}**")
        st.table(popular_songs[popular_songs['artist'] == artist2][['title', 'views']])

    # Positive/Negative Songs
    st.markdown("### 🎵 Top Positive and Negative Songs")
    pos1, neg1 = get_top_songs_by_sentiment(comparison_data, artist1)
    pos2, neg2 = get_top_songs_by_sentiment(comparison_data, artist2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top Positive Songs by {artist1}**")
        st.table(pos1[['title', 'views', 'sentiment']])
        st.markdown(f"**Top Negative Songs by {artist1}**")
        st.table(neg1[['title', 'views', 'sentiment']])

    with col2:
        st.markdown(f"**Top Positive Songs by {artist2}**")
        st.table(pos2[['title', 'views', 'sentiment']])
        st.markdown(f"**Top Negative Songs by {artist2}**")
        st.table(neg2[['title', 'views', 'sentiment']])

    # Most Frequent Words
    st.markdown("### 🔡 Most Frequently Used Words")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Common Words by {artist1}**")
        freq_words1 = get_most_frequent_words(comparison_data, artist1)
        st.table(freq_words1)

    with col2:
        st.markdown(f"**Common Words by {artist2}**")
        freq_words2 = get_most_frequent_words(comparison_data, artist2)
        st.table(freq_words2)

    # Graph Section (After Data Tables)
    st.markdown("---")

    # Sentiment Over Time Plot
    st.markdown("### 📈 Sentiment Over Time")
    st.info("**Sentiment Analysis:** Measures the positivity/negativity of lyrics over the years.")
    sentiment_by_year = comparison_data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    st.line_chart(sentiment_by_year, color=["#32CD32", "#FF6347"])

    # Lexical Complexity Over Time Plot
    st.markdown("### 📚 Lexical Complexity Over Time")
    st.info("**Lexical Complexity:** Measures vocabulary diversity over the years for both artists.")
    
    comparison_data['lexical_diversity'] = comparison_data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split())
    )
    lexical_by_year = comparison_data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lexical_by_year, color=["#1E90FF", "#FFA500"])
