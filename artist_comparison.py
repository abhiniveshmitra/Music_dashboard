import streamlit as st
import pandas as pd
from collections import Counter

# Updated stopwords list
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every", "little"
])

def get_top_songs_by_sentiment(data, artist, top_n=3):
    artist_data = data[data['artist'] == artist]
    top_positive = artist_data.sort_values(by='sentiment', ascending=False).head(top_n)
    top_negative = artist_data.sort_values(by='sentiment').head(top_n)
    return top_positive[['title', 'sentiment']], top_negative[['title', 'sentiment']]

def get_most_frequent_words(data, artist, top_n=10):
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()
    filtered_words = [
        word.lower() for word in words
        if len(word) > 4 and word.lower() not in stop_words
    ]
    most_common = Counter(filtered_words).most_common(top_n)
    return pd.DataFrame(most_common, columns=['Word', 'Frequency'])

def compare_artists(data):
    st.title("🎸 Compare Two Rock Artists")

    # Dropdowns to select artists
    artist1 = st.selectbox("Select First Artist", data['artist'].unique())
    artist2 = st.selectbox("Select Second Artist", data['artist'].unique(), index=1)

    # Filter the data
    comparison_data = data[(data['artist'] == artist1) | (data['artist'] == artist2)]

    # Explain the Comparison
    st.markdown("### This section compares two artists based on sentiment, word frequency, and lexical complexity.")

    # Sentiment Comparison Over Time
    sentiment_by_year = comparison_data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    artist_years = comparison_data['year'].unique()

    st.markdown("#### 📈 Sentiment Over Time")
    st.info("**Sentiment Analysis:** Measures the positivity/negativity of lyrics over the years. Higher scores indicate more positive songs.")
    st.line_chart(sentiment_by_year.loc[artist_years], color=["#32CD32", "#FF6347"])

    # Most Popular Songs
    st.markdown("#### 🔥 Most Popular Songs (Top 3 by Views)")
    st.info("**Popularity Analysis:** Displays the top 3 songs by views for each artist.")
    popular_songs = comparison_data.sort_values(by='views', ascending=False).groupby('artist').head(3)
    st.dataframe(popular_songs[['title', 'artist', 'views']])

    # Positive/Negative Songs
    st.markdown("#### 🎵 Top Positive and Negative Songs")
    st.info("**Emotional Spectrum:** Shows the songs with the most positive and negative sentiment scores.")
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
    st.info("**Lexical Complexity:** Evaluates the diversity of vocabulary used by each artist.")
    comparison_data['lexical_diversity'] = comparison_data['lyrics'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()))
    lexical_comparison = comparison_data.groupby('artist')['lexical_diversity'].mean()
    st.bar_chart(lexical_comparison)

    # Most Frequent Words
    st.markdown("#### 🔡 Most Frequently Used Words")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Common Words by {artist1}**")
        freq_words1 = get_most_frequent_words(comparison_data, artist1)
        st.table(freq_words1)
    
    with col2:
        st.markdown(f"**Common Words by {artist2}**")
        freq_words2 = get_most_frequent_words(comparison_data, artist2)
        st.table(freq_words2)
