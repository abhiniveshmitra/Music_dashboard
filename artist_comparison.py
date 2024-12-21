import streamlit as st
import pandas as pd
from collections import Counter
from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment

# Manual stopwords list (no NLTK required)
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not"
])

# Lexical Complexity Analysis
def lexical_complexity_analysis(data, artist):
    artist_data = data[data['artist'] == artist]
    
    # Lexical diversity: unique words / total words
    artist_data['lexical_diversity'] = artist_data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()))
    avg_diversity = artist_data['lexical_diversity'].mean()
    
    return artist_data[['year', 'lexical_diversity']].groupby('year').mean(), avg_diversity


# Most Frequently Used Words (Filter Short Words + Stopwords)
def get_most_frequent_words(data, artist, top_n=10):
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()

    # Remove words with <= 4 letters and stopwords
    filtered_words = [
        word.lower() for word in words 
        if len(word) > 4 and word.lower() not in stop_words
    ]

    most_common = Counter(filtered_words).most_common(top_n)
    
    return pd.DataFrame(most_common, columns=['Word', 'Frequency'])


# Compare Two Artists
def compare_artists(data):
    # Centered and Enlarged Title
    st.markdown("<h2 style='text-align: center;'>🎤 Compare Two Artists</h2>", unsafe_allow_html=True)
    
    # Artist Selection
    artist_options = data['artist'].unique()
    col1, col2 = st.columns(2)
    with col1:
        artist1 = st.selectbox("Select First Artist", artist_options, key='artist1')
    with col2:
        artist2 = st.selectbox("Select Second Artist", artist_options, key='artist2')

    if artist1 and artist2:
        comparison_data = data[data['artist'].isin([artist1, artist2])]
        
        # Ensure sentiment analysis
        comparison_data = analyze_sentiment(comparison_data)

        # Filter columns and keep necessary fields
        comparison_data = comparison_data[['title', 'artist', 'sentiment', 'views', 'year', 'lyrics']]

        artist_years = comparison_data['year'].unique()

        col1, col2 = st.columns(2)

        # Artist 1 Section
        with col1:
            st.write(f"## 🎵 {artist1}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist1]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            # Lexical Complexity Analysis
            lexical_trend, lexical_complexity = lexical_complexity_analysis(comparison_data, artist1)
            st.write(f"**Lexical Complexity:** {lexical_complexity:.2f}")
            st.line_chart(lexical_trend, color=["#FF5733"])

            st.write("### Most Frequently Used Words")
            freq_words = get_most_frequent_words(comparison_data, artist1)
            st.table(freq_words)

            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist1)
            st.write("### Top Positive Songs")
            st.table(top_pos[['title', 'sentiment', 'views']].reset_index(drop=True))
            st.write("### Top Negative Songs")
            st.table(top_neg[['title', 'sentiment', 'views']].reset_index(drop=True))

        # Artist 2 Section
        with col2:
            st.write(f"## 🎵 {artist2}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist2]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            lexical_trend, lexical_complexity = lexical_complexity_analysis(comparison_data, artist2)
            st.write(f"**Lexical Complexity:** {lexical_complexity:.2f}")
            st.line_chart(lexical_trend, color=["#4B9CD3"])

            st.write("### Most Frequently Used Words")
            freq_words = get_most_frequent_words(comparison_data, artist2)
            st.table(freq_words)

            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist2)
            st.write("### Top Positive Songs")
            st.table(top_pos[['title', 'sentiment', 'views']].reset_index(drop=True))
            st.write("### Top Negative Songs")
            st.table(top_neg[['title', 'sentiment', 'views']].reset_index(drop=True))

        # Sentiment Over Time Visualization
        st.subheader("📈 Sentiment Comparison Over Time")
        sentiment_by_year = comparison_data.groupby(['year', 'artist'])['sentiment'].mean().unstack()
        st.line_chart(sentiment_by_year.loc[artist_years], color=["#FF5733", "#4B9CD3"])

        # Popularity Over Time (Views)
        st.subheader("📊 Popularity Over Time (by Views)")
        views_by_year = comparison_data.groupby(['year', 'artist'])['views'].sum().unstack()
        st.bar_chart(views_by_year.loc[artist_years], color=["#1F77B4", "#FF7F0E"])


# Sentiment Explanation Function
def explain_sentiment(sentiment_score):
    if sentiment_score > 0.5:
        st.success("😊 Highly Positive – Uplifting and joyful songs.")
    elif sentiment_score > 0:
        st.info("🙂 Positive – A generally optimistic tone.")
    elif sentiment_score < -0.5:
        st.error("😞 Highly Negative – Sad or dark themes.")
    else:
        st.warning("😐 Neutral – Mixed or balanced sentiment.")
