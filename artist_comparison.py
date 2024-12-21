from sentiment_analysis import analyze_sentiment
import streamlit as st
import pandas as pd
from collections import Counter

# Updated Stopwords List
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every", "little", "you’re", "don’t", "ain’t", "can’t"
])

# Convert Year to Decade
def to_decade(year):
    return (year // 10) * 10


# Frequent Words Analysis (1-based indexing)
def get_most_frequent_words(data, artist, top_n=10):
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()
    filtered_words = [word.lower() for word in words if len(word) > 4 and word.lower() not in stop_words]
    most_common = Counter(filtered_words).most_common(top_n)
    df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
    df.index = df.index + 1  # 1-based indexing for frequent words only
    return df


# Top Songs by Sentiment (Minimum Views Filter)
def get_filtered_top_songs_by_sentiment(data, artist, top_n=3):
    filtered_data = data[(data['artist'] == artist) & (data['views'] >= 1000)]

    if filtered_data.empty:
        return pd.DataFrame(columns=['title', 'sentiment', 'views']), pd.DataFrame(columns=['title', 'sentiment', 'views'])

    top_positive = filtered_data.sort_values(by='sentiment', ascending=False).head(top_n).reset_index(drop=True)
    top_negative = filtered_data.sort_values(by='sentiment').head(top_n).reset_index(drop=True)

    return top_positive[['title', 'sentiment', 'views']], top_negative[['title', 'sentiment', 'views']]


# Artist Comparison (Force Two Artists)
def compare_artists(data):
    st.title("🎸 Compare Two Rock Artists")

    # Sidebar - Decade Filter
    st.sidebar.header("Filter by Decade")
    data['decade'] = data['year'].apply(to_decade)
    available_decades = sorted(data['decade'].unique())
    selected_decades = st.sidebar.multiselect("Select Decades", available_decades, default=available_decades)

    # Filter Data by Selected Decades
    filtered_data = data[data['decade'].isin(selected_decades)]

    # Select Two Artists Only
    artist_options = filtered_data['artist'].unique()
    selected_artists = st.multiselect("Select Two Artists to Compare", artist_options, default=artist_options[:2], max_selections=2)

    if len(selected_artists) != 2:
        st.warning("Please select exactly **two artists** for comparison.")
        return

    artist1, artist2 = selected_artists

    # Filter Data for Comparison
    comparison_data = filtered_data[filtered_data['artist'].isin([artist1, artist2])]

    # Apply Sentiment Analysis if Missing
    if 'sentiment' not in comparison_data.columns:
        comparison_data = analyze_sentiment(comparison_data)

    # Display Popular Songs (Minimum 1000 views)
    st.markdown("### 🔥 Most Popular Songs (Top 3 by Views)")
    popular_songs = (
        comparison_data[comparison_data['views'] >= 1000]
        .sort_values(by='views', ascending=False)
        .groupby('artist')
        .head(3)
        .reset_index(drop=True)  # Drop index completely
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Most Popular Songs by {artist1}**")
        st.table(popular_songs[popular_songs['artist'] == artist1][['title', 'views']])

    with col2:
        st.markdown(f"**Most Popular Songs by {artist2}**")
        st.table(popular_songs[popular_songs['artist'] == artist2][['title', 'views']])

    # Positive/Negative Songs (Min. 1000 Views)
    st.markdown("### 🎵 Top Positive and Negative Songs (Min. 1000 Views)")
    pos1, neg1 = get_filtered_top_songs_by_sentiment(comparison_data, artist1)
    pos2, neg2 = get_filtered_top_songs_by_sentiment(comparison_data, artist2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top Positive Songs by {artist1}**")
        st.table(pos1)
        st.markdown(f"**Top Negative Songs by {artist1}**")
        st.table(neg1)

    with col2:
        st.markdown(f"**Top Positive Songs by {artist2}**")
        st.table(pos2)
        st.markdown(f"**Top Negative Songs by {artist2}**")
        st.table(neg2)

    # Graph Section (After Data Tables)
    st.markdown("---")

    # Sentiment Over Time Plot (Yearly)
    st.markdown("### 📈 Sentiment Over Time (Yearly)")
    sentiment_by_year = comparison_data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    st.line_chart(sentiment_by_year)

    # Lexical Complexity Over Time Plot (Yearly)
    st.markdown("### 📚 Lexical Complexity Over Time (Yearly)")
    comparison_data['lexical_diversity'] = comparison_data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split())
    )
    lexical_by_year = comparison_data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lexical_by_year)
