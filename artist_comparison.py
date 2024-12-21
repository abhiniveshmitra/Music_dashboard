import streamlit as st
import pandas as pd
from collections import Counter
from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment

# Manual stopwords list (no NLTK required)
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every" ,"don't", "you’re", "ain’t","there’s"
])


# Dynamic color generator (Neon Green & Red)
def generate_colors(n):
    base_colors = ["#39FF14", "#FF073A"]  # Neon Green & Red
    return base_colors[:n] + ["#6A0DAD"] * (n - len(base_colors))

# Lexical Complexity Analysis
def lexical_complexity_analysis(data, artist):
    artist_data = data[data['artist'] == artist]
    artist_data['lexical_diversity'] = artist_data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()))
    avg_diversity = artist_data['lexical_diversity'].mean()

    # Fill missing years for smooth chart
    return artist_data[['year', 'lexical_diversity']].groupby('year').mean().reindex(
        range(artist_data['year'].min(), artist_data['year'].max() + 1), fill_value=None).interpolate(), avg_diversity

# Most Frequently Used Words (Filter Short Words + Stopwords)
def get_most_frequent_words(data, artist, top_n=10):
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()
    filtered_words = [
        word.lower() for word in words 
        if len(word) > 4 and word.lower() not in stop_words
    ]
    most_common = Counter(filtered_words).most_common(top_n)
    return pd.DataFrame(most_common, columns=['Word', 'Frequency'])

# Compare Two Artists
def compare_artists(data):
    st.markdown("<h2 style='text-align: center;'>🎤 Compare Two Artists</h2>", unsafe_allow_html=True)
    
    # Artist Selection (Force Two Artists)
    artist_options = data['artist'].unique()
    selected_artists = st.multiselect("Select Two Artists", artist_options, default=artist_options[:2], max_selections=2)

    if len(selected_artists) == 2:
        artist1, artist2 = selected_artists
        comparison_data = data[data['artist'].isin([artist1, artist2])]
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

            st.write("### Most Frequently Used Words")
            freq_words = get_most_frequent_words(comparison_data, artist1)
            st.table(freq_words)

            # Top 3 Most Popular Songs
            st.write("### Most Popular Songs (Top 3 by Views)")
            most_popular = comparison_data[comparison_data['artist'] == artist1].nlargest(3, 'views')
            st.table(most_popular[['title', 'views', 'year']].reset_index(drop=True))

        # Artist 2 Section
        with col2:
            st.write(f"## 🎵 {artist2}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist2]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            freq_words = get_most_frequent_words(comparison_data, artist2)
            st.write("### Most Frequently Used Words")
            st.table(freq_words)

            most_popular = comparison_data[comparison_data['artist'] == artist2].nlargest(3, 'views')
            st.write("### Most Popular Songs (Top 3 by Views)")
            st.table(most_popular[['title', 'views', 'year']].reset_index(drop=True))

        # Combined Lexical Complexity Plot
        st.subheader("📈 Lexical Complexity Over Time (Comparison)")
        lexical_trend1, _ = lexical_complexity_analysis(comparison_data, artist1)
        lexical_trend2, _ = lexical_complexity_analysis(comparison_data, artist2)

        combined_complexity = pd.concat([lexical_trend1, lexical_trend2], axis=1)
        combined_complexity.columns = [artist1, artist2]
        st.line_chart(combined_complexity, color=["#39FF14", "#FF073A"])

        # Sentiment Over Time Visualization
        st.subheader("📈 Sentiment Comparison Over Time")
        sentiment_by_year = comparison_data.groupby(['year', 'artist'])['sentiment'].mean().unstack()
        colors = generate_colors(len(sentiment_by_year.columns))
        st.line_chart(sentiment_by_year.loc[artist_years], color=colors)

        # Popularity Over Time (Views)
        st.subheader("📊 Popularity Over Time (by Views)")
        views_by_year = comparison_data.groupby(['year', 'artist'])['views'].sum().unstack()
        colors = generate_colors(len(views_by_year.columns))
        st.bar_chart(views_by_year.loc[artist_years], color=colors)

def explain_sentiment(sentiment_score):
    if sentiment_score > 0.5:
        st.success("😊 Highly Positive – Uplifting and joyful songs.")
    elif sentiment_score > 0:
        st.info("🙂 Positive – A generally optimistic tone.")
    elif sentiment_score < -0.5:
        st.error("😞 Highly Negative – Sad or dark themes.")
    else:
        st.warning("😐 Neutral – Mixed or balanced sentiment.")
