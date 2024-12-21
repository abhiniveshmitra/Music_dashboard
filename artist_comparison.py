import streamlit as st
import pandas as pd
from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment

# Cache for performance optimization
@st.cache_data
def generate_wordcloud(data, artist):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    if artist_lyrics:
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(artist_lyrics)
        st.image(wordcloud.to_array(), use_column_width=True)
    else:
        st.warning(f"No lyrics available for {artist}")

# Get debut album, most popular albums, and famous songs
def get_artist_info(data, artist):
    artist_data = data[data['artist'] == artist]

    # Most Popular Songs (Top 3 by Views)
    top_songs = artist_data.sort_values(by='views', ascending=False).head(3)[['title', 'views']]
    
    # Debut Album (Earliest Year)
    debut_album = artist_data.loc[artist_data['year'].idxmin()][['title', 'year']]

    # Most Popular Album by Views
    popular_album = artist_data.groupby('album')['views'].sum().idxmax()

    # Famous Lyrics Snippet (Top Songs)
    famous_lyrics = artist_data.nlargest(1, 'views')['lyrics'].iloc[0].split('\n')[:2]
    
    return top_songs, debut_album, popular_album, famous_lyrics

def compare_artists(data):
    st.subheader("🎤 Compare Two Artists Side by Side")
    
    # Artist Selection
    artist_options = data['artist'].unique()
    col1, col2 = st.columns(2)
    with col1:
        artist1 = st.selectbox("Select First Artist", artist_options, key='artist1')
    with col2:
        artist2 = st.selectbox("Select Second Artist", artist_options, key='artist2')

    if artist1 and artist2:
        comparison_data = data[data['artist'].isin([artist1, artist2])]
        
        # Ensure Sentiment Analysis
        comparison_data = analyze_sentiment(comparison_data)

        # Layout for Comparison
        col1, col2 = st.columns(2)

        # Artist 1 Info
        with col1:
            st.write(f"## 🎵 {artist1}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist1]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            # Top Songs by Sentiment
            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist1)
            st.write("### Top Positive Songs")
            st.table(top_pos)
            st.write("### Top Negative Songs")
            st.table(top_neg)
            
            # Word Cloud
            st.write("### Word Cloud")
            generate_wordcloud(data, artist1)

            # Additional Insights
            top_songs, debut_album, popular_album, lyrics = get_artist_info(data, artist1)
            st.write("### 🎵 Famous Songs")
            st.table(top_songs)
            st.write(f"**Debut Album:** {debut_album['title']} ({int(debut_album['year'])})")
            st.write(f"**Most Popular Album:** {popular_album}")
            st.write("**Famous Lyrics Snippet:**")
            for line in lyrics:
                st.write(f"_\"{line}\"_")

        # Artist 2 Info
        with col2:
            st.write(f"## 🎵 {artist2}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist2]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            # Top Songs by Sentiment
            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist2)
            st.write("### Top Positive Songs")
            st.table(top_pos)
            st.write("### Top Negative Songs")
            st.table(top_neg)
            
            # Word Cloud
            st.write("### Word Cloud")
            generate_wordcloud(data, artist2)

            # Additional Insights
            top_songs, debut_album, popular_album, lyrics = get_artist_info(data, artist2)
            st.write("### 🎵 Famous Songs")
            st.table(top_songs)
            st.write(f"**Debut Album:** {debut_album['title']} ({int(debut_album['year'])})")
            st.write(f"**Most Popular Album:** {popular_album}")
            st.write("**Famous Lyrics Snippet:**")
            for line in lyrics:
                st.write(f"_\"{line}\"_")

# Explain sentiment
def explain_sentiment(sentiment_score):
    if sentiment_score > 0.5:
        st.success("😊 Highly Positive – Uplifting and joyful songs.")
    elif sentiment_score > 0:
        st.info("🙂 Positive – Lightly optimistic tone.")
    elif sentiment_score < -0.5:
        st.error("😞 Highly Negative – Sad or dark themes.")
    else:
        st.warning("😐 Neutral – Mixed sentiment.")
