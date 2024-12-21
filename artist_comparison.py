import streamlit as st
import pandas as pd
from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment

def lexical_complexity_analysis(data, artist):
    artist_data = data[data['artist'] == artist]
    
    # Lexical diversity: unique words / total words
    artist_data['lexical_diversity'] = artist_data['lyrics'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()))
    avg_diversity = artist_data['lexical_diversity'].mean()
    
    return avg_diversity

def get_album_info(data, artist):
    artist_data = data[data['artist'] == artist]
    
    # Debut album (earliest year)
    debut_album = artist_data.loc[artist_data['year'].idxmin()][['album', 'year']]
    
    # Most popular album (highest total views)
    most_popular_album = artist_data.groupby('album')['views'].sum().idxmax()
    
    return debut_album, most_popular_album

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
        
        # Ensure sentiment analysis for comparison data
        comparison_data = analyze_sentiment(comparison_data)

        # Filter columns and remove full lyrics, unnecessary fields
        comparison_data = comparison_data[['title', 'artist', 'sentiment', 'views', 'year', 'lyrics', 'album']]

        # Get years when artists were active
        artist_years = comparison_data['year'].unique()

        col1, col2 = st.columns(2)

        # Artist 1 Section
        with col1:
            st.write(f"## 🎵 {artist1}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist1]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            # Lexical Complexity
            lexical_complexity = lexical_complexity_analysis(comparison_data, artist1)
            st.write(f"**Lexical Complexity:** {lexical_complexity:.2f}")

            # Top Songs by Sentiment
            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist1)
            st.write("### Top Positive Songs")
            st.table(top_pos[['title', 'sentiment', 'views']].reset_index(drop=True))
            st.write("### Top Negative Songs")
            st.table(top_neg[['title', 'sentiment', 'views']].reset_index(drop=True))

            # Album Info
            debut_album, most_popular_album = get_album_info(comparison_data, artist1)
            st.write(f"**Debut Album:** {debut_album['album']} ({int(debut_album['year'])})")
            st.write(f"**Most Popular Album:** {most_popular_album}")

        # Artist 2 Section
        with col2:
            st.write(f"## 🎵 {artist2}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist2]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            lexical_complexity = lexical_complexity_analysis(comparison_data, artist2)
            st.write(f"**Lexical Complexity:** {lexical_complexity:.2f}")

            # Top Songs by Sentiment
            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist2)
            st.write("### Top Positive Songs")
            st.table(top_pos[['title', 'sentiment', 'views']].reset_index(drop=True))
            st.write("### Top Negative Songs")
            st.table(top_neg[['title', 'sentiment', 'views']].reset_index(drop=True))

            debut_album, most_popular_album = get_album_info(comparison_data, artist2)
            st.write(f"**Debut Album:** {debut_album['album']} ({int(debut_album['year'])})")
            st.write(f"**Most Popular Album:** {most_popular_album}")

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
