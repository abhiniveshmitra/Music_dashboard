import streamlit as st
import pandas as pd
from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment

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
        comparison_data = comparison_data[['title', 'artist', 'sentiment', 'views', 'year']]

        # Get data relevant to artist active years
        artist_years = comparison_data['year'].unique()

        # Layout for Side by Side Comparison
        col1, col2 = st.columns(2)

        # Artist 1 Section
        with col1:
            st.write(f"## 🎵 {artist1}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist1]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            # Top Songs by Sentiment
            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist1)
            st.write("### Top Positive Songs")
            st.table(top_pos[['title', 'sentiment', 'views']].reset_index(drop=True))
            st.write("### Top Negative Songs")
            st.table(top_neg[['title', 'sentiment', 'views']].reset_index(drop=True))

            # Most Popular Song
            most_popular = comparison_data[comparison_data['artist'] == artist1].nlargest(1, 'views')
            st.write("### Most Popular Song")
            st.table(most_popular[['title', 'views', 'year']].reset_index(drop=True))

        # Artist 2 Section
        with col2:
            st.write(f"## 🎵 {artist2}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist2]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            # Top Songs by Sentiment
            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist2)
            st.write("### Top Positive Songs")
            st.table(top_pos[['title', 'sentiment', 'views']].reset_index(drop=True))
            st.write("### Top Negative Songs")
            st.table(top_neg[['title', 'sentiment', 'views']].reset_index(drop=True))

            most_popular = comparison_data[comparison_data['artist'] == artist2].nlargest(1, 'views')
            st.write("### Most Popular Song")
            st.table(most_popular[['title', 'views', 'year']].reset_index(drop=True))

        # Sentiment Over Time Visualization
        st.subheader("📈 Sentiment Comparison Over Time")
        sentiment_by_year = comparison_data.groupby(['year', 'artist'])['sentiment'].mean().unstack()
        st.line_chart(sentiment_by_year.loc[artist_years])

        # Popularity Over Time (Views)
        st.subheader("📊 Popularity Over Time (by Views)")
        views_by_year = comparison_data.groupby(['year', 'artist'])['views'].sum().unstack()
        st.bar_chart(views_by_year.loc[artist_years])

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
