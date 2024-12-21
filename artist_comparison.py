import streamlit as st
import pandas as pd
from sentiment_analysis import analyze_sentiment, get_top_songs_by_sentiment

def lexical_complexity_analysis(data, artist):
    artist_data = data[data['artist'] == artist]
    artist_data['lexical_diversity'] = artist_data['lyrics'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()))
    return artist_data['lexical_diversity'].mean()

def get_album_info(data, artist):
    artist_data = data[data['artist'] == artist]
    
    if 'album' in artist_data.columns:
        debut_album = artist_data.loc[artist_data['year'].idxmin()][['album', 'year']]
        most_popular_album = artist_data.groupby('album')['views'].sum().idxmax()
    else:
        debut_album = pd.Series({'album': 'Unknown', 'year': 'N/A'})
        most_popular_album = 'Unknown'
    
    return debut_album, most_popular_album

def explain_sentiment(sentiment_score):
    if sentiment_score > 0.5:
        st.success("😊 Highly Positive – Uplifting and joyful songs.")
    elif sentiment_score > 0:
        st.info("🙂 Positive – A generally optimistic tone.")
    elif sentiment_score < -0.5:
        st.error("😞 Highly Negative – Sad or dark themes.")
    else:
        st.warning("😐 Neutral – Mixed or balanced sentiment.")

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
        
        # Ensure sentiment analysis
        comparison_data = analyze_sentiment(comparison_data)

        # Filter existing columns only
        columns_to_keep = ['title', 'artist', 'sentiment', 'views', 'year', 'lyrics']
        if 'album' in comparison_data.columns:
            columns_to_keep.append('album')
        
        comparison_data = comparison_data[columns_to_keep]

        artist_years = comparison_data['year'].unique()

        col1, col2 = st.columns(2)

        # Artist 1 Section
        with col1:
            st.write(f"## 🎵 {artist1}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist1]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            lexical_complexity = lexical_complexity_analysis(comparison_data, artist1)
            st.write(f"**Lexical Complexity:** {lexical_complexity:.2f}")

            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist1)
            st.write("### Top Positive Songs")
            st.table(top_pos[['title', 'sentiment', 'views']].reset_index(drop=True))
            st.write("### Top Negative Songs")
            st.table(top_neg[['title', 'sentiment', 'views']].reset_index(drop=True))

            debut_album, most_popular_album = get_album_info(comparison_data, artist1)
            st.write(f"**Debut Album:** {debut_album['album']} ({debut_album['year']})")
            st.write(f"**Most Popular Album:** {most_popular_album}")

        # Artist 2 Section
        with col2:
            st.write(f"## 🎵 {artist2}")
            avg_sentiment = comparison_data[comparison_data['artist'] == artist2]['sentiment'].mean()
            st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")
            explain_sentiment(avg_sentiment)

            lexical_complexity = lexical_complexity_analysis(comparison_data, artist2)
            st.write(f"**Lexical Complexity:** {lexical_complexity:.2f}")

            top_pos, top_neg = get_top_songs_by_sentiment(comparison_data, artist2)
            st.write("### Top Positive Songs")
            st.table(top_pos[['title', 'sentiment', 'views']].reset_index(drop=True))
            st.write("### Top Negative Songs")
            st.table(top_neg[['title', 'sentiment', 'views']].reset_index(drop=True))

            debut_album, most_popular_album = get_album_info(comparison_data, artist2)
            st.write(f"**Debut Album:** {debut_album['album']} ({debut_album['year']})")
            st.write(f"**Most Popular Album:** {most_popular_album}")
