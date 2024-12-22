from sentiment_analysis import analyze_sentiment
import streamlit as st
import pandas as pd
from collections import Counter
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Named Entity Recognition (NER)
# -------------------------
@st.cache_data
def extract_entities(data):
    entities = []
    for lyrics in data['lyrics'].dropna():
        doc = nlp(lyrics)
        entities.extend([ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'GPE', 'ORG']])
    return pd.Series(entities).value_counts().head(10)


def display_entities(data):
    st.subheader("🔎 Named Entity Recognition – Key People and Places")
    entities = extract_entities(data)
    st.bar_chart(entities)

# -------------------------
# Topic Modeling (LDA)
# -------------------------
@st.cache_data
def perform_topic_modeling(data, num_topics=5):
    vectorizer = CountVectorizer(max_df=0.85, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(data['lyrics'].dropna())
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    return lda, vectorizer.get_feature_names_out()


def display_lda_topics(lda, feature_names, num_words=10):
    st.subheader("📚 LDA Topic Modeling – Discover Themes in Lyrics")
    for idx, topic in enumerate(lda.components_):
        st.markdown(f"**Topic {idx + 1}:**")
        st.write(", ".join([feature_names[i] for i in topic.argsort()[-num_words:]]))

# -------------------------
# Artist Comparison Section (Enhanced)
# -------------------------
def compare_artists(data):
    st.title("🎸 Artist Comparison")

    artist1, artist2 = data['artist'].unique()

    # Popular Songs
    st.markdown("### 🔥 Most Popular Songs (Top 3 by Views)")
    popular_songs = (
        data[data['views'] >= 1000]
        .sort_values(by='views', ascending=False)
        .groupby('artist')
        .head(3)
        .reset_index(drop=True)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Most Popular Songs by {artist1}**")
        st.table(popular_songs[popular_songs['artist'] == artist1][['title', 'views']])

    with col2:
        st.markdown(f"**Most Popular Songs by {artist2}**")
        st.table(popular_songs[popular_songs['artist'] == artist2][['title', 'views']])

    # Top Positive/Negative Songs
    st.markdown("### 🎵 Top Positive and Negative Songs (Min. 1000 Views)")
    pos1, neg1 = get_filtered_top_songs_by_sentiment(data, artist1)
    pos2, neg2 = get_filtered_top_songs_by_sentiment(data, artist2)

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

    # Graph Section
    st.markdown("---")

    # Sentiment Over Time (Yearly)
    st.markdown("### 📈 Sentiment Over Time (Yearly)")
    sentiment_by_year = data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    st.line_chart(sentiment_by_year, color=["#FF5733", "#4B0082"])

    # Lexical Complexity Over Time (Yearly)
    st.markdown("### 📚 Lexical Complexity Over Time (Yearly)")
    data['lexical_diversity'] = data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split())
    )
    lexical_by_year = data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lexical_by_year, color=["#1E90FF", "#FFD700"])

    # Named Entity Recognition (NER) Visualization
    display_entities(data)

    # Topic Modeling Visualization
    lda_model, feature_names = perform_topic_modeling(data)
    display_lda_topics(lda_model, feature_names)
