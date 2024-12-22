import streamlit as st
import pandas as pd
from collections import Counter
from sentiment_analysis import analyze_sentiment
import subprocess
import spacy
import text2emotion as te
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# ------------------------------------------------------
# Try loading the spaCy model; if not found, download it
# ------------------------------------------------------
@st.cache_data
def load_spacy_model():
    try:
        nlp_model = spacy.load("en_core_web_sm")
        return nlp_model
    except OSError:
        st.warning("SpaCy model 'en_core_web_sm' not found, auto-downloading...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp_model = spacy.load("en_core_web_sm")
        return nlp_model

nlp = load_spacy_model()

# Stopwords List
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every", "little", "you’re", "don’t", "ain’t", "can’t"
])

def get_most_frequent_words(data, artist, top_n=10):
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()
    filtered_words = [word.lower() for word in words if len(word) > 4 and word.lower() not in stop_words]
    most_common = Counter(filtered_words).most_common(top_n)
    df = pd.DataFrame(most_common, columns=['Word', 'Frequency'])
    df.index = df.index + 1  # 1-based indexing for frequent words only
    return df

def get_filtered_top_songs_by_sentiment(data, artist, top_n=3):
    filtered_data = data[(data['artist'] == artist) & (data['views'] >= 1000)]

    if filtered_data.empty:
        return pd.DataFrame(columns=['title', 'sentiment', 'views']), pd.DataFrame(columns=['title', 'sentiment', 'views'])

    top_positive = filtered_data.sort_values(by='sentiment', ascending=False).head(top_n).reset_index(drop=True)
    top_negative = filtered_data.sort_values(by='sentiment').head(top_n).reset_index(drop=True)

    return top_positive[['title', 'sentiment', 'views']], top_negative[['title', 'sentiment', 'views']]

def extract_named_entities(text):
    """
    Extract named entities from a text string using spaCy.
    Returns a frequency count of entity texts.
    """
    if not nlp:
        return {}
    doc = nlp(text)
    entity_texts = [ent.text for ent in doc.ents]
    return Counter(entity_texts)

def get_top_5_entities_for_artist(data, artist):
    """
    Combine all the artist's lyrics into one big string, then extract top 5 entities by frequency.
    """
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    entity_counts = extract_named_entities(artist_lyrics)
    if not entity_counts:
        return []
    return entity_counts.most_common(5)

def get_topics_for_artist(data, artist, num_topics=5):
    """
    Use gensim LDA to get topics for an artist's combined lyrics.
    Return top words for each of the discovered topics.
    """
    artist_lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()
    # Combine all lyrics, then split into tokens
    tokens = []
    for lyric in artist_lyrics_list:
        # Simple tokenization + lower + stopword removal
        lyric_tokens = [
            w.lower() for w in lyric.split()
            if w.lower() not in stop_words and len(w) > 2
        ]
        tokens.append(lyric_tokens)

    if not tokens:
        return []

    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    try:
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=42, passes=1)
    except ValueError:
        return []

    topics = lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)

    topic_list = []
    for topic_num, word_probs in topics:
        words_only = [wp[0] for wp in word_probs]
        topic_list.append((f"Topic {topic_num}", words_only))

    return topic_list

def get_emotions_for_artist(data, artist):
    """
    Use text2emotion to extract overall (aggregated) emotion from all lyrics of a given artist.
    Return the top 3 emotions in descending order.
    """
    artist_lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()
    if not artist_lyrics_list:
        return []

    cumulative_emotions = Counter()
    
    for lyric in artist_lyrics_list:
        emotion_scores = te.get_emotion(lyric)
        for emo, score in emotion_scores.items():
            cumulative_emotions[emo] += score

    if not cumulative_emotions:
        return []

    total = sum(cumulative_emotions.values())
    emotion_percentages = [(emo, round((val / total) * 100, 2)) for emo, val in cumulative_emotions.items()]
    emotion_percentages.sort(key=lambda x: x[1], reverse=True)
    return emotion_percentages[:3]

def compare_artists(data):
    st.title("🎸 Artist Comparison")

    # Expect exactly 2 unique artists in the data subset
    unique_artists = data['artist'].unique()
    if len(unique_artists) < 2:
        st.error("Not enough artists selected.")
        return

    artist1, artist2 = unique_artists[0], unique_artists[1]

    # --------------------
    # Popular Songs
    # --------------------
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

    # --------------------
    # Top Positive/Negative Songs
    # --------------------
    st.markdown("### 🎵 Top Positive and Negative Songs (Min. 1000 Views)")
    pos
