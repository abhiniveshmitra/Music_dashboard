import streamlit as st
import subprocess
import spacy
import text2emotion as te
import pandas as pd
from collections import Counter
from sentiment_analysis import analyze_sentiment

# Gensim imports that may trigger the SciPy "triu" mismatch
from gensim.corpora import Dictionary
from gensim.models import LdaModel


# --------------------------------------------------------------------
# 1) CACHED: spaCy loader - auto-download "en_core_web_sm" if missing
# --------------------------------------------------------------------
@st.cache_data
def load_spacy_model():
    """
    Attempt to load spaCy's en_core_web_sm.
    If not present, install it, then load again.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("SpaCy model 'en_core_web_sm' not found, auto-downloading...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# --------------------------------------------------------------------
# Stopwords
# --------------------------------------------------------------------
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every", "little", "you’re", "don’t", "ain’t", "can’t"
])

# --------------------------------------------------------------------
# 2) Optional: Cache frequent words. (Small function, might not need.)
# --------------------------------------------------------------------
@st.cache_data
def get_most_frequent_words(data, artist, top_n=10):
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()
    filtered_words = [
        word.lower()
        for word in words
        if len(word) > 4 and word.lower() not in stop_words
    ]
    counts = Counter(filtered_words).most_common(top_n)
    df = pd.DataFrame(counts, columns=['Word', 'Frequency'])
    df.index = df.index + 1  # 1-based indexing for display
    return df

# --------------------------------------------------------------------
# 3) Top songs by sentiment (no heavy caching needed)
# --------------------------------------------------------------------
def get_filtered_top_songs_by_sentiment(data, artist, top_n=3):
    filtered_data = data[(data['artist'] == artist) & (data['views'] >= 1000)]
    if filtered_data.empty:
        return (
            pd.DataFrame(columns=['title', 'sentiment', 'views']),
            pd.DataFrame(columns=['title', 'sentiment', 'views'])
        )

    top_positive = filtered_data.sort_values(by='sentiment', ascending=False).head(top_n).reset_index(drop=True)
    top_negative = filtered_data.sort_values(by='sentiment').head(top_n).reset_index(drop=True)

    return top_positive[['title', 'sentiment', 'views']], top_negative[['title', 'sentiment', 'views']]

# --------------------------------------------------------------------
# 4) Named Entity Recognition
# --------------------------------------------------------------------
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

@st.cache_data
def get_top_5_entities_for_artist(data, artist):
    """
    Combine the artist's lyrics, then extract top 5 entities by frequency.
    """
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    entity_counts = extract_named_entities(artist_lyrics)
    return entity_counts.most_common(5) if entity_counts else []

# --------------------------------------------------------------------
# 5) Topic Modeling with Gensim LDA
# --------------------------------------------------------------------
@st.cache_data
def get_topics_for_artist(data, artist, num_topics=5):
    """
    Use gensim LDA to get topics for an artist's combined lyrics.
    Return top words for each discovered topic.
    """
    artist_lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()

    # Tokenize
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

    # Create dictionary + corpus
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    try:
        lda_model = LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_state=42,
            passes=1
        )
    except ValueError:
        # In case there's not enough data
        return []

    topics = lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)
    topic_list = []
    for topic_num, word_probs in topics:
        words_only = [wp[0] for wp in word_probs]
        topic_list.append((f"Topic {topic_num}", words_only))

    return topic_list

# --------------------------------------------------------------------
# 6) Emotion Detection
# --------------------------------------------------------------------
@st.cache_data
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
    emotion_percentages = [
        (emo, round((val / total) * 100, 2))
        for emo, val in cumulative_emotions.items()
    ]
    emotion_percentages.sort(key=lambda x: x[1], reverse=True)
    return emotion_percentages[:3]

# --------------------------------------------------------------------
# 7) Main Comparison Function
# --------------------------------------------------------------------
def compare_artists(data):
    st.title("🎸 Artist Comparison")

    # We expect exactly 2 unique artists in the filtered data
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
    # Top Positive/Negative
    # --------------------
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

    # --------------------
    # Sentiment & Lexical Diversity Over Time
    # --------------------
    st.markdown("---")
    st.markdown("### 📈 Sentiment Over Time (Yearly)")

    sentiment_by_year = data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    st.line_chart(sentiment_by_year, use_container_width=True)

    st.markdown("### 📚 Lexical Complexity Over Time (Yearly)")
    data['lexical_diversity'] = data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split())
        if x and len(str(x).split()) > 0 else 0
    )
    lexical_by_year = data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lexical_by_year, use_container_width=True)

    # --------------------
    # Named Entity Recognition
    # --------------------
    st.markdown("---")
    st.markdown("### 🏷 Named Entities (Top 5)")
    col1, col2 = st.columns(2)
    entities1 = get_top_5_entities_for_artist(data, artist1)
    entities2 = get_top_5_entities_for_artist(data, artist2)

    with col1:
        st.write(f"**Top Entities for {artist1}**")
        if entities1:
            df_entities1 = pd.DataFrame(entities1, columns=["Entity", "Count"])
            st.table(df_entities1)
        else:
            st.write("No entities found or spaCy model not loaded.")

    with col2:
        st.write(f"**Top Entities for {artist2}**")
        if entities2:
            df_entities2 = pd.DataFrame(entities2, columns=["Entity", "Count"])
            st.table(df_entities2)
        else:
            st.write("No entities found or spaCy model not loaded.")

    # --------------------
    # Topic Modeling (5 Topics)
    # --------------------
    st.markdown("---")
    st.markdown("### 🧩 Topic Modeling (5 Topics)")
    col1, col2 = st.columns(2)
    topics1 = get_topics_for_artist(data, artist1, num_topics=5)
    topics2 = get_topics_for_artist(data, artist2, num_topics=5)

    with col1:
        st.write(f"**Topics for {artist1}**")
        if topics1:
            for topic_label, words in topics1:
                st.write(f"{topic_label}: {', '.join(words)}")
        else:
            st.write("Not enough data or something went wrong for LDA.")

    with col2:
        st.write(f"**Topics for {artist2}**")
        if topics2:
            for topic_label, words in topics2:
                st.write(f"{topic_label}: {', '.join(words)}")
        else:
            st.write("Not enough data or something went wrong for LDA.")

    # --------------------
    # Emotion Detection
    # --------------------
    st.markdown("---")
    st.markdown("### ❤️ Dominant Emotions (Top 3)")
    col1, col2 = st.columns(2)
    emotions1 = get_emotions_for_artist(data, artist1)
    emotions2 = get_emotions_for_artist(data, artist2)

    with col1:
        st.write(f"**Top Emotions for {artist1}**")
        if emotions1:
            df_emo1 = pd.DataFrame(emotions1, columns=["Emotion", "Score (out of 100)"])
            st.table(df_emo1)
        else:
            st.write("No emotion data found.")

    with col2:
        st.write(f"**Top Emotions for {artist2}**")
        if emotions2:
            df_emo2 = pd.DataFrame(emotions2, columns=["Emotion", "Score (out of 100)"])
            st.table(df_emo2)
        else:
            st.write("No emotion data found.")
