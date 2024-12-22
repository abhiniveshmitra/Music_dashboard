import streamlit as st
import pandas as pd
import re
from collections import Counter, defaultdict
from sentiment_analysis import analyze_sentiment
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# ---------- NEW: NLTK Imports for POS & bigrams ----------
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Ensure nltk data is present (quiet downloads)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# ---------- STOPWORDS & CLEANING ----------
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well", "dont", "where", "never", "youre", "gonna", "going", "could",
    "about", "cant", "yeah", "right", "every", "little", "youre", "dont", 
    "aint", "all", "like", "down", "just", "got", "her", "his", "im",
    "ill", "ive", "id", "its", "one", "time", "will", "what", "come",
    "wanna", "when", "more", "here", "want", "day", "man", "now"
])

def clean_word(w: str) -> str:
    """Remove punctuation, make lowercase."""
    return re.sub(r"[^\w\s]", "", w).lower()

# ---------- (1) FREQUENT WORDS ----------
@st.cache_data
@st.cache_data
def get_most_frequent_words(data, artist, top_n=10):
    """
    Returns the top-n most frequent words for a given artist,
    ignoring words with length <5 and any in stop_words.
    """
    lyrics_series = data[data['artist'] == artist]['lyrics'].dropna()
    # Combine all lyrics (rows) into one big string
    combined = " ".join(lyrics_series.astype(str))  # <-- This fixes the error

    # Now tokenize that big string
    raw_tokens = combined.split()

    cleaned_tokens = []
    for token in raw_tokens:
        t = clean_word(token)  # remove punctuation, lowercase
        if len(t) >= 5 and t not in stop_words:
            cleaned_tokens.append(t)

    counts = Counter(cleaned_tokens).most_common(top_n)
    df = pd.DataFrame(counts, columns=['Word', 'Frequency'])
    df.index = df.index + 1
    return df

def get_filtered_top_songs_by_sentiment(data, artist, top_n=3):
    """Top positive & negative songs (min 1000 views)."""
    subset = data[(data['artist'] == artist) & (data['views'] >= 1000)]
    if subset.empty:
        return (
            pd.DataFrame(columns=['title','sentiment','views']),
            pd.DataFrame(columns=['title','sentiment','views'])
        )

    top_positive = subset.sort_values(by='sentiment', ascending=False).head(top_n).reset_index(drop=True)
    top_negative = subset.sort_values(by='sentiment').head(top_n).reset_index(drop=True)
    return top_positive[['title','sentiment','views']], top_negative[['title','sentiment','views']]

# ---------- (2) TOPIC MODELING (kept same) ----------
@st.cache_data
def get_topics_for_artist(data, artist, num_topics=5):
    """Gensim LDA => 5 topics for the artist's lyrics."""
    lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()

    token_docs = []
    for lyric in lyrics_list:
        tokens = []
        for w in lyric.split():
            cw = clean_word(w)
            if len(cw) >= 5 and cw not in stop_words:
                tokens.append(cw)
        token_docs.append(tokens)

    if not token_docs:
        return []

    dictionary = Dictionary(token_docs)
    corpus = [dictionary.doc2bow(doc) for doc in token_docs]

    try:
        lda = LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_state=42,
            passes=1
        )
    except ValueError:
        return []

    return lda.show_topics(num_topics=num_topics, num_words=5, formatted=False)

def interpret_topics_as_emotions(topics):
    """We keep the same naive logic from before or rename it if you prefer."""
    results = []
    for idx, word_probs in topics:
        top_words = [wp[0] for wp in word_probs]
        # We'll label them all "General" or "Unknown" now.
        results.append(("Unknown", ", ".join(top_words)))
    return results

# ---------- (3) SIMPLE CUSTOM EMOTION DETECTOR ----------
# Instead of text2emotion, define a small lexicon for 5 emotions:
emotion_lexicon = {
    "joy": {"happy", "joy", "delight", "laugh", "smile", "pleasure"},
    "sadness": {"sad", "cry", "tears", "alone", "lonely", "blue", "sorrow"},
    "anger": {"anger", "rage", "mad", "furious", "hate", "angry"},
    "fear": {"fear", "scare", "scared", "afraid", "fright", "horror", "dread"},
    "love": {"love", "heart", "romance", "darling", "sweet", "kiss"},
}

def find_emotions_for_artist(data, artist):
    """
    Scan all lyrics for each artist. For each word (cleaned),
    check if it belongs to an emotion set. Tally counts.
    Also store the words that triggered each emotion.
    """
    lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()
    if not lyrics_list:
        return {}  # no data

    # Tally counts, plus track which words contributed
    emotion_counts = Counter()
    emotion_words_map = defaultdict(set)

    for lyric in lyrics_list:
        tokens = word_tokenize(lyric.lower())
        for tok in tokens:
            w = clean_word(tok)
            if len(w) < 5:
                continue
            # Check if w in any emotion set
            for emotion, lexset in emotion_lexicon.items():
                if w in lexset:
                    emotion_counts[emotion] += 1
                    emotion_words_map[emotion].add(w)

    # Sort by frequency
    if not emotion_counts:
        return {}

    # Return a dict: {emotion: (count, {words...})}
    result = {}
    for emotion, cnt in emotion_counts.most_common():
        result[emotion] = (cnt, emotion_words_map[emotion])
    return result

# ---------- (4) POS DISTRIBUTION ----------
# We'll show top 5 POS tags for each artist
@st.cache_data
def pos_distribution_for_artist(data, artist):
    """
    Tag each word in the artist's lyrics with NLTK's pos_tag,
    then return a distribution of the POS tags (top 5).
    """
    lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    tokens = word_tokenize(lyrics)
    tagged = pos_tag(tokens)  # yields (word, POS_tag)
    counts = Counter(tag for (_, tag) in tagged)
    top_5 = counts.most_common(5)
    df = pd.DataFrame(top_5, columns=["POS Tag", "Count"])
    df.index = df.index + 1
    return df

# ---------- (5) TOP 5 BIGRAM COLLOCATIONS ----------
@st.cache_data
def top_bigrams_for_artist(data, artist, top_n=5):
    """
    Find top 5 bigram collocations in the artist's lyrics
    using NLTK's BigramCollocationFinder + PMI measure.
    """
    lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    tokens = [
        clean_word(w) for w in word_tokenize(lyrics)
        if len(clean_word(w)) >= 5 and clean_word(w) not in stop_words
    ]
    if not tokens:
        return pd.DataFrame(columns=["Bigram", "PMI"])

    finder = BigramCollocationFinder.from_words(tokens)
    # filter out bigrams that appear <2 times, for instance
    finder.apply_freq_filter(2)

    scored = finder.score_ngrams(BigramAssocMeasures.pmi)
    top_n_scored = scored[:top_n]

    # each item: ((word1, word2), pmi_score)
    bigrams = []
    for (w1, w2), pmi_val in top_n_scored:
        bigrams.append((f"{w1} {w2}", round(pmi_val, 3)))

    df = pd.DataFrame(bigrams, columns=["Bigram", "PMI"])
    df.index = df.index + 1
    return df

# ------------------------------------------------------
# MAIN compare_artists
# ------------------------------------------------------
def compare_artists(data):
    st.title("🎸 Artist Comparison")

    unique_artists = data['artist'].unique()
    if len(unique_artists) < 2:
        st.error("Not enough artists selected.")
        return

    artist1, artist2 = unique_artists[0], unique_artists[1]

    # -----------------------------------
    # Popular Songs
    # -----------------------------------
    st.markdown("### 🔥 Most Popular Songs (Top 3 by Views)")
    popular_songs = (
        data[data['views'] >= 1000]
        .sort_values(by='views', ascending=False)
        .groupby('artist')
        .head(3)
        .reset_index(drop=True)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Most Popular Songs by {artist1}**")
        st.table(popular_songs[popular_songs['artist'] == artist1][['title', 'views']])

    with c2:
        st.markdown(f"**Most Popular Songs by {artist2}**")
        st.table(popular_songs[popular_songs['artist'] == artist2][['title', 'views']])

    # -----------------------------------
    # Top Positive/Negative
    # -----------------------------------
    st.markdown("### 🎵 Top Positive and Negative Songs (Min. 1000 Views)")
    pos1, neg1 = get_filtered_top_songs_by_sentiment(data, artist1)
    pos2, neg2 = get_filtered_top_songs_by_sentiment(data, artist2)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Top Positive Songs by {artist1}**")
        st.table(pos1)
        st.markdown(f"**Top Negative Songs by {artist1}**")
        st.table(neg1)

    with c2:
        st.markdown(f"**Top Positive Songs by {artist2}**")
        st.table(pos2)
        st.markdown(f"**Top Negative Songs by {artist2}**")
        st.table(neg2)

    # -----------------------------------
    # Sentiment Over Time
    # -----------------------------------
    st.markdown("---")
    st.markdown("### 📈 Sentiment Over Time (Yearly)")
    senti_year = data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    st.line_chart(senti_year, use_container_width=True)

    # -----------------------------------
    # Lexical Diversity
    # -----------------------------------
    st.markdown("### 📚 Lexical Complexity Over Time (Yearly)")
    data['lexical_diversity'] = data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()) if x and len(str(x).split()) > 0 else 0
    )
    lex_year = data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lex_year, use_container_width=True)

    # -----------------------------------
    # Frequent Words
    # -----------------------------------
    st.markdown("---")
    st.markdown("### 💬 Top 10 Frequent Words")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**{artist1}**")
        freq1 = get_most_frequent_words(data, artist1, top_n=10)
        st.table(freq1)

    with c2:
        st.write(f"**{artist2}**")
        freq2 = get_most_frequent_words(data, artist2, top_n=10)
        st.table(freq2)

    # -----------------------------------
    # Topic Modeling => "Unknown" labels
    # -----------------------------------
    st.markdown("---")
    st.markdown("### ❌ Topic Modeling (5 Topics)")
    c1, c2 = st.columns(2)

    raw_topics1 = get_topics_for_artist(data, artist1, num_topics=5)
    raw_topics2 = get_topics_for_artist(data, artist2, num_topics=5)

    explained1 = interpret_topics_as_emotions(raw_topics1)
    explained2 = interpret_topics_as_emotions(raw_topics2)

    with c1:
        st.write(f"**Topics for {artist1}**")
        if explained1:
            df1 = pd.DataFrame(explained1, columns=["Label", "Top Words"])
            st.table(df1)
        else:
            st.write("No topics found.")

    with c2:
        st.write(f"**Topics for {artist2}**")
        if explained2:
            df2 = pd.DataFrame(explained2, columns=["Label", "Top Words"])
            st.table(df2)
        else:
            st.write("No topics found.")

    # -----------------------------------
    # (A) Custom Emotion Lexicon
    # -----------------------------------
    st.markdown("---")
    st.markdown("### ❤️ Top 5 Emotions (Lexicon-Based)")
    c1, c2 = st.columns(2)

    emo_map_1 = find_emotions_for_artist(data, artist1)
    emo_map_2 = find_emotions_for_artist(data, artist2)

    with c1:
        st.write(f"**Top Emotions for {artist1}**")
        if emo_map_1:
            # emo_map_1 => { emotion: (count, set_of_words), ... } in descending freq
            items = list(emo_map_1.items())[:5]  # top 5
            # Flatten into a small table
            rows = []
            for emotion, (cnt, words) in items:
                rows.append((emotion, cnt, ", ".join(sorted(words))))
            df_emo_1 = pd.DataFrame(rows, columns=["Emotion", "Count", "Trigger Words"])
            st.table(df_emo_1)
        else:
            st.write("No emotions found (via custom lexicon).")

    with c2:
        st.write(f"**Top Emotions for {artist2}**")
        if emo_map_2:
            items = list(emo_map_2.items())[:5]
            rows = []
            for emotion, (cnt, words) in items:
                rows.append((emotion, cnt, ", ".join(sorted(words))))
            df_emo_2 = pd.DataFrame(rows, columns=["Emotion", "Count", "Trigger Words"])
            st.table(df_emo_2)
        else:
            st.write("No emotions found (via custom lexicon).")

    # -----------------------------------
    # (B) POS Distribution
    # -----------------------------------
    st.markdown("---")
    st.markdown("### 📊 Part-of-Speech (POS) Distribution (Top 5)")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**POS for {artist1}**")
        df_pos_1 = pos_distribution_for_artist(data, artist1)
        st.table(df_pos_1)

    with c2:
        st.write(f"**POS for {artist2}**")
        df_pos_2 = pos_distribution_for_artist(data, artist2)
        st.table(df_pos_2)

    # -----------------------------------
    # (C) Top 5 Bigram Collocations
    # -----------------------------------
    st.markdown("---")
    st.markdown("### 🔗 Top 5 Bigram Collocations")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Bigrams for {artist1}**")
        df_bi_1 = top_bigrams_for_artist(data, artist1, top_n=5)
        st.table(df_bi_1)

    with c2:
        st.write(f"**Bigrams for {artist2}**")
        df_bi_2 = top_bigrams_for_artist(data, artist2, top_n=5)
        st.table(df_bi_2)
