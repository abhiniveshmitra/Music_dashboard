import streamlit as st
import pandas as pd
import re
from collections import Counter
from sentiment_analysis import analyze_sentiment
import text2emotion as te
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# ----------------------------------------
# 1) STOPWORDS (customizable as you like)
# ----------------------------------------
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every", "little", "you’re", "don’t", 
    "ain’t", "can’t", "all", "like", "down", "just", "got", "her", "his", "i’m",
    "i'll", "i've", "i'd", "it’s", "one", "time", "will", "what", "come", "love",
    "wanna", "when", "more", "here", "there", "want", "day", "man", "now"
])

# ------------------------------------------------------
# 2) FREQUENT WORDS (ignore <5 letters & in stopwords)
# ------------------------------------------------------
@st.cache_data
def get_most_frequent_words(data, artist, top_n=10):
    """
    Returns the top-n most frequent words for a given artist,
    ignoring words with length < 5, plus stopwords.
    """
    lyrics = data[data['artist'] == artist]['lyrics'].dropna()
    combined = " ".join(lyrics)
    tokens = combined.split()

    filtered = [
        w.lower() for w in tokens
        if len(w) >= 5 and w.lower() not in stop_words
    ]
    counts = Counter(filtered).most_common(top_n)

    df = pd.DataFrame(counts, columns=['Word', 'Frequency'])
    df.index = df.index + 1
    return df

# ------------------------------------------------------
# 3) GET POSITIVE/NEGATIVE SONGS
# ------------------------------------------------------
def get_filtered_top_songs_by_sentiment(data, artist, top_n=3):
    subset = data[(data['artist'] == artist) & (data['views'] >= 1000)]
    if subset.empty:
        return (
            pd.DataFrame(columns=['title','sentiment','views']),
            pd.DataFrame(columns=['title','sentiment','views'])
        )

    top_positive = subset.sort_values(by='sentiment', ascending=False).head(top_n).reset_index(drop=True)
    top_negative = subset.sort_values(by='sentiment').head(top_n).reset_index(drop=True)
    return top_positive[['title','sentiment','views']], top_negative[['title','sentiment','views']]

# ------------------------------------------------------
# 4) TOPIC MODELING => "EMOTION" LABELS
#    Also ignoring <5 letters & stopwords
# ------------------------------------------------------
@st.cache_data
def get_topics_for_artist(data, artist, num_topics=5):
    """
    Gensim LDA to discover 5 "topics" in the artist's lyrics,
    ignoring words <5 letters & any stopwords.
    Returns: [(topicIndex, [(word, prob), ...]), ...]
    """
    lyrics = data[data['artist'] == artist]['lyrics'].dropna().tolist()

    tokenized_docs = []
    for lyric in lyrics:
        tokens = [
            w.lower() for w in lyric.split()
            if len(w) >= 5 and w.lower() not in stop_words
        ]
        tokenized_docs.append(tokens)

    if not tokenized_docs:
        return []

    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

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

def guess_emotion_label(words):
    """
    Naively guess an 'emotion' label from top words.
    The user specifically wants labels like 'Anger', 'Joy', 'Sadness', 'Love', 'Fear'.
    """
    # Convert to lower, for safety
    lower_words = set(w.lower() for w in words)

    # Example sets
    anger_set = {"anger", "hate", "rage", "mad"}
    joy_set = {"happy", "smile", "laugh", "sunny", "joy", "delight"}
    sadness_set = {"sad", "cry", "tears", "alone", "lonely", "blue"}
    love_set = {"love", "heart", "kiss", "romance", "darling", "sweet"}
    fear_set = {"fear", "scare", "afraid", "nightmare", "horror"}

    # Simple check
    if lower_words & love_set:
        return "Love"
    elif lower_words & joy_set:
        return "Joy"
    elif lower_words & sadness_set:
        return "Sadness"
    elif lower_words & anger_set:
        return "Anger"
    elif lower_words & fear_set:
        return "Fear"
    else:
        return "General"

def interpret_topics_as_emotions(topics):
    """
    Convert LDA's topics into a list of (EmotionLabel, TopWords).
    """
    results = []
    for _, word_probs in topics:
        top_words = [wp[0] for wp in word_probs]
        # Assign 1 of the 5 emotion labels (or General) based on top words
        label = guess_emotion_label(top_words)
        word_str = ", ".join(top_words)
        results.append((label, word_str))
    return results

# ------------------------------------------------------
# 5) EMOTION DETECTION (no caching; remove emojis)
# ------------------------------------------------------
def remove_emojis_and_nonascii(text):
    """
    Remove emojis or weird Unicode chars
    so text2emotion won't crash.
    """
    # Remove all non-ASCII
    cleaned = re.sub(r'[^\x00-\x7F]+',' ', text)
    return cleaned

def get_emotions_for_artist(data, artist):
    """
    text2emotion can crash on weird emojis, so remove them.
    Return top 3 aggregated emotions as (Emotion, Score).
    """
    lyrics = data[data['artist'] == artist]['lyrics'].dropna().tolist()
    if not lyrics:
        return []

    total_emotions = Counter()

    for lyric in lyrics:
        # Clean each lyric
        safe_lyric = remove_emojis_and_nonascii(lyric)
        # Now compute emotion
        try:
            score_dict = te.get_emotion(safe_lyric)
        except Exception:
            # If text2emotion still fails, skip this lyric
            continue
        
        for emo, val in score_dict.items():
            total_emotions[emo] += val

    if not total_emotions:
        return []

    sum_val = sum(total_emotions.values())
    # Convert to percentages
    emotion_list = [
        (emo, round((val / sum_val) * 100, 2))
        for emo, val in total_emotions.items()
    ]
    # Sort descending
    emotion_list.sort(key=lambda x: x[1], reverse=True)
    # Return top 3
    return emotion_list[:3]

# ------------------------------------------------------
# 6) MAIN: compare_artists
# ------------------------------------------------------
def compare_artists(data):
    st.title("🎸 Artist Comparison")

    # We expect 2 artists
    unique_artists = data['artist'].unique()
    if len(unique_artists) < 2:
        st.error("Not enough artists selected.")
        return

    artist1, artist2 = unique_artists[0], unique_artists[1]

    # --------------------------------------------------
    # Popular Songs
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Top Positive/Negative
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Sentiment Over Time
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("### 📈 Sentiment Over Time (Yearly)")
    sentiment_by_year = data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    st.line_chart(sentiment_by_year, use_container_width=True)

    # --------------------------------------------------
    # Lexical Diversity
    # --------------------------------------------------
    st.markdown("### 📚 Lexical Complexity Over Time (Yearly)")
    data['lexical_diversity'] = data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()) if x and len(str(x).split()) > 0 else 0
    )
    lex_by_year = data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lex_by_year, use_container_width=True)

    # --------------------------------------------------
    # Frequent Words
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Topic Modeling => "Emotion" Topics
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("### ❌ Topic Modeling (5 'Emotion' Topics)")
    c1, c2 = st.columns(2)

    raw_topics1 = get_topics_for_artist(data, artist1, num_topics=5)
    raw_topics2 = get_topics_for_artist(data, artist2, num_topics=5)

    explained1 = interpret_topics_as_emotions(raw_topics1)
    explained2 = interpret_topics_as_emotions(raw_topics2)

    with c1:
        st.write(f"**Topics for {artist1}**")
        if explained1:
            df1 = pd.DataFrame(explained1, columns=["Emotion Topic", "Top Words"])
            st.table(df1)
        else:
            st.write("No topics found.")

    with c2:
        st.write(f"**Topics for {artist2}**")
        if explained2:
            df2 = pd.DataFrame(explained2, columns=["Emotion Topic", "Top Words"])
            st.table(df2)
        else:
            st.write("No topics found.")

    # --------------------------------------------------
    # text2emotion: Overall Emotions
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("### ❤️ Dominant Emotions (Top 3)")
    c1, c2 = st.columns(2)

    # Summarize all lyrics for each artist
    emotions1 = get_emotions_for_artist(data, artist1)
    emotions2 = get_emotions_for_artist(data, artist2)

    with c1:
        st.write(f"**Top Emotions for {artist1}**")
        if emotions1:
            df_emo1 = pd.DataFrame(emotions1, columns=["Emotion", "Score (out of 100)"])
            st.table(df_emo1)
        else:
            st.write("No emotion data found.")

    with c2:
        st.write(f"**Top Emotions for {artist2}**")
        if emotions2:
            df_emo2 = pd.DataFrame(emotions2, columns=["Emotion", "Score (out of 100)"])
            st.table(df_emo2)
        else:
            st.write("No emotion data found.")
