import streamlit as st
import pandas as pd
import re
from collections import Counter
from sentiment_analysis import analyze_sentiment
import text2emotion as te
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# ----------------------------------------
# STOPWORDS (customize as needed)
# ----------------------------------------
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah", "right", "every", "little", "you’re", "don’t", 
    "ain’t", "can’t", "all", "like", "down", "just", "got", "her", "his", "i’m",
    "i'll", "i've", "i'd", "it’s", "one", "time", "will", "what", "come",
    "wanna", "when", "more", "here", "want", "day", "man", "now"
])

# ----------------------------------------
# A helper to strip punctuation from words
# e.g., "there's" => "theres", "night," => "night"
# ----------------------------------------
def clean_word(w: str) -> str:
    # remove punctuation
    return re.sub(r"[^\w\s]", "", w).lower()

# ------------------------------------------------------
# FREQUENT WORDS (ignore <5 letters & stopwords)
# ------------------------------------------------------
@st.cache_data
def get_most_frequent_words(data, artist, top_n=10):
    """
    Returns the top-n most frequent words for a given artist,
    ignoring words with length <5, plus stopwords.
    Also removes punctuation from tokens (e.g., "night," -> "night").
    """
    lyrics_series = data[data['artist'] == artist]['lyrics'].dropna()
    combined = " ".join(lyrics_series)
    raw_tokens = combined.split()

    # Clean & filter
    cleaned_tokens = []
    for token in raw_tokens:
        t = clean_word(token)
        if len(t) >= 5 and t not in stop_words:
            cleaned_tokens.append(t)

    counts = Counter(cleaned_tokens).most_common(top_n)
    df = pd.DataFrame(counts, columns=['Word', 'Frequency'])
    df.index = df.index + 1
    return df

# ------------------------------------------------------
# TOP POSITIVE/NEGATIVE SONGS
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
# TOPIC MODELING => EMOTION LABELS
# ignoring <5 letter words & punctuation
# ------------------------------------------------------
@st.cache_data
def get_topics_for_artist(data, artist, num_topics=5):
    """
    Gensim LDA on the artist's lyrics, ignoring words <5 letters, punctuation, and stopwords.
    Returns LDA topics => [ (topicIndex, [(word, prob), ... ]), ... ]
    """
    lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()

    token_docs = []
    for lyric in lyrics_list:
        raw_tokens = lyric.split()
        cleaned = []
        for token in raw_tokens:
            t = clean_word(token)
            if len(t) >= 5 and t not in stop_words:
                cleaned.append(t)
        token_docs.append(cleaned)

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

def guess_emotion_label(words):
    """
    Attempt to label the topic as Anger, Joy, Sadness, Love, Fear, 
    or fallback to "Neutral/No Specific Emotion".
    """
    # define sets
    anger_set = {"anger", "hate", "rage", "mad"}
    joy_set = {"happy", "smile", "laugh", "sunny", "joy", "delight", "delighted"}
    sadness_set = {"sad", "cry", "tears", "alone", "lonely", "blue"}
    love_set = {"love", "heart", "kiss", "romance", "darling", "sweet"}
    fear_set = {"fear", "scare", "afraid", "nightmare", "horror"}

    # convert to a set for quick membership check
    wset = set(words)
    if wset & love_set:
        return "Love"
    elif wset & joy_set:
        return "Joy"
    elif wset & sadness_set:
        return "Sadness"
    elif wset & anger_set:
        return "Anger"
    elif wset & fear_set:
        return "Fear"
    else:
        return "Neutral/No Specific Emotion"

def interpret_topics_as_emotions(topics):
    """
    LDA => list of (EmotionLabel, TopWordsString)
    """
    results = []
    for _, word_probs in topics:
        # top words
        top_words = [wp[0] for wp in word_probs]
        label = guess_emotion_label(top_words)
        results.append((label, ", ".join(top_words)))
    return results

# ------------------------------------------------------
# text2emotion => fix "No emotion data found"
# remove emojis, punctuation, etc.
# ------------------------------------------------------
def cleanup_lyric_for_text2emotion(lyric):
    # Remove any non-ascii
    lyric = re.sub(r'[^\x00-\x7F]+',' ', lyric)
    # Remove punctuation
    lyric = re.sub(r'[^\w\s]', ' ', lyric)
    return lyric

def get_emotions_for_artist(data, artist):
    lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()
    if not lyrics_list:
        return []

    total_emotions = Counter()
    for lyric in lyrics_list:
        safe_lyric = cleanup_lyric_for_text2emotion(lyric)
        # text2emotion sometimes fails if the string is empty
        if not safe_lyric.strip():
            continue
        try:
            emo_dict = te.get_emotion(safe_lyric)
            for e, val in emo_dict.items():
                total_emotions[e] += val
        except:
            # If text2emotion fails, skip
            continue

    if not total_emotions:
        return []

    summation = sum(total_emotions.values())
    emotion_list = [
        (e, round((val / summation) * 100, 2)) for e, val in total_emotions.items()
    ]
    emotion_list.sort(key=lambda x: x[1], reverse=True)
    return emotion_list[:3]

# ------------------------------------------------------
# MAIN COMPARISON FUNCTION
# ------------------------------------------------------
def compare_artists(data):
    st.title("🎸 Artist Comparison")

    # We expect 2 selected artists
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

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Most Popular Songs by {artist1}**")
        st.table(popular_songs[popular_songs['artist'] == artist1][['title', 'views']])

    with c2:
        st.markdown(f"**Most Popular Songs by {artist2}**")
        st.table(popular_songs[popular_songs['artist'] == artist2][['title', 'views']])

    # --------------------
    # Top Positive/Negative
    # --------------------
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

    # --------------------
    # Sentiment Over Time
    # --------------------
    st.markdown("---")
    st.markdown("### 📈 Sentiment Over Time (Yearly)")
    senti_year = data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    st.line_chart(senti_year, use_container_width=True)

    # --------------------
    # Lexical Complexity
    # --------------------
    st.markdown("### 📚 Lexical Complexity Over Time (Yearly)")
    data['lexical_diversity'] = data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()) if x and len(str(x).split()) > 0 else 0
    )
    lex_year = data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lex_year, use_container_width=True)

    # --------------------
    # Frequent Words
    # --------------------
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

    # --------------------
    # Topic Modeling => Emotion
    # --------------------
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

    # --------------------
    # text2emotion
    # --------------------
    st.markdown("---")
    st.markdown("### ❤️ Dominant Emotions (Top 3)")
    c1, c2 = st.columns(2)

    emo1 = get_emotions_for_artist(data, artist1)
    emo2 = get_emotions_for_artist(data, artist2)

    with c1:
        st.write(f"**Top Emotions for {artist1}**")
        if emo1:
            df_emo1 = pd.DataFrame(emo1, columns=["Emotion", "Score (out of 100)"])
            st.table(df_emo1)
        else:
            st.write("No emotion data found.")

    with c2:
        st.write(f"**Top Emotions for {artist2}**")
        if emo2:
            df_emo2 = pd.DataFrame(emo2, columns=["Emotion", "Score (out of 100)"])
            st.table(df_emo2)
        else:
            st.write("No emotion data found.")
