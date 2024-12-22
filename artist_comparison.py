import streamlit as st
import pandas as pd
import re
from collections import Counter
from sentiment_analysis import analyze_sentiment
import text2emotion as te
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# ------------------------------------------
# 1) STOPWORDS: includes filler words, etc.
# ------------------------------------------
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
# 2) FREQUENT WORDS: exclude words shorter than 5 chars
# ------------------------------------------------------
@st.cache_data
def get_most_frequent_words(data, artist, top_n=10):
    """
    Returns the top-n most frequent (non-stopword) words, 
    ignoring words with length < 5.
    """
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()
    filtered_words = [
        w.lower() for w in words 
        if len(w) >= 5 and w.lower() not in stop_words
    ]
    counts = Counter(filtered_words).most_common(top_n)
    df = pd.DataFrame(counts, columns=['Word', 'Frequency'])
    df.index = df.index + 1
    return df

def get_filtered_top_songs_by_sentiment(data, artist, top_n=3):
    """
    Returns top positive & negative songs (min 1000 views).
    """
    filtered_data = data[(data['artist'] == artist) & (data['views'] >= 1000)]
    if filtered_data.empty:
        return (pd.DataFrame(columns=['title','sentiment','views']),
                pd.DataFrame(columns=['title','sentiment','views']))

    top_positive = filtered_data.sort_values(by='sentiment', ascending=False).head(top_n).reset_index(drop=True)
    top_negative = filtered_data.sort_values(by='sentiment').head(top_n).reset_index(drop=True)
    return top_positive[['title','sentiment','views']], top_negative[['title','sentiment','views']]

# --------------------------------------------------------------------------------
# 3) TOPIC MODELING: label "topics" with naive "emotions" instead of Topic 0,1,2...
# --------------------------------------------------------------------------------
@st.cache_data
def get_topics_for_artist(data, artist, num_topics=5):
    """
    Use gensim LDA to get topics for an artist's combined lyrics.
    Return list of (topicIndex, [(word, prob),...]).
    """
    artist_lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()

    tokens = []
    for lyric in artist_lyrics_list:
        lyric_tokens = [
            w.lower() for w in lyric.split()
            if w.lower() not in stop_words and len(w) >= 2
        ]
        tokens.append(lyric_tokens)

    if not tokens:
        return []

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
        return []

    return lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)

def guess_emotion_from_words(words):
    """
    Naively guess an 'emotion' label based on presence of certain words.
    Feel free to refine or expand the logic below.
    """
    # Lower them all
    words_lower = set(w.lower() for w in words)

    # Some example words to detect
    sad_words = {"cry", "sad", "tears", "alone", "lonely", "blue"}
    happy_words = {"happy", "joy", "smile", "sunshine", "laugh"}
    angry_words = {"anger", "rage", "hate", "mad"}
    love_words = {"kiss", "heart", "love", "romance", "darling", "sweet"}
    fear_words = {"fear", "scared", "afraid", "nightmare", "horror"}

    # Simple checks in descending priority
    if words_lower & love_words:
        return "Love / Romance"
    elif words_lower & happy_words:
        return "Joy / Happiness"
    elif words_lower & sad_words:
        return "Sadness / Melancholy"
    elif words_lower & angry_words:
        return "Anger / Frustration"
    elif words_lower & fear_words:
        return "Fear / Darkness"
    else:
        return "General / ambiguous"

def interpret_topics_as_emotions(topics):
    """
    Convert the LDA output (list of (topic_num, [(word, prob), ...]))
    into a table of (EmotionLabel, TopWords).
    """
    result = []
    for _, word_probs in topics:
        # Extract top words only
        top_words = [wp[0] for wp in word_probs]
        # Guess an "emotion" from top words
        emotion_label = guess_emotion_from_words(top_words)
        words_joined = ", ".join(top_words)
        result.append((emotion_label, words_joined))
    return result

# ------------------------------------------------------
# 4) EMOTION DETECTION: remove caching, sanitize lyrics
# ------------------------------------------------------
def get_emotions_for_artist(data, artist):
    """
    text2emotion can fail on certain unusual chars/emojis, 
    so we sanitize. Then return top 3 emotions (percentage).
    """
    artist_lyrics_list = data[data['artist'] == artist]['lyrics'].dropna().tolist()
    if not artist_lyrics_list:
        return []

    cumulative_emotions = Counter()
    for lyric in artist_lyrics_list:
        # Remove non-ASCII or potential trouble chars
        clean_lyric = re.sub(r'[^\x00-\x7F]+',' ', lyric)
        # Now get emotion
        emotion_scores = te.get_emotion(clean_lyric)
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

# ------------------------------------------------------
# 5) MAIN COMPARISON FUNCTION
# ------------------------------------------------------
def compare_artists(data):
    st.title("🎸 Artist Comparison")

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
    # Sentiment Over Time
    # --------------------
    st.markdown("---")
    st.markdown("### 📈 Sentiment Over Time (Yearly)")
    sentiment_by_year = data.groupby(['year', 'artist'])['sentiment'].mean().unstack().fillna(0)
    st.line_chart(sentiment_by_year, use_container_width=True)

    # --------------------
    # Lexical Diversity
    # --------------------
    st.markdown("### 📚 Lexical Complexity Over Time (Yearly)")
    data['lexical_diversity'] = data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()) if x and len(str(x).split()) > 0 else 0
    )
    lexical_by_year = data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lexical_by_year, use_container_width=True)

    # --------------------
    # Frequent Words
    # --------------------
    st.markdown("---")
    st.markdown("### 💬 Top 10 Frequent Words")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{artist1}**")
        freq1 = get_most_frequent_words(data, artist1, top_n=10)
        st.table(freq1)

    with col2:
        st.write(f"**{artist2}**")
        freq2 = get_most_frequent_words(data, artist2, top_n=10)
        st.table(freq2)

    # --------------------
    # Topic Modeling => Emotions
    # --------------------
    st.markdown("---")
    st.markdown("### ❌ Topic Modeling (5 Topics)")
    col1, col2 = st.columns(2)
    topics1 = get_topics_for_artist(data, artist1, num_topics=5)
    topics2 = get_topics_for_artist(data, artist2, num_topics=5)

    # Convert raw LDA output into "emotion-labeled" table
    explained1 = interpret_topics_as_emotions(topics1)
    explained2 = interpret_topics_as_emotions(topics2)

    with col1:
        st.write(f"**Topics for {artist1}**")
        if explained1:
            df1 = pd.DataFrame(explained1, columns=["Emotion Label", "Top Words"])
            st.table(df1)
        else:
            st.write("No topics found.")

    with col2:
        st.write(f"**Topics for {artist2}**")
        if explained2:
            df2 = pd.DataFrame(explained2, columns=["Emotion Label", "Top Words"])
            st.table(df2)
        else:
            st.write("No topics found.")

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
