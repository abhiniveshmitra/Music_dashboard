import streamlit as st
import pandas as pd
from collections import Counter
from sentiment_analysis import analyze_sentiment
import text2emotion as te
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Stopwords List
stop_words = set([
    "the", "and", "is", "in", "it", "of", "to", "on", "that", "this", "for",
    "with", "as", "was", "at", "by", "from", "which", "an", "be", "or", "are",
    "but", "if", "then", "so", "such", "there", "has", "have", "had", "a", "he",
    "she", "they", "we", "you", "your", "our", "my", "their", "its", "out", "not",
    "well,", "don't", "where", "never", "you're", "gonna", "going", "could",
    "about", "can't", "yeah,", "right", "every", "little", "you’re", "don’t", "ain’t", "can’t"
])

@st.cache_data
def get_most_frequent_words(data, artist, top_n=10):
    """
    Returns the top-n most frequent words for a given artist.
    """
    artist_lyrics = " ".join(data[data['artist'] == artist]['lyrics'].dropna())
    words = artist_lyrics.split()
    filtered_words = [w.lower() for w in words if len(w) > 4 and w.lower() not in stop_words]
    counts = Counter(filtered_words).most_common(top_n)
    df = pd.DataFrame(counts, columns=['Word', 'Frequency'])
    df.index = df.index + 1  # 1-based indexing
    return df

def get_filtered_top_songs_by_sentiment(data, artist, top_n=3):
    """
    Returns top positive & negative songs (min 1000 views) for a given artist.
    """
    filtered_data = data[(data['artist'] == artist) & (data['views'] >= 1000)]
    if filtered_data.empty:
        return pd.DataFrame(columns=['title', 'sentiment', 'views']), pd.DataFrame(columns=['title', 'sentiment', 'views'])

    top_positive = filtered_data.sort_values(by='sentiment', ascending=False).head(top_n).reset_index(drop=True)
    top_negative = filtered_data.sort_values(by='sentiment').head(top_n).reset_index(drop=True)

    return top_positive[['title', 'sentiment', 'views']], top_negative[['title', 'sentiment', 'views']]

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
        lda_model = LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            random_state=42,
            passes=1
        )
    except ValueError:
        return []

    topics = lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)
    topic_list = []
    for topic_num, word_probs in topics:
        words_only = [wp[0] for wp in word_probs]
        topic_list.append((f"Topic {topic_num}", words_only))

    return topic_list

@st.cache_data
def get_emotions_for_artist(data, artist):
    """
    Aggregates text2emotion scores across all lyrics by the artist.
    Returns top 3 emotions by percentage.
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
    # Lexical Diversity Over Time
    # --------------------
    st.markdown("### 📚 Lexical Complexity Over Time (Yearly)")
    data['lexical_diversity'] = data['lyrics'].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()) if x and len(str(x).split()) > 0 else 0
    )
    lexical_by_year = data.groupby(['year', 'artist'])['lexical_diversity'].mean().unstack().fillna(0)
    st.line_chart(lexical_by_year, use_container_width=True)

    # --------------------
    # Frequent Words (Optional Showcase)
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
            st.write("Not enough data or no topics found.")

    with col2:
        st.write(f"**Topics for {artist2}**")
        if topics2:
            for topic_label, words in topics2:
                st.write(f"{topic_label}: {', '.join(words)}")
        else:
            st.write("Not enough data or no topics found.")

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
