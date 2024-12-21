import streamlit as st
from data_loader import load_data
from sentiment_analysis import plot_sentiment_trend
from wordcloud_generator import display_wordcloud
from topic_modeling import display_lda_topics

# Load Data (Batch Process for Large CSV)
data = load_data()

# Title and Description
st.set_page_config(layout="wide", page_title="Rock Lyrics Dashboard", page_icon="🎸")
st.title("🎸 Rock Lyrics Analysis Dashboard")
st.write("Explore rock music from 1950 to 2000 through lyrics analysis, sentiment, and topic modeling.")

# Sidebar Filters
st.sidebar.header("Filters")
if 'year' in data.columns:
    decades = st.sidebar.multiselect("Select Decades", data['year'].unique().tolist(), default=data['year'].unique())
    selected_artist = st.sidebar.multiselect("Select Artists", data['artist'].unique().tolist())
    word_count_filter = st.sidebar.slider("Max Word Count", min_value=50, max_value=600, value=600)
else:
    st.sidebar.warning("Column 'year' not found. Please check the CSV structure.")

# Apply Filters (Limit Data During Startup)
filtered_data = data[data['year'].isin(decades)] if 'year' in data.columns else data
filtered_data = filtered_data[filtered_data['lyric_length'] <= word_count_filter]
if selected_artist:
    filtered_data = filtered_data[filtered_data['artist'].isin(selected_artist)]

# Visualizations (Deferred Execution)
st.subheader("🎵 Number of Rock Songs by Year")
if 'year' in filtered_data.columns:
    yearly_counts = filtered_data.groupby('year').size()
    st.bar_chart(yearly_counts)

st.subheader("🎤 Top 10 Artists by Number of Songs")
top_artists = filtered_data['artist'].value_counts().head(10)
st.bar_chart(top_artists)

# Lazy Execution - Avoid Heavy Computation at Startup
if st.button("Generate Sentiment Analysis"):
    plot_sentiment_trend(filtered_data)

if st.button("Generate Word Cloud"):
    display_wordcloud(filtered_data)

if st.button("Run Topic Modeling"):
    display_lda_topics(filtered_data)
