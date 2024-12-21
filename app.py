import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    # Google Drive File ID extracted from your link
    file_id = "170b9XFkXNTt9i9nT6C-Rx1DQ4nmhpKCa"
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

data = load_data()

# Title and Description
st.title("🎸 Rock Lyrics Analysis Dashboard")
st.write("Explore the evolution of rock music through sentiment, topic modeling, and artist contributions.")

# Sidebar Filters
st.sidebar.header("Filters")
#decades = st.sidebar.multiselect("Select Decades", data['year'].unique().tolist(), default=data['year'].unique())
selected_topic = st.sidebar.selectbox("Select Topic", ['All'] + sorted(data['dominant_topic'].unique().tolist()))
filter_by_rock = st.sidebar.checkbox("Show Only Rock Songs", value=True)
filter_by_wordcount = st.sidebar.slider("Max Word Count", min_value=50, max_value=1000, value=600)

# Filter Data
filtered_data = data[data['year'].isin(decades)]

if selected_topic != 'All':
    filtered_data = filtered_data[filtered_data['dominant_topic'] == selected_topic]

# Apply Rock Filter
if filter_by_rock:
    filtered_data = filtered_data[filtered_data['tag'] == 'rock']

# Apply Word Count Filter
filtered_data['lyric_length'] = filtered_data['lyrics'].apply(lambda x: len(str(x).split()))
filtered_data = filtered_data[filtered_data['lyric_length'] <= filter_by_wordcount]

# Visualization 1: Topic Distribution Over Time
st.subheader("🎶 Topic Distribution Over Decades")
topic_over_time = filtered_data.groupby(['year', 'dominant_topic']).size().unstack(fill_value=0)
st.line_chart(topic_over_time)

# Visualization 2: Sentiment Distribution by Topic
st.subheader("😊 Sentiment by Topic")
st.write(sns.boxplot(x='dominant_topic', y='sentiment', data=filtered_data).figure)

# Visualization 3: Top Artists by Topic Contribution
st.subheader("🎤 Top Artists by Topic Contribution")
artist_topics = filtered_data.groupby('artist')['dominant_topic'].value_counts().unstack(fill_value=0)
top_artists = artist_topics.sum(axis=1).sort_values(ascending=False).head(20)
artist_topics = artist_topics.loc[top_artists.index]
st.bar_chart(artist_topics)

# Show Raw Data (Optional)
st.subheader("🎼 Explore the Data")
st.dataframe(filtered_data)
