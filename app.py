import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cache data to optimize performance
@st.cache_data
def load_data():
    # Google Drive File ID extracted from the link
    file_id = "1BQqV3fdFJYVQU8qhfhwAjn5DDQ2SBHhR"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Load the CSV directly from Google Drive
    return pd.read_csv(url)

# Load the dataset
data = load_data()

# Title and Description
st.title("🎸 Rock Lyrics Analysis Dashboard")
st.write("Explore rock music from 1950 to 2000 through lyrics analysis, sentiment, and topic modeling.")

# Sidebar Filters
st.sidebar.header("Filters")
decades = st.sidebar.multiselect("Select Decades", data['year'].unique().tolist(), default=data['year'].unique())
selected_artist = st.sidebar.multiselect("Select Artists", data['artist'].unique().tolist())
word_count_filter = st.sidebar.slider("Max Word Count", min_value=50, max_value=600, value=600)

# Apply Filters
filtered_data = data[data['year'].isin(decades)]
filtered_data = filtered_data[filtered_data['lyric_length'] <= word_count_filter]
if selected_artist:
    filtered_data = filtered_data[filtered_data['artist'].isin(selected_artist)]

# Visualization 1 – Yearly Distribution of Songs
st.subheader("🎵 Number of Rock Songs by Year")
yearly_counts = filtered_data.groupby('year').size()
fig, ax = plt.subplots()
yearly_counts.plot(kind='bar', ax=ax)
plt.xlabel("Year")
plt.ylabel("Number of Songs")
st.pyplot(fig)

# Visualization 2 – Top 10 Artists
st.subheader("🎤 Top 10 Artists by Number of Songs")
top_artists = filtered_data['artist'].value_counts().head(10)
fig, ax = plt.subplots()
top_artists.plot(kind='bar', ax=ax)
plt.xlabel("Artist")
plt.ylabel("Number of Songs")
st.pyplot(fig)

# Visualization 3 – Sentiment Distribution by Year
st.subheader("😊 Sentiment Distribution (if available)")
if 'sentiment' in filtered_data.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='year', y='sentiment', data=filtered_data, ax=ax)
    plt.xticks(rotation=45)
    plt.title("Sentiment Distribution by Year")
    st.pyplot(fig)
else:
    st.write("Sentiment data not available in this dataset.")

# Show Raw Data (Optional)
st.subheader("🗂 Explore the Data")
st.dataframe(filtered_data)
