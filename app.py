import streamlit as st
from data_loader import load_data
from sentiment_analysis import search_sentiment_analysis, analyze_sentiment
from artist_comparison import compare_artists

st.set_page_config(layout="wide", page_title="Rock Lyrics Dashboard", page_icon="🎸")

data = load_data()

st.title("🎸 Rock Lyrics Analysis Dashboard")
st.write("Explore rock music through sentiment analysis, artist comparisons, and lyrical insights.")

# Sidebar Filters
if 'year' in data.columns:
    decades = st.sidebar.multiselect("Select Decades", data['year'].unique().tolist(), default=data['year'].unique())
else:
    st.sidebar.warning("Year column missing.")

filtered_data = data[data['year'].isin(decades)] if 'year' in data.columns else data

# Apply Sentiment Analysis Before Comparison
filtered_data = analyze_sentiment(filtered_data)

# Artist Sentiment Search
search_sentiment_analysis(filtered_data)
# Expanded Artist Comparison
if st.button("Compare Artists"):
    compare_artists(filtered_data)
