import streamlit as st

# Must be the first Streamlit command:
st.set_page_config(layout="wide", page_title="Rock Lyrics Dashboard", page_icon="🎸")

import pandas as pd
from loader import load_data
from sentiment_analysis import search_sentiment_analysis, analyze_sentiment
from artist_comparison import compare_artists

def main():
    # --------------------
    # Data Loading
    # --------------------
    data = load_data()

    # --------------------
    # Header Section – Introduction
    # --------------------
    st.markdown(
        """
        <style>
        .big-font {
            font-size:40px !important;
            text-align: center;
            font-weight: bold;
            color: #ff6347;
        }
        .subheader {
            font-size:22px !important;
            text-align: center;
            font-weight: bold;
            color: #1e90ff;
        }
        .highlight {
            font-size:18px;
            font-style: italic;
            text-align: center;
            color: #555;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="big-font">🎸 The Evolution of Rock Music (1950 - 2000)</p>', unsafe_allow_html=True)
    st.markdown('<p class="highlight">Explore how rock music evolved over the decades – from lyrical sentiment to artist comparisons.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # --------------------
    # Sidebar – Filters
    # --------------------
    st.sidebar.header("🎚 Filters")
    data['decade'] = data['year'].apply(lambda x: (x // 10) * 10)

    # Decade Filter
    available_decades = sorted(data['decade'].unique())
    selected_decades = st.sidebar.multiselect("Filter by Decades", available_decades, default=available_decades)

    # Artist Selection – Two Artists Only
    artist_options = data['artist'].unique()
    selected_artists = st.sidebar.multiselect("Select Two Artists to Compare", artist_options, max_selections=2, default=artist_options[:2])

    # Word Count Filter
    word_count_filter = st.sidebar.slider("Max Word Count", min_value=50, max_value=600, value=600)

    # --------------------
    # Filter Data
    # --------------------
    filtered_data = data[data['decade'].isin(selected_decades)]
    filtered_data = filtered_data[filtered_data['lyric_length'] <= word_count_filter]

    if selected_artists and len(selected_artists) == 2:
        filtered_data = filtered_data[filtered_data['artist'].isin(selected_artists)]
        # Ensure Sentiment is Applied
        if 'sentiment' not in filtered_data.columns:
            filtered_data = analyze_sentiment(filtered_data)
    else:
        st.sidebar.error("Please select exactly **two artists** for comparison.")

    # --------------------
    # Title and Main Section
    # --------------------
    st.markdown('<p class="subheader">🎵 Rock Music Through the Years</p>', unsafe_allow_html=True)
    st.write("Discover the number of rock songs released over the years and the artists who defined this era.")

    # --------------------
    # Visualization 1 – Yearly Song Distribution
    # --------------------
    st.subheader("📅 Number of Rock Songs by Year")
    st.markdown("This graph shows the distribution of rock songs released each year. Use the filters on the left to narrow down by decade or artist.")

    yearly_counts = filtered_data.groupby('year').size()
    yearly_counts_df = pd.DataFrame({'Year': yearly_counts.index, 'Count': yearly_counts.values})
    st.bar_chart(yearly_counts_df.set_index('Year'), use_container_width=True)

    # --------------------
    # Visualization 2 – Most Popular Artists by Cumulative Listens (in Millions)
    # --------------------
    st.subheader("🔥 Most Popular Artists by Cumulative Listens (Millions)")
    st.markdown("This chart displays the top artists with the highest cumulative listens (views). Popularity is calculated across **all artists**, not limited to selected filters.")

    # Aggregate listens globally and convert to millions
    top_popular_artists = (
        data.groupby('artist')['views'].sum()
        .sort_values(ascending=False)
        .head(10)
        .apply(lambda x: round(x / 1_000_000, 2))  # Convert to millions
    )
    top_popular_df = pd.DataFrame({'Artist': top_popular_artists.index, 'Views (M)': top_popular_artists.values})
    st.bar_chart(top_popular_df.set_index('Artist'), use_container_width=True)

    # --------------------
    # Artist Comparison Section
    # --------------------
    if len(selected_artists) == 2:
        st.subheader("🎤 Compare Two Rock Legends")
        st.markdown("""
        Select two artists from the sidebar to compare their lyrical diversity, most popular songs, and emotional tone.  
        This section dives deep into how two artists' styles contrast across different time periods.
        """)
        compare_artists(filtered_data)

if __name__ == "__main__":
    main()
