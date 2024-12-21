from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def generate_wordcloud(data, decade_choice):
    st.subheader("☁️ Word Cloud of Rock Lyrics by Decade")

    # Dropdown to select the decade
    decade_choice = st.selectbox("Select Decade for Word Cloud", data['year'].unique(), key="wordcloud_decade")

    # Filter data
    filtered_data = data[data['year'] == decade_choice]

    if filtered_data.empty:
        st.warning("No data found for the selected decade.")
        return

    lyrics = " ".join(filtered_data['lyrics'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="black").generate(lyrics)

    # Display Word Cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
