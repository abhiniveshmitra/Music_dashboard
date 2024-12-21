from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def generate_wordcloud(data, decade_choice):
    filtered_data = data[(data['year'] == decade_choice) & (data['lyrics'].notna()) & (data['lyrics'].str.strip() != '')]
    lyrics_text = ' '.join(filtered_data['lyrics'])
    
    if not lyrics_text:
        st.warning(f"No valid lyrics found for {decade_choice}.")
        return None

    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(lyrics_text)
    return wordcloud

def display_wordcloud(data):
    st.subheader("🌬️ Word Cloud by Decade")
    if 'year' in data.columns:
        decade_choice = st.selectbox("Select Decade", sorted(data['year'].unique()))
        wordcloud = generate_wordcloud(data, decade_choice)
        if wordcloud:
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
