from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def generate_wordcloud(data, decade_choice):
    # 1. Filter for the selected decade and clean lyrics
    filtered_data = data[(data['year'] == decade_choice) & 
                         (data['lyrics'].notna()) & 
                         (data['lyrics'].str.strip() != '')]
    
    # 2. Combine lyrics into a single string
    lyrics_text = ' '.join(filtered_data['lyrics'])

    # 3. Handle Empty Lyrics Case
    if not lyrics_text.strip():  
        st.warning(f"No valid lyrics found for {decade_choice}. Please try a different year.")
        return None

    # 4. Generate the Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(lyrics_text)
    return wordcloud


def display_wordcloud(data):
    st.subheader("🌬️ Word Cloud of Rock Lyrics by Decade")

    # Dropdown to select decade
    if 'year' in data.columns:
        decade_choice = st.selectbox("Select Decade for Word Cloud", sorted(data['year'].unique()))

        # Generate and Display Word Cloud
        wordcloud = generate_wordcloud(data, decade_choice)
        if wordcloud:
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Word Cloud for {decade_choice}", fontsize=14)
            st.pyplot(plt)
