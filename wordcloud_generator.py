from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

def generate_wordcloud(data):
    st.subheader("🌬️ Word Cloud of Rock Lyrics by Decade")
    if 'year' in data.columns:
        decade_choice = st.selectbox("Select Decade for Word Cloud", data['year'].unique())
        lyrics_text = ' '.join(data[data['year'] == decade_choice]['lyrics'].dropna())
        
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(lyrics_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
