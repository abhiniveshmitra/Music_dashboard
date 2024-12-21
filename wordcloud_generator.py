from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def generate_wordcloud(data, decade_choice):
    filtered_data = data[
        (data['year'] == decade_choice) & 
        (data['lyrics'].notna()) & 
        (data['lyrics'].str.strip() != '')
    ]
    lyrics_text = ' '.join(filtered_data['lyrics'])
    
    if not lyrics_text:
        st.warning(f"No valid lyrics found for {decade_choice}. Please select another decade.")
        return None

    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(lyrics_text)
    return wordcloud

def display_wordcloud(data):
    st.subheader("🌬️ Word Cloud of Rock Lyrics by Decade")

    if 'year' in data.columns:
        decade_choice = st.selectbox(
            "Select Decade for Word Cloud",
            options=[None] + sorted(data['year'].unique()),
            format_func=lambda x: "Select a Decade" if x is None else str(x)
        )

        if st.button("Generate Word Cloud", key="generate_wordcloud"):
            wordcloud = generate_wordcloud(data, decade_choice)
            
            # Store the generated word cloud in session state
            st.session_state['wordcloud'] = wordcloud

    # Display the word cloud if it exists in session state
    if 'wordcloud' in st.session_state and st.session_state['wordcloud']:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(st.session_state['wordcloud'], interpolation='bilinear')
        ax.axis("off")
        plt.title(f"Word Cloud for {decade_choice}", fontsize=14, color='white')
        st.pyplot(fig)
