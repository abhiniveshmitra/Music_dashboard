import gensim
from gensim import corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit as st
import streamlit.components.v1 as components
import random

@st.cache_data
def perform_lda(data, num_topics=3):
    # Subset to reduce memory load (random sample of 500 rows)
    if len(data) > 500:
        data = data.sample(500, random_state=42)
    
    lyrics = data['lyrics'].dropna().apply(lambda x: x.split())
    dictionary = corpora.Dictionary(lyrics)
    corpus = [dictionary.doc2bow(text) for text in lyrics]
    
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return vis

def display_lda_topics(data):
    st.subheader("🧩 Topic Modeling of Rock Lyrics (LDA)")

    # Customizable Topic Slider
    num_topics = st.slider("Select Number of Topics", min_value=2, max_value=5, value=3, key="lda_slider")

    # Unique Button Key to Prevent Duplicate Errors
    if st.button("Run Topic Modeling", key="lda_button"):
        with st.spinner("Running LDA Topic Modeling... This may take a few moments"):
            lda_vis = perform_lda(data, num_topics)

            # Convert LDA visualization to HTML
            html_vis = pyLDAvis.prepared_data_to_html(lda_vis)
            
            # Display directly in Streamlit
            components.html(html_vis, height=800, scrolling=True)
            st.success("LDA Visualization complete!")
