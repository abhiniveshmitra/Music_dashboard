import gensim
from gensim import corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit as st
import streamlit.components.v1 as components
import random

@st.cache_data
def perform_lda(data, num_topics=3):
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

    num_topics = st.slider("Select Number of Topics", min_value=2, max_value=5, value=3, key="lda_slider")

    if st.button("Run Topic Modeling", key="lda_button"):
        with st.spinner("Running LDA Topic Modeling..."):
            lda_vis = perform_lda(data, num_topics)
            st.session_state['lda_vis'] = pyLDAvis.prepared_data_to_html(lda_vis)

    if 'lda_vis' in st.session_state:
        components.html(st.session_state['lda_vis'], height=800, scrolling=True)
