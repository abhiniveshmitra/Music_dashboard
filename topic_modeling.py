import gensim
from gensim import corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit as st
import random

@st.cache_data
def perform_lda(data, num_topics=3):
    # Subset to reduce memory load (random sample of 500 rows)
    if len(data) > 500:
        data = data.sample(500, random_state=42)
    
    # Preprocess and split lyrics
    lyrics = data['lyrics'].dropna().apply(lambda x: x.split())
    
    # Create Dictionary and Corpus
    dictionary = corpora.Dictionary(lyrics)
    corpus = [dictionary.doc2bow(text) for text in lyrics]
    
    # Train LDA Model
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    # Generate Visualization
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return vis

def display_lda_topics(data):
    st.subheader("🧩 Topic Modeling of Rock Lyrics (LDA)")
    
    # User inputs for customization
    num_topics = st.slider("Select Number of Topics", min_value=2, max_value=5, value=3)
    
    # Button to Trigger Topic Modeling
    if st.button("Run Topic Modeling"):
        with st.spinner("Running LDA Topic Modeling... This may take a few moments"):
            lda_vis = perform_lda(data, num_topics)
            pyLDAvis.save_html(lda_vis, 'lda_vis.html')
            st.success("LDA Visualization complete!")
            st.write("🔗 [View LDA Topic Visualization](lda_vis.html)")
