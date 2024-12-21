import gensim
from gensim import corpora
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import streamlit as st

@st.cache_data
def perform_lda(data):
    lyrics = data['lyrics'].dropna().apply(lambda x: x.split())
    dictionary = corpora.Dictionary(lyrics)
    corpus = [dictionary.doc2bow(text) for text in lyrics]
    lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    return vis

def display_lda_topics(data):
    st.subheader("🧩 Topic Modeling of Rock Lyrics")
    lda_vis = perform_lda(data)
    pyLDAvis.save_html(lda_vis, 'lda_vis.html')
    st.write("🔗 [View LDA Topic Visualization](lda_vis.html)")
