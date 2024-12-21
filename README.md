# 🎸 Rock Lyrics Analysis Dashboard  
### Explore the Evolution of Rock Music (1950-2000)  

## 📌 Project Overview  
The **Rock Lyrics Analysis Dashboard** leverages **Natural Language Processing (NLP)** to explore the evolution of rock music between **1950 and 2000**. This interactive tool analyzes lyrical sentiment, word frequency, and artist comparisons to uncover trends in emotional tone, vocabulary diversity, and artist popularity.  

By applying **NLP techniques** to lyrical data, this project reveals how **rock music's themes, moods, and complexity** have shifted over the decades.  

---

## 📊 Dataset Information  
- The dataset was **sourced from Kaggle**, containing song lyrics scraped from **Genius.com**.  
- The original dataset was preprocessed to filter **rock songs** from **1950 to 2000**, resulting in a **cleaned CSV** for analysis.  

---

## 🚀 Features  
### 🎵 1. Yearly Song Distribution  
- Displays the **number of rock songs** released each year.  
- **Interactive filtering** by decade to analyze specific time periods.  

### 🔥 2. Most Popular Artists (by Listens)  
- Shows the **top 10 artists** with the highest cumulative listens (views).  
- Values are formatted in **millions** for clarity.  

### 🎭 3. Sentiment Analysis (NLP)  
- **NLP-powered sentiment analysis** evaluates the emotional tone of rock lyrics over the years.  
- Lyrics are analyzed to determine whether the sentiment is **positive, negative, or neutral**.  
- **Visualizes sentiment trends** to uncover how rock music has reflected joy, sadness, or rebellion throughout the decades.  

### 🎤 4. Artist Comparison (NLP and Complexity Analysis)  
- Compare **two artists** to visualize differences in:  
  - **Most popular songs** (by views).  
  - **Positive and negative songs** (based on NLP sentiment scores).  
  - **Lexical complexity** – NLP measures the uniqueness of vocabulary used by different artists.  

---

## 📊 How NLP Powers the Dashboard  
### 1. **Sentiment Analysis**  
- The project uses **TextBlob (NLP library)** to perform sentiment analysis on lyrics.  
- Each lyric is analyzed using **TextBlob’s polarity score**:  
  - **Positive Sentiment** – Lyrics convey joy, love, or excitement.  
  - **Negative Sentiment** – Lyrics express sadness, anger, or frustration.  
  - **Neutral Sentiment** – Balanced or factual lyrics.  
- **Visualization**:  
  - A **line chart** shows **sentiment over time** for filtered songs and artists.  

---

### 2. **Word Frequency Analysis (NLP)**  
- By tokenizing lyrics and applying **stopword filtering**, the dashboard identifies the **most frequently used words** by each artist.  
- **NLP filters out common stopwords** (e.g., "the", "and", "is") to highlight meaningful words that define an artist's lyrical style.  

---

### 3. **Lexical Complexity (Unique Vocabulary Measurement)**  
- Lexical complexity is calculated by analyzing the **diversity of words** used in lyrics.  
- A higher score indicates **greater vocabulary variety**.  
- This is achieved by measuring the **ratio of unique words to total words** in each lyric.  

---

## 📈 How It Works  
1. **Load the Data** – Preprocessed rock songs from 1950 to 2000 are imported into the dashboard.  
2. **Apply NLP Sentiment Analysis** – Each lyric is analyzed for **sentiment polarity** (positive, neutral, negative).  
3. **Filter by Decade/Artist** – Use the sidebar to explore specific decades and compare two artists.  
4. **Visualize Trends** – Dynamic bar charts and line graphs visualize how **sentiment and complexity** evolve over time.  
5. **Compare Two Artists** – Side-by-side comparisons reveal differences in lyrical style, complexity, and popularity.  

---

## 📂 Project Structure  

