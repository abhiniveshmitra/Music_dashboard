import streamlit as st
import pandas as pd
import gdown
from io import BytesIO
from zipfile import ZipFile
import os

@st.cache_data
def load_data():
    # Google Drive ZIP file ID
    file_id = "1bw3EvezRiUj9sV3vTT6OtY840pxcPpW1"
    zip_output = 'ezyzip.zip'
    csv_output = 'filtered_rock_1950_2000_cleaned.csv'
    
    # Use gdown to bypass Google Drive virus scan warning
    if not os.path.exists(csv_output):
        gdown.download(f'https://drive.google.com/uc?id={file_id}&confirm=t', zip_output, quiet=False)

        # Unzip the file
        with ZipFile(zip_output, 'r') as zip_ref:
            zip_ref.extractall()
            st.success("File unzipped successfully!")

    # Load CSV
    return pd.read_csv(csv_output)

# Load data
data = load_data()

# Title and Description
st.title("🎸 Rock Lyrics Analysis Dashboard")
st.write("Explore rock music from 1950 to 2000 through lyrics analysis, sentiment, and topic modeling.")

# Sidebar Filters
st.sidebar.header("Filters")
if 'year' in data.columns:
    decades = st.sidebar.multiselect("Select Decades", data['year'].unique().tolist(), default=data['year'].unique())
    selected_artist = st.sidebar.multiselect("Select Artists", data['artist'].unique().tolist())
    word_count_filter = st.sidebar.slider("Max Word Count", min_value=50, max_value=600, value=600)
else:
    st.sidebar.warning("Column 'year' not found. Please check the CSV structure.")

# Apply Filters
filtered_data = data[data['year'].isin(decades)] if 'year' in data.columns else data
filtered_data = filtered_data[filtered_data['lyric_length'] <= word_count_filter]
if selected_artist:
    filtered_data = filtered_data[filtered_data['artist'].isin(selected_artist)]

# Visualization 1 – Yearly Distribution of Songs
st.subheader("🎵 Number of Rock Songs by Year")
if 'year' in filtered_data.columns:
    yearly_counts = filtered_data.groupby('year').size()
    st.bar_chart(yearly_counts)
else:
    st.write("Year data not available for visualization.")

# Visualization 2 – Top 10 Artists
st.subheader("🎤 Top 10 Artists by Number of Songs")
top_artists = filtered_data['artist'].value_counts().head(10)
st.bar_chart(top_artists)

# Show Raw Data
st.subheader("🗂 Explore the Data")
st.dataframe(filtered_data)
