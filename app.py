import streamlit as st
import pandas as pd
import gdown
from io import BytesIO
from zipfile import ZipFile
import requests

@st.cache_data
def load_data():
    # Google Drive ZIP file ID (Replace with the new file ID)
    file_id = "1bw3EvezRiUj9sV3vTT6OtY840pxcPpW1"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Stream the ZIP file from Google Drive
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with ZipFile(BytesIO(response.content)) as zip_ref:
            # Extract CSV filename from ZIP
            csv_filename = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
            with zip_ref.open(csv_filename) as file:
                data = pd.read_csv(file)  # Load CSV directly into pandas
            st.success(f"Loaded {csv_filename} successfully!")
            return data
    else:
        st.error("Failed to download the file.")
        return pd.DataFrame()  # Return empty DataFrame if download fails

# Load the dataset
data = load_data()

# Debugging: Display columns
st.write("Columns in DataFrame:", data.columns.tolist())

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
