import streamlit as st
import pandas as pd
import gdown
from zipfile import ZipFile
import os

@st.cache_data
def load_data():
    file_id = "1bw3EvezRiUj9sV3vTT6OtY840pxcPpW1"
    zip_output = 'ezyzip.zip'
    csv_output = 'filtered_rock_1950_2000_cleaned.csv'
    
    # Check if CSV already exists
    if os.path.exists(csv_output):
        st.info("Using cached CSV file.")
        data = pd.read_csv(csv_output)
    else:
        st.info("Downloading ZIP from Google Drive...")
        gdown.download(f'https://drive.google.com/uc?id={file_id}&confirm=t', zip_output, quiet=False)
        
        # Unzip the File
        with ZipFile(zip_output, 'r') as zip_ref:
            zip_ref.extractall()
            st.success("File unzipped successfully!")
        
        # Load CSV
        data = pd.read_csv(csv_output)
    
    # Filter for English Songs (language == 'en')
    if 'language' in data.columns:
        data = data[data['language'] == 'en']
        st.success("Filtered to English-only songs.")
    else:
        st.warning("Language column missing. Unable to filter non-English songs.")
    
    return data
