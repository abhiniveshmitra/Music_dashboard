import streamlit as st
import pandas as pd
import gdown
from zipfile import ZipFile
import os

@st.cache_data
def load_data():
    """
    Downloads and loads the CSV (if not already present).
    Filters only English lyrics.
    """
    file_id = "1bw3EvezRiUj9sV3vTT6OtY840pxcPpW1"
    zip_output = 'ezyzip.zip'
    csv_output = 'filtered_rock_1950_2000_cleaned.csv'
    
    # Download ZIP from Google Drive (if needed)
    if not os.path.exists(csv_output):
        gdown.download(f'https://drive.google.com/uc?id={file_id}&confirm=t', zip_output, quiet=False)
        with ZipFile(zip_output, 'r') as zip_ref:
            zip_ref.extractall()
            st.success("File unzipped successfully!")
    
    # Load CSV
    data = pd.read_csv(csv_output)
    
    # Filter to English Lyrics Only
    data = data[data['language'] == 'en']
    
    return data
