import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    # Use relative path — works on any machine including Streamlit Cloud
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
    
    # Load the CSV file
    df_weather = pd.read_csv(file_path, low_memory=False)

    # Convert any column with 'date' in its name to datetime (date only)
    for col in df_weather.columns:
        if 'datetime' in col.lower():
            df_weather[col] = pd.to_datetime(df_weather[col], errors='coerce').dt.date

    return df_weather