import streamlit as st
import os
import pandas as pd
from datetime import datetime, timedelta
from Utils.data_utils import check_and_get_file, fetch_weather_data, get_lat_lon
from Utils.analysis_utils import analyze_data


def data_analysis_page():
    st.title("Data Analysis")
    city_name = st.text_input("Enter City Name:").capitalize()
    if city_name:
        folder = "Datasets"
        os.makedirs(folder, exist_ok=True)
        today = datetime.now()
        till_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")

        # Check for existing dataset with correct date
        file_name = check_and_get_file(city_name, till_date, folder)

        if file_name:
            st.write(f"Data for {city_name} till {till_date} already exists. Loading from file...")
            data = pd.read_csv(file_name, parse_dates=["date"])
        else:
            try:
                latitude, longitude = get_lat_lon(city_name)
                st.write(f"Fetching data for {city_name}...")
                data = fetch_weather_data(city_name, latitude, longitude)
                file_name = os.path.join(folder, f"{city_name}_{till_date}.csv")
                data.to_csv(file_name, index=False)
                st.write(f"Data saved to {file_name}.")
            except Exception as e:
                st.error(f"Error: {e}")
                return

        # Perform analysis
        analyze_data(data, city_name)
