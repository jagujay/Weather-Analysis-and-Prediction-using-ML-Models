from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
import openmeteo_requests
import pandas as pd
import os
import requests_cache
from retry_requests import retry

# Initialize API client with retry and cache
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def get_lat_lon(city_name):
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError(f"City '{city_name}' not found.")


# Function to fetch weather data
def fetch_weather_data(city_name, latitude, longitude):
    today = datetime.now()
    end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = "1990-01-01"

    # API parameters
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
            "daylight_duration", "precipitation_sum", "precipitation_hours", "wind_speed_10m_max"
        ],
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Extract daily data
    response = responses[0]
    daily = response.Daily()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
        "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
        "temperature_2m_mean": daily.Variables(2).ValuesAsNumpy(),
        "apparent_temperature_max": daily.Variables(3).ValuesAsNumpy(),
        "apparent_temperature_min": daily.Variables(4).ValuesAsNumpy(),
        "apparent_temperature_mean": daily.Variables(5).ValuesAsNumpy(),
        "daylight_duration": daily.Variables(6).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(7).ValuesAsNumpy(),
        "precipitation_hours": daily.Variables(8).ValuesAsNumpy(),
        "wind_speed_10m_max": daily.Variables(9).ValuesAsNumpy(),
    }

    daily_data = pd.DataFrame(daily_data)[daily_data["date"] >= "1990-01-01"]
    return daily_data


# Overwrite Check Function
def check_and_get_file(city_name, till_date, folder):
    """
    Check for an existing dataset for the city with the correct date.
    Return the file path if it exists; otherwise, delete older datasets.
    """
    city_name = city_name.capitalize()
    existing_files = [f for f in os.listdir(folder) if city_name in f]
    current_file = f"{city_name}_{till_date}.csv"

    # Check if the file with the same date already exists
    if current_file in existing_files:
        return os.path.join(folder, current_file)  # Return the existing file path

    # If old files exist with different dates, delete them
    for file in existing_files:
        os.remove(os.path.join(folder, file))
    return None  # No current file found, proceed to fetch new data


def clean_data(data):
    """
    Handle missing values in the dataset.
    - Drop rows with NaN or fill missing values using interpolation.
    """
    if data.isnull().values.any():
        print("Missing values detected. Cleaning the data...")
        # Fill missing values using interpolation
        data = data.interpolate(method='time', limit_direction='both')
        # If interpolation doesn't work, fill with the column mean
        data = data.fillna(data.mean())

        # Check if missing values still exist
        if data.isnull().values.any():
            print("Warning: Some missing values remain after cleaning.")
        else:
            print("Data cleaned successfully.")
    return data
