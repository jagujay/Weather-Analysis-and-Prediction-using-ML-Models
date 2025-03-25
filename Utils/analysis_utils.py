import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def analyze_data(data, city_name):
    st.write(f"### Data Summary for {city_name}")
    st.write(data.describe())

    # Temperature Analysis
    st.write(f"### Temperature Trends in {city_name}")
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    axes[0].plot(data["date"], data["temperature_2m_max"], label="Max Temp", color="tab:blue")
    axes[0].plot(data["date"], data["temperature_2m_min"], label="Min Temp", color="tab:orange")
    axes[0].set_title("Max and Min Temperatures")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].legend()

    axes[1].plot(data["date"], data["temperature_2m_mean"], color='tab:red')
    axes[1].set_title("Mean Temperature")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[2].plot(data["date"], data["temperature_2m_mean"].rolling(window=7).mean(), color='tab:green')
    axes[2].set_title("7-Day Rolling Average Temperature")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for ax in axes:
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Temperature (°C)", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Precipitation Analysis
    st.write(f"### Precipitation Trends in {city_name}")
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    axes[0].plot(data["date"], data["precipitation_sum"], color="tab:blue")
    axes[0].set_title("Daily Precipitation")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    sns.histplot(data["precipitation_sum"].dropna(), kde=True, bins=30, ax=axes[1], color="tab:blue")
    axes[1].set_title("Precipitation Distribution")
    axes[1].set_xlabel("Precipitation (mm)")

    axes[2].plot(data["date"], data["precipitation_sum"].rolling(window=7).mean(), color="tab:green")
    axes[2].set_title("7-Day Rolling Average Precipitation")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for ax in axes:
        ax.set_xlabel("Year", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Daylight Analysis
    st.write(f"### Daylight Duration Trends in {city_name}")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data["date"], data["daylight_duration"], color="tab:orange")
    ax.set_title("Daylight Duration")
    ax.set_xlabel("Year")
    ax.set_ylabel("Duration (seconds)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Wind Speed Analysis
    st.write(f"### Wind Speed Analysis in {city_name}")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(data["date"], data["wind_speed_10m_max"], color="tab:purple")
    ax.set_title("Daily Wind Speed")
    ax.set_xlabel("Year")
    ax.set_ylabel("Wind Speed (km/h)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Seasonal Decomposition
    st.write(f"### Seasonal Decomposition for Temperature in {city_name}")
    decomposed = seasonal_decompose(data["temperature_2m_mean"].dropna(), model='additive', period=365)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(data["date"][:len(decomposed.observed)], decomposed.observed, label="Observed", color="tab:blue")
    axes[0].set_title("Observed")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[1].plot(data["date"][:len(decomposed.observed)], decomposed.trend, label="Trend", color="tab:orange")
    axes[1].set_title("Trend")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[2].plot(data["date"][:len(decomposed.observed)], decomposed.seasonal, label="Seasonal", color="tab:green")
    axes[2].set_title("Seasonal")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[3].plot(data["date"][:len(decomposed.observed)], decomposed.resid, label="Residual", color="tab:red")
    axes[3].set_title("Residual")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[3].set_xlabel("Year")

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    # Yearly Trends
    st.write(f"### Yearly Trends in {city_name}")
    data["year"] = data["date"].dt.year
    yearly_mean_temp = data.groupby("year")["temperature_2m_mean"].mean()
    yearly_precip = data.groupby("year")["precipitation_sum"].sum()

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    axes[0].plot(yearly_mean_temp.index, yearly_mean_temp, marker='o', color="tab:red")
    axes[0].set_title("Yearly Average Temperature")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Temperature (°C)")

    axes[1].plot(yearly_precip.index, yearly_precip, marker='o', color="tab:blue")
    axes[1].set_title("Yearly Total Precipitation")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Precipitation (mm)")

    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
