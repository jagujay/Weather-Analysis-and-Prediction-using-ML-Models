import streamlit as st
import os
import pandas as pd
from Utils.sarima_utils import sarima_forecast
from Utils.constants import rename_mapping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Utils.data_utils import clean_data
import numpy as np


# SARIMA Page
def sarima_model_page():
    st.title("SARIMA Model")
    st.write("### Select a City for SARIMA")

    # File Selection
    city_files = [f for f in os.listdir("Datasets") if f.endswith(".csv")]
    city_names = {f.split('_')[0]: f for f in city_files}
    selected_city = st.selectbox("Select a City", list(city_names.keys()))

    if selected_city:
        file_path = os.path.join("Datasets", city_names[selected_city])
        data = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

        # Rename features for better display
        data = data.rename(columns=rename_mapping)
        st.write("### Set SARIMA Parameters")

        # SARIMA Parameters Layout
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("Non-Seasonal AR(p)", min_value=0, value=1)
            P = st.number_input("Seasonal AR(P)", min_value=0, value=1)
        with col2:
            d = st.number_input("Non-Seasonal Differencing(d)", min_value=0, value=1)
            D = st.number_input("Seasonal Differencing(D)", min_value=0, value=1)
        with col3:
            q = st.number_input("Non-Seasonal MA(q)", min_value=0, value=1)
            Q = st.number_input("Seasonal MA(Q)", min_value=0, value=1)
        m = st.number_input("Seasonal Period(m)", min_value=1, value=12)
        future_days = st.slider("Future Days to Predict", 1, 30, 7)

        # Filter relevant features
        selected_features = ["Mean Temperature", "Feels-Like Temperature",
                             "Total Precipitation", "Daylight Duration", "Max Wind Speed"]
        filtered_data = data[selected_features]
        filtered_data = clean_data(filtered_data)

        # Forecast using SARIMA
        st.write("## SARIMA Forecasts")
        forecasts, summaries, overall_metrics = sarima_forecast(filtered_data, int(p), int(d), int(q),
                                               int(P), int(D), int(Q), int(m), future_days)

        # Prepare Metrics data for display in a table
        table_data = {
            "Feature": [],
            "MSE": [],
            "Accuracy (%)": []
        }

        for feature, metrics in overall_metrics.items():
            table_data["Feature"].append(feature)
            table_data["MSE"].append(metrics["MSE"])
            if feature == "Total Precipitation":
                table_data["Accuracy (%)"].append(np.nan)
            else:
                table_data["Accuracy (%)"].append(metrics["Accuracy"])

        # Convert to DataFrame
        metrics_df = pd.DataFrame(table_data)

        # Overall Accuracy
        overall_accuracy = np.nanmean([
            metrics["Accuracy"] for feature, metrics in overall_metrics.items() if feature != "Total Precipitation"
        ])
        overall_mse = np.nanmean([metrics["MSE"] for metrics in overall_metrics.values()])

        # Display overall metrics
        st.write("### Overall Model Evaluation")
        st.write(f"- Overall Mean Squared Error (MSE): {overall_mse:.4f}")
        st.write(f"- Overall Accuracy: {overall_accuracy:.2f}%")

        # Display as table
        st.write("### Evaluation Metrics Summary")
        st.table(metrics_df)

        # Save summaries to CSV
        os.makedirs("Assets/SARIMA/Summaries", exist_ok=True)
        summary_df = pd.DataFrame(summaries)
        summary_file = os.path.join("Assets/SARIMA/Summaries", f"{selected_city}_SARIMA_Summary.csv")
        summary_df.to_csv(summary_file, index_label="Feature")
        st.success(f"SARIMA summaries saved to: {summary_file}")

        # Save forecasts to CSV
        os.makedirs("Assets/SARIMA/Predictions", exist_ok=True)
        predictions_file = os.path.join("Assets/SARIMA/Predictions", f"{selected_city}_SARIMA_Predictions.csv")
        combined_forecasts = pd.DataFrame({feature: forecast for feature, forecast in forecasts.items() if forecast is not None}, index=pd.date_range(start=filtered_data.index[-1] + pd.Timedelta(days=1), periods=future_days))
        combined_forecasts.to_csv(predictions_file, index_label="date")
        st.success(f"SARIMA forecasts saved to: {predictions_file}")

        # Display Forecasts and Plots
        for feature, forecast in forecasts.items():
            if forecast is not None:
                st.write(f"### Forecast for {feature}")
                forecast_df = pd.DataFrame(forecast, index=pd.date_range(start=filtered_data.index[-1] + pd.Timedelta(days=1), periods=future_days)).round(2)
                forecast_df.columns = [feature]  # Align column name
                st.dataframe(forecast_df)

                # Plot forecast
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(forecast_df.index, forecast_df[feature], color="orange", label="Forecast")
                ax.set_title(f"SARIMA Forecast for {feature}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Values")
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)