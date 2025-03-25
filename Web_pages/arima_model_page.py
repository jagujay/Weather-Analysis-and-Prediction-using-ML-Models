import numpy as np
import streamlit as st
import os
import pandas as pd
from Utils.arima_utils import arima_forecast, check_stationarity
from Utils.constants import rename_mapping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Utils.data_utils import clean_data


# ARIMA Model Page
def arima_model_page():
    st.title("ARIMA Model")

    # Load dataset
    city_files = [f for f in os.listdir("Datasets") if f.endswith(".csv")]
    city_names = {f.split('_')[0]: f for f in city_files}
    selected_city = st.selectbox("Select a City", list(city_names.keys()))

    if selected_city:
        file_path = os.path.join("Datasets", city_names[selected_city])
        data = pd.read_csv(file_path, parse_dates=["date"], index_col="date")

        # Rename features for presentability
        data.rename(columns=rename_mapping, inplace=True)

        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("ARIMA(p): Autoregressive Order", min_value=0, value=1)
        with col2:
            d = st.number_input("ARIMA(d): Differencing Order", min_value=0, value=1)
        with col3:
            q = st.number_input("ARIMA(q): Moving Average Order", min_value=0, value=1)
        future_days = st.slider("Select the number of future days to predict", 1, 30, 7)

        # Filter for specific features
        selected_features = list(rename_mapping.values())
        filtered_data = data[selected_features]
        filtered_data = clean_data(filtered_data)

        # Stationarity Check
        st.write("## Stationarity Check")
        stationarity_results = []
        for column in filtered_data.columns:
            result = check_stationarity(filtered_data[column])
            stationarity_results.append({"Feature": column, "ADF Statistic": result["ADF Statistic"],
                                         "p-value": result["p-value"]})
        stationarity_df = pd.DataFrame(stationarity_results)
        st.table(stationarity_df)

        # Train ARIMA
        st.write("## ARIMA Forecasts")
        forecasts, summaries, overall_metrics = arima_forecast(filtered_data, int(p), int(d), int(q), future_days)

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

        # Display as table
        st.write("### Evaluation Metrics Summary")
        st.table(metrics_df)

        # Overall Accuracy
        overall_accuracy = np.nanmean([
            metrics["Accuracy"] for feature, metrics in overall_metrics.items() if feature != "Total Precipitation"
        ])
        overall_mse = np.nanmean([metrics["MSE"] for metrics in overall_metrics.values()])

        # Display overall metrics
        st.write("### Overall Model Evaluation")
        st.write(f"- Overall Mean Squared Error (MSE): {overall_mse:.4f}")
        st.write(f"- Overall Accuracy: {overall_accuracy:.2f}%")

        # Save ARIMA summaries
        os.makedirs("Assets/ARIMA/Summaries", exist_ok=True)
        summary_df = pd.DataFrame(summaries)
        summary_file = os.path.join("Assets/ARIMA/Summaries", f"{selected_city}_ARIMA_Summary.csv")
        summary_df.to_csv(summary_file, index=False)
        st.success(f"ARIMA summaries saved to: {summary_file}")

        # Prepare predictions and save to file
        os.makedirs("Assets/ARIMA/Predictions", exist_ok=True)
        predictions = {}
        future_dates = pd.date_range(start=filtered_data.index[-1] + pd.Timedelta(days=1), periods=future_days)

        # Display forecasts and graphs
        for feature, forecast in forecasts.items():
            if forecast is not None:
                st.write(f"### {feature}")
                forecast_df = pd.DataFrame(forecast, index=future_dates).round(2)
                # Rename the column to match the feature name
                forecast_df.columns = [feature]

                predictions[feature] = forecast_df

                # Save predictions to CSV
                save_path = os.path.join("Assets/ARIMA/Predictions", f"{selected_city}_ARIMA_Predictions.csv")
                combined_forecast_df = pd.concat(predictions.values(), axis=1)
                combined_forecast_df.to_csv(save_path, index_label="date")
                st.success(f"Forecast saved to: {save_path}")

                # Display forecast
                st.dataframe(forecast_df)

                # Plot forecasts
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(future_dates, forecast, color="tab:orange", label="Predicted")
                ax.set_title(f"Forecast for {feature}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Values")
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)
