import streamlit as st
import os
import pandas as pd
from Utils.comparision_utils import load_predictions
import matplotlib.pyplot as plt


def model_comparison_page():
    """Page to compare LSTM, ARIMA, and SARIMA predictions."""
    st.title("Model Comparison")
    st.write("### Compare Predictions of LSTM, ARIMA, and SARIMA Models")

    # Select City and Feature
    city_files = [f for f in os.listdir("Assets/LSTM/Predictions") if f.endswith("_LSTM_Predictions.csv")]
    city_names = {f.split('_')[0]: f for f in city_files}
    selected_city = st.selectbox("Select a City for Comparison", list(city_names.keys()))

    selected_feature = st.selectbox("Select a Feature to Compare",
                                    ["Mean Temperature", "Feels-Like Temperature",
                                     "Total Precipitation", "Daylight Duration", "Max Wind Speed"])

    if selected_city and selected_feature:
        try:
            # File Paths
            lstm_file = os.path.join("Assets/LSTM/Predictions", f"{selected_city}_LSTM_Predictions.csv")
            arima_file = os.path.join("Assets/ARIMA/Predictions", f"{selected_city}_ARIMA_Predictions.csv")
            sarima_file = os.path.join("Assets/SARIMA/Predictions", f"{selected_city}_SARIMA_Predictions.csv")

            # Load Data
            lstm_data = load_predictions(selected_city, "Assets/LSTM/Predictions", "LSTM")
            arima_data = load_predictions(selected_city, "Assets/ARIMA/Predictions", "ARIMA")
            sarima_data = load_predictions(selected_city, "Assets/SARIMA/Predictions", "SARIMA")

            # Check if all data is loaded
            if lstm_data is None or arima_data is None or sarima_data is None:
                st.warning("One or more models are missing predictions.")
                return

            # Combine Data
            comparison_df = pd.DataFrame({
                "LSTM": lstm_data[selected_feature],
                "ARIMA": arima_data[selected_feature],
                "SARIMA": sarima_data[selected_feature]
            }).round(2)

            # Display Comparison Table
            st.write("### Model Comparison Table")
            st.dataframe(comparison_df)

            # Save Combined Comparison Table
            os.makedirs("Assets/Comparisons", exist_ok=True)
            comparison_file = os.path.join("Assets/Comparisons", f"{selected_city}_Model_Comparison.csv")
            comparison_df.to_csv(comparison_file, index_label="date")
            st.success(f"Comparison table saved to: {comparison_file}")

            # Plot Comparison
            st.write("### Comparison Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(comparison_df.index, comparison_df["LSTM"], label="LSTM", linestyle='--', color='tab:blue')
            ax.plot(comparison_df.index, comparison_df["ARIMA"], label="ARIMA", linestyle='-.', color='tab:orange')
            ax.plot(comparison_df.index, comparison_df["SARIMA"], label="SARIMA", linestyle=':', color='tab:green')
            ax.set_title(f"Comparison of {selected_feature}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Values")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error loading comparison data: {e}")
