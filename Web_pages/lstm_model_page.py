import streamlit as st
import os
import pandas as pd
from Utils.lstm_utils import train_lstm_model
from Utils.constants import rename_mapping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Utils.data_utils import clean_data


# LSTM Model Page
def lstm_model_page():
    st.title("LSTM Model")
    st.write("### Select a City for LSTM")

    city_files = [f for f in os.listdir("Datasets") if f.endswith(".csv")]
    city_names = {f.split('_')[0]: f for f in city_files}
    selected_city = st.selectbox("Select City", list(city_names.keys()))

    future_days = st.slider("Select number of future days for prediction", 1, 30, 7)

    if selected_city:
        # Load data
        file_path = os.path.join("Datasets", city_names[selected_city])
        data = pd.read_csv(file_path, parse_dates=["date"])
        data.set_index("date", inplace=True)
        features = list(rename_mapping.keys())
        data = data[features]

        # Clean the data (handle missing values)
        data = clean_data(data)

        # Check for missing values
        if data.isnull().values.any():
            st.error("The dataset contains missing values. Please clean the data and try again.")
            return

        # Warning for unreliable forecasts
        if future_days > 14:
            st.warning("Note: Predictions beyond 14 days may be less reliable due to the chaotic "
                       "nature of weather systems.")

        # Train and Predict
        st.write("Training the LSTM model...")
        try:
            train_loss, test_loss, forecast, r2 = train_lstm_model(data.values, future_days, n_steps=30)

            # Prepare forecast DataFrame
            last_date = pd.to_datetime(data.index[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
            forecast_df = pd.DataFrame(forecast, columns=features, index=future_dates).rename(
                columns=rename_mapping).round(2)

            # Save predictions to assets/LSTM/Predictions
            os.makedirs("Assets/LSTM/Predictions", exist_ok=True)
            save_path = os.path.join("Assets/LSTM/Predictions", f"{selected_city}_LSTM_Predictions.csv")
            forecast_df.to_csv(save_path, index_label="date")
            st.success(f"Forecast saved to: {save_path}")

            # Save summaries to assets/LSTM/Summaries
            os.makedirs("Assets/LSTM/Summaries", exist_ok=True)
            summary_path = os.path.join("Assets/LSTM/Summaries", f"{selected_city}_LSTM_Summary.csv")
            summary_df = pd.DataFrame({
                "Metric": ["Train Loss", "Test Loss"],
                "Value": [round(train_loss, 4), round(test_loss, 4)]
            })
            summary_df.to_csv(summary_path, index=False)
            st.success(f"Summary saved to: {summary_path}")

            # Display results
            st.write(f"Train Loss: {train_loss:.4f}")
            st.write(f"Test Loss: {test_loss:.4f}")
            st.write(f"Coefficient of Determination (R² value): {r2}%")
            st.write("### Future Weather Forecast")
            st.dataframe(forecast_df)

            # Plot Forecasts
            for feature in forecast_df.columns:
                st.write(f"**{feature} Forecast**")
                fig, ax = plt.subplots(figsize=(15,10))
                ax.plot(future_dates, forecast_df[feature], label=feature, color="tab:blue")
                ax.set_title(f"LSTM Forecast for {feature}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Values")
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                ax.legend()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
