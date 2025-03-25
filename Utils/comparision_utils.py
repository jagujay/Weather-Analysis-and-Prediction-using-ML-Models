import os
import pandas as pd
import streamlit as st


def load_predictions(city, folder, model_type):
    """Load predictions for a given model type."""
    file_path = os.path.join(folder, f"{city}_{model_type}_Predictions.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=["date"], index_col="date")
    else:
        st.warning(f"No predictions found for {model_type}. Run the {model_type} model first.")
        return None
