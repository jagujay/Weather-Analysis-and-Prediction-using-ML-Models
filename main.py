import streamlit as st
from Web_pages.about_page import about_page
from Web_pages.data_analysis_page import data_analysis_page
from Web_pages.lstm_model_page import lstm_model_page
from Web_pages.arima_model_page import arima_model_page
from Web_pages.sarima_model_page import sarima_model_page
from Web_pages.model_comparision_page import model_comparison_page


def main():
    # Set page config
    st.set_page_config(page_title="Weather Analysis and Prediction",
                       page_icon="🌤️",
                       initial_sidebar_state="expanded")

    # Set page configuration
    # st.set_page_config(
    #     page_title="Weather Analysis and Prediction",
    #     page_icon="🌤️",
    #     layout="wide",
    #     initial_sidebar_state="expanded"
    # )

    # Inject custom CSS
    # st.markdown(
    #     """
    #     <style>
    #         .main .block-container {
    #             max-width: 100%;
    #             padding: 2rem;
    #         }
    #         h1, h2, h3, h4 {
    #             font-size: 2rem !important;
    #         }
    #         body {
    #             font-size: 1.1rem;
    #         }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    st.sidebar.title("Weather Analysis and Prediction")
    page = st.sidebar.radio("Try it out!", ["About", "Data Analysis", "LSTM Model", "ARIMA Model",
                                                "SARIMA Model", "Model Comparison"])
    if page == "About":
        about_page()
    elif page == "Data Analysis":
        data_analysis_page()
    elif page == "LSTM Model":
        lstm_model_page()
    elif page == "ARIMA Model":
        arima_model_page()
    elif page == "SARIMA Model":
        sarima_model_page()
    elif page == "Model Comparison":
        model_comparison_page()


if __name__ == "__main__":
    main()
