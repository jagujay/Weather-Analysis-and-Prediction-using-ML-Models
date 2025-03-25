from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
from Utils.constants import inverse_rename_mapping
from Utils.metrics_utils import calculate_metrics, calculate_metrics_precipitation


# Function to check stationarity and make series stationary
def make_stationary(series):
    """Ensure series is stationary using differencing."""
    p_value = adfuller(series.dropna())[1]
    if p_value >= 0.05:  # If not stationary, apply differencing
        stationary_series = series.diff().dropna()
        if stationary_series.empty:
            raise ValueError("Differencing led to an empty series.")
        return stationary_series
    return series


# SARIMA Forecast Function (Handles Renamed Features)
def sarima_forecast(data, p, d, q, P, D, Q, m, future_days):
    """Train SARIMA models and forecast future values."""
    forecasts = {}
    summaries = []
    overall_metrics = {}

    for feature in data.columns:
        try:
            # Map display name back to the original column name
            original_name = inverse_rename_mapping.get(feature, feature)
            stationary_series = make_stationary(data[feature])

            # Train SARIMA model
            model = sm.tsa.statespace.SARIMAX(stationary_series,
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, m))
            model_fit = model.fit(disp=False)

            # Generate forecast
            forecast = model_fit.forecast(steps=future_days)

            # Calculate Errors for forecast
            actual_values = data[feature][-future_days:]  # Actual values for the forecasted period

            forecasts[feature] = forecast.rename("Forecast")  # Standardize column name

            print(f"Actual Values Length: {len(actual_values)}")
            print(f"Forecast Length: {len(forecast)}")

            # Calculate error metrics if actual values exist
            if feature == "Total Precipitation":
                metrics = calculate_metrics_precipitation(actual_values, forecast)
            else:
                metrics = calculate_metrics(actual_values, forecast)

            summaries.append({
                "Feature": feature,
                "AR Coefficient": model_fit.params.get("ar.L1", np.nan),
                "MA Coefficient": model_fit.params.get("ma.L1", np.nan),
                "Sigma2": model_fit.params.get("sigma2", np.nan),
                "AIC": model_fit.aic,
                "BIC": model_fit.bic,
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "R²": metrics["R²"],
                "MAPE (%)": metrics["MAPE"],
                "SMAPE (%)": metrics["SMAPE"],
                "Accuracy (%)": metrics["Accuracy"]
            })

            # Aggregate metrics across features
            overall_metrics[feature] = {
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "R²": metrics["R²"],
                "MAPE": metrics["MAPE"],
                "SMAPE": metrics["SMAPE"],
                "Accuracy": metrics["Accuracy"]
            }

        except Exception as e:
            forecasts[feature] = None
            summaries.append({"Feature": feature, "Error": str(e)})
    return forecasts, summaries, overall_metrics
