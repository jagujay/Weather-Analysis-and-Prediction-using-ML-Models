from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np
from Utils.metrics_utils import calculate_metrics, calculate_metrics_precipitation


# Function to perform stationarity check
def check_stationarity(series):
    result = adfuller(series.dropna())
    return {"ADF Statistic": result[0], "p-value": result[1]}


# Function to train ARIMA model and forecast values
def arima_forecast(data, p, d, q, future_days):
    forecasts = {}
    summaries = []
    overall_metrics = {}

    for feature, column_name in data.items():
        try:
            # Train ARIMA model
            model = sm.tsa.ARIMA(column_name.dropna(), order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=future_days)
            forecasts[feature] = forecast

            # Calculate Errors for forecast
            actual_values = column_name.dropna()[-future_days:]  # Last 'future_days' as actual

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
