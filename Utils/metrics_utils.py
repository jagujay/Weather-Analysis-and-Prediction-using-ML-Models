from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def calculate_overall_accuracy(data, y_test_original, predictions_original):
    feature_mape = []
    epsilon = 1e-10

    for feature_idx in range(y_test_original.shape[1]):
        actual = y_test_original[:, feature_idx]
        predicted = predictions_original[:, feature_idx]

        # Exclude features with all zeros or very small values
        if np.all(actual < epsilon):
            continue

        # Avoid division by zero
        actual_non_zero = np.where(actual == 0, epsilon, actual)
        mape = np.mean(np.abs((actual - predicted) / actual_non_zero)) * 100
        feature_mape.append(mape)

    # Calculate overall accuracy (ignoring NaN or invalid MAPE values)
    if feature_mape:
        overall_mape = np.mean(feature_mape)
        overall_accuracy = 100 - overall_mape
    else:
        overall_accuracy = None

    return overall_accuracy


def calculate_metrics(actual_values, forecast):
    try:
        # Debugging: Check lengths and shapes
        print(f"Actual Values Length: {len(actual_values)}, Forecast Length: {len(forecast)}")
        print(f"Actual Values Shape: {actual_values.shape}, Forecast Shape: {forecast.shape}")

        # Ensure both arrays are flattened
        actual_values = np.array(actual_values).flatten()
        forecast = np.array(forecast).flatten()

        print(f"Aligned Shapes - Actual Values: {actual_values.shape}, Forecast: {forecast.shape}")

        # Mean Squared Error (MSE)
        mse = mean_squared_error(actual_values, forecast)
        mae = mean_absolute_error(actual_values, forecast)
        r2 = r2_score(actual_values, forecast)

        # Avoid division by zero for MAPE and SMAPE
        epsilon = 1e-10  # Small value to prevent division by zero
        actual_non_zero = np.where(actual_values == 0, epsilon, actual_values)

        # MAPE
        mape = np.mean(np.abs((actual_values - forecast) / actual_non_zero)) * 100

        # SMAPE
        smape = np.mean(2 * np.abs(actual_values - forecast) /
                        (np.abs(actual_values) + np.abs(forecast) + epsilon)) * 100

        # Accuracy
        accuracy = 100 - mape

        print(f"mse: {mse}")
        print(f"mae: {mae}")
        print(f"r2: {r2}")
        print(f"mape: {mape}")
        print(f"smape: {smape}")
        print(f"accuracy: {accuracy}")

    except Exception as e:
        print("______________________________________________________________________________________________")
        print(e)
        mse, mae, r2, mape, smape, accuracy = [None] * 6

    return {
        "MSE": mse,
        "MAE": mae,
        "R²": r2,
        "MAPE": mape,
        "SMAPE": smape,
        "Accuracy": accuracy
    }


def calculate_metrics_precipitation(actual_values, forecast):
    try:
        # Debugging: Print actual and forecast values
        print("Actual Values (Precipitation):", actual_values)
        print("Forecast Values (Precipitation):", forecast)

        # Ensure both arrays are flattened
        actual_values = np.array(actual_values).flatten()
        forecast = np.array(forecast).flatten()

        # Replace zero values in actual_values with a small epsilon
        epsilon = 1e-10
        actual_non_zero = np.where(actual_values == 0, epsilon, actual_values)

        # Mean Squared Error (MSE)
        mse = mean_squared_error(actual_values, forecast)
        mae = mean_absolute_error(actual_values, forecast)
        r2 = r2_score(actual_values, forecast)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual_values - forecast) / actual_non_zero)) * 100

        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = np.mean(2 * np.abs(actual_values - forecast) /
                        (np.abs(actual_values) + np.abs(forecast) + epsilon)) * 100

        # Accuracy
        accuracy = 100 - mape

    except Exception as e:
        mse, mae, r2, mape, smape, accuracy = [None] * 6
        print(f"Error in calculate_metrics_precipitation: {e}")

    print(f"mse: {mse}")
    print(f"mae: {mae}")
    print(f"r2: {r2}")
    print(f"mape: {mape}")
    print(f"smape: {smape}")
    print(f"accuracy: {accuracy}")

    return {
        "MSE": mse,
        "MAE": mae,
        "R²": r2,
        "MAPE": mape,
        "SMAPE": smape,
        "Accuracy": accuracy
    }