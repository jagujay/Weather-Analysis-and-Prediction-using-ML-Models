from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# Function to prepare LSTM data
def prepare_lstm_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


# Function to train LSTM model and generate predictions
def train_lstm_model(data, future_days=7, n_steps=30):
    # Check for missing values in the data
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values. Please clean the data before training the model.")

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare data for LSTM
    X, y = prepare_lstm_data(scaled_data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], data.shape[1]))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")  # Debugging

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train or Test data is empty. Ensure sufficient data for splitting.")

    # Build LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(data.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop])
    # Remove callabcks and change number of epochs if needed

    # Print the train and validation loss
    print(f"Final Train Loss: {history.history['loss'][-1]}")
    if 'val_loss' in history.history:
        print(f"Final Test Loss: {history.history['val_loss'][-1]}")
    else:
        print("Validation loss not computed.")

    # Evaluate the model on test data
    predictions = model.predict(X_test, verbose=0)

    # Inverse transform test and prediction data
    y_test_original = scaler.inverse_transform(y_test)
    predictions_original = scaler.inverse_transform(predictions)

    # Calculate error metrics
    mse = mean_squared_error(y_test_original, predictions_original)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    # Percentage accuracy
    percentage_accuracy = 100 - (np.mean(np.abs(y_test_original - predictions_original) / y_test_original) * 100)

    print(f"Test MSE: {mse}")
    print(f"Test MAE: {mae}")
    print(f"Test R^2: {r2}")

    # Predict future values
    input_seq = scaled_data[-n_steps:]
    forecast = []
    for _ in range(future_days):
        input_reshaped = input_seq.reshape((1, n_steps, data.shape[1]))
        prediction = model.predict(input_reshaped, verbose=0)
        print(f"Prediction Step {_ + 1}: {prediction}")  # Debugging
        forecast.append(prediction[0])
        input_seq = np.vstack((input_seq[1:], prediction))

    # Inverse transform the forecast
    forecast = scaler.inverse_transform(forecast)
    print("Final Forecast:", forecast)  # Debugging
    return history.history['loss'][-1], history.history.get('val_loss', [None])[-1], forecast, round(r2 * 100)
