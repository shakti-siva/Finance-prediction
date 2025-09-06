import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
import os

# Config
DATA_PATH = "final_dataset.csv"
MODEL_PATH = "Models/lstm_model.keras"
SCALER_PATH = "Models/scaler_9_features.pkl"
FUTURE_DAYS = 5  # number of days to predict

# Load dataset
data = pd.read_csv(DATA_PATH)
data.fillna(method='bfill', inplace=True)

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Features used for training
features = ['High', 'Low', 'Open', 'Volume', 'pos', 'neu', 'neg', 'compound', 'Target_Up']

# Take the last sequence (matching LSTM input length)
sequence_length = 5  # adjust if your LSTM uses a different window
last_sequence = data[features].values[-sequence_length:]

# Scale features
last_sequence_scaled = scaler.transform(last_sequence)

# Reshape for LSTM: (1, sequence_length, n_features)
last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, len(features))

# Predict future days
future_preds_scaled = []
current_sequence = last_sequence_scaled.copy()

for _ in range(FUTURE_DAYS):
    pred_scaled = model.predict(current_sequence, verbose=0)
    future_preds_scaled.append(pred_scaled[0, -1])  # last feature is Target/Close

    # Update sequence: drop first row, append predicted values
    pred_row = np.zeros((1, len(features)))
    pred_row[0, -1] = pred_scaled[0, -1]  # only the target is predicted, others can stay 0
    current_sequence = np.concatenate([current_sequence[:, 1:, :], pred_row.reshape(1,1,len(features))], axis=1)

# Convert predictions back to real prices
future_preds_scaled = np.array(future_preds_scaled).reshape(-1,1)
# Need to inverse-transform: fill other features with zeros
inverse_input = np.zeros((FUTURE_DAYS, len(features)))
inverse_input[:, -1] = future_preds_scaled[:,0]
future_preds_real = scaler.inverse_transform(inverse_input)[:, -1]

# Build future dates
last_date = pd.to_datetime(data['Date'].iloc[-1])
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)

# Save to CSV
df_future = pd.DataFrame({"Predicted_Close": future_preds_real}, index=future_dates)
df_future.to_csv("future_predictions.csv")
print("✅ Future predictions saved to 'future_predictions.csv'")
print(df_future)
