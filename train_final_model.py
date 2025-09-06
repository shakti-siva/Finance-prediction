import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Paths
data_path = "D:/Projects/Finance Prediction/final_dataset.csv"
models_dir = "D:/Projects/Finance Prediction/Models"
os.makedirs(models_dir, exist_ok=True)

# Load dataset
data = pd.read_csv(data_path)
print("Dataset shape:", data.shape)

# Ensure numeric
for col in data.columns:
    if col not in ["Date"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Fill missing values
data.fillna(method='bfill', inplace=True)

# Feature columns (everything except Date + Close as target)
feature_cols = [c for c in data.columns if c not in ["Date", "Close"]]
target_col = "Close"

# Scale all features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[feature_cols].values)

# Scale target separately
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(data[[target_col]].values)

# Save scalers
joblib.dump(scaler, os.path.join(models_dir, "scaler_features.pkl"))
joblib.dump(target_scaler, os.path.join(models_dir, "scaler_target.pkl"))
print("✅ Feature and target scalers saved")

# Sequence builder
def create_sequences(X, y, seq_length=5):
    Xs, ys = [], []
    for i in range(seq_length, len(X)):
        Xs.append(X[i-seq_length:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

seq_length = 5
X, y = create_sequences(X_scaled, y_scaled, seq_length)

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# Predict
y_pred = model.predict(X_test)
y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = target_scaler.inverse_transform(y_pred)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"LSTM RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Save model
model.save(os.path.join(models_dir, "lstm_model.keras"))
print("✅ Training complete. Model + scalers saved.")
