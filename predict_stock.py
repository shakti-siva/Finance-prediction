import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Paths
data_path = "D:/Projects/Finance Prediction/final_dataset.csv"
models_dir = "D:/Projects/Finance Prediction/Models"
model_path = os.path.join(models_dir, "lstm_model.keras")
scaler_features_path = os.path.join(models_dir, "scaler_features.pkl")
scaler_target_path = os.path.join(models_dir, "scaler_target.pkl")

print(f"Loading dataset from: {data_path}")
data = pd.read_csv(data_path)
print("Dataset loaded, shape:", data.shape)

# Ensure numeric
for col in data.columns:
    if col != "Date":
        data[col] = pd.to_numeric(data[col], errors="coerce")

data.fillna(method='bfill', inplace=True)

# Feature/target split
feature_cols = [c for c in data.columns if c not in ["Date", "Close"]]
target_col = "Close"

# Load scalers
scaler_features = joblib.load(scaler_features_path)
scaler_target = joblib.load(scaler_target_path)

# Scale features
X_scaled = scaler_features.transform(data[feature_cols].values)
y_scaled = scaler_target.transform(data[[target_col]].values)

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

# Load model
model = load_model(model_path)
print(f"✅ Model loaded from {model_path}")

# Predict on test set
y_pred = model.predict(X_test)

# Rescale
y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler_target.inverse_transform(y_pred)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Save predictions
results = pd.DataFrame({
    "Date": data["Date"].iloc[-len(y_test):].values,
    "Actual": y_test_rescaled.flatten(),
    "Predicted": y_pred_rescaled.flatten()
})
results.to_csv("predictions.csv", index=False)
print("📂 Predictions saved to 'predictions.csv'")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(results["Date"], results["Actual"], label="Actual")
plt.plot(results["Date"], results["Predicted"], label="Predicted")
plt.xticks(rotation=45)
plt.title("Predicted vs Actual Close Prices")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.close()
print("📊 Plot saved to 'predicted_vs_actual.png'")

# --- Next-day forecasting ---
last_sequence = X_scaled[-seq_length:]  # last 5 days of features
last_sequence = np.expand_dims(last_sequence, axis=0)  # reshape (1, seq_len, features)

next_day_scaled = model.predict(last_sequence)
next_day_price = scaler_target.inverse_transform(next_day_scaled)[0][0]

print(f"🔮 Next day predicted Close price: {next_day_price:.2f}")
