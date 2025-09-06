import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset
data = pd.read_csv("final_dataset.csv")
data.fillna(method='bfill', inplace=True)  # fill missing values

# Select the 9 features your LSTM uses
features = ['High', 'Low', 'Open', 'Volume', 'pos', 'neu', 'neg', 'compound', 'Target_Up']

# Fit scaler
scaler = StandardScaler()
scaler.fit(data[features])

# Save it to Models folder
joblib.dump(scaler, "Models/scaler_9_features.pkl")
print("✅ Scaler saved as 'Models/scaler_9_features.pkl'")
