# stock_sentiment.py
import pandas as pd
import numpy as np

def generate_fake_sentiment(dates):
    np.random.seed(42)  # reproducible
    data = {
        "pos": np.random.uniform(0, 0.6, len(dates)),
        "neu": np.random.uniform(0.2, 0.8, len(dates)),
        "neg": np.random.uniform(0, 0.6, len(dates)),
    }
    df = pd.DataFrame(data, index=dates)
    df["compound"] = df["pos"] - df["neg"]
    return df

if __name__ == "__main__":
    # Create sentiment for business (trading) days
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    sentiment = generate_fake_sentiment(dates)

    print("Sample Sentiment Data:")
    print(sentiment.head())

    # Save for use in build_dataset.py
    sentiment.to_csv("sentiment.csv")

