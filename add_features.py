# add_features.py

import pandas as pd
import ta  # pip install ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trading indicators like SMA, RSI, MACD, Bollinger Bands"""
    # Simple Moving Averages
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    # RSI (Relative Strength Index)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    return df


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling averages of sentiment features"""
    sentiment_cols = [col for col in df.columns if col in ["pos", "neg", "neu", "compound"]]

    for col in sentiment_cols:
        df[f"{col}_3d"] = df[col].rolling(3).mean()
        df[f"{col}_7d"] = df[col].rolling(7).mean()

    return df


def main():
    print("📂 Loading dataset...")
    df = pd.read_csv("final_dataset.csv")

    print("➕ Adding technical indicators...")
    df = add_technical_indicators(df)

    print("➕ Adding rolling sentiment features...")
    df = add_sentiment_features(df)

    print("💾 Saving updated dataset as final_dataset_with_features.csv")
    df.to_csv("final_dataset_with_features.csv", index=False)

    print("✅ Done! New dataset shape:", df.shape)


if __name__ == "__main__":
    main()
