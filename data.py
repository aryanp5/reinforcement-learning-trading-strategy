"""Download daily prices and build the features used by the environment and baselines."""
import pandas as pd
import yfinance as yf


def download_prices(ticker, start_date, end_date):
    """Daily closing prices for one ticker, adjusted for splits and dividends."""
    data = yf.download(ticker, start=start_date, end=end_date,
                       auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker} between {start_date} and {end_date}.")

    close = data["Close"]
    # Recent yfinance versions return one column per ticker, even for a single ticker
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.to_frame(name="Close")


def add_features(prices, trend_window=20):
    """Add the columns the rest of the project needs.

    Trend       : Close / 20-day average - 1. Scale-free, so +3% means the same
                  whether the stock trades at $20 or $200.
    SMA50/200   : only used by the moving-average crossover baseline.
    Next_Return : tomorrow's return, i.e. what today's position earns. It is used for
                  the reward only and is never part of the state, so there's no look-ahead.
    """
    df = prices.copy()
    df["Trend"] = df["Close"] / df["Close"].rolling(trend_window).mean() - 1
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["Next_Return"] = df["Close"].pct_change().shift(-1)

    # Drops the 200-day warm-up at the start and the last day (no next return yet)
    return df.dropna()


if __name__ == "__main__":
    df = add_features(download_prices("AAPL", "2020-01-01", "2021-01-01"))
    print(df.head())
    print(df.describe().round(4))