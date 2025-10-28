# src/data_loader.py
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import List, Optional

def _read_csv_series(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date')
    # Try columns in order of preference
    for col in ['Adj Close', 'AdjClose', 'Close']:
        if col in df.columns:
            s = df.set_index('Date')[col].dropna()
            s.name = os.path.basename(path).split(".")[0]
            return s
    raise ValueError(f"CSV {path} must contain 'Date' and one of ['Adj Close','AdjClose','Close']")

def _download_yahoo(ticker: str, start: Optional[str], end: Optional[str]) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"Empty data for {ticker}")
    s = df['Adj Close'].dropna()
    s.name = ticker
    s.index = pd.to_datetime(s.index)
    return s

def load_prices(tickers: List[str], start: Optional[str], end: Optional[str], source: str='yahoo', csv_dir: str='./data') -> pd.DataFrame:
    """Load adjusted close prices for given tickers from Yahoo or local CSVs."""
    series = []
    for t in tickers:
        if source == 'csv':
            path = os.path.join(csv_dir, f"{t}.csv")
            s = _read_csv_series(path)
        else:
            s = _download_yahoo(t, start, end)
        series.append(s)
    prices = pd.concat(series, axis=1).sort_index().dropna(how='all')
    prices = prices.dropna(axis=0, how='any')
    return prices

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = np.log(prices / prices.shift(1))
    return r.dropna()
