"""
Feature engineering for the XGBoost return-regression model.

All features use ONLY past/current candle data (no future leakage).
"""

import numpy as np
import pandas as pd
import ta

FEATURE_COLS = [
    "ema_diff",
    "ema9_slope",
    "rsi_14",
    "macd_hist",
    "atr_ratio",
    "bb_width",
    "bb_position",
    "volume_ratio",
    "ret_1",
    "ret_3",
    "range_norm",
    "body_norm",
    "symbol_encoded",
]


def compute_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all regression model features to *df* in-place and return it."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema9"] = ema9
    df["ema21"] = ema21
    df["ema_diff"] = ema9 - ema21
    df["ema9_slope"] = ema9.diff(3)

    # Momentum
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd_hist"] = macd.macd_diff()

    # Volatility
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_ratio"] = atr / close.replace(0, np.nan)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_width"] = bb_range / close.replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / bb_range

    # Volume
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = volume / vol_ma

    # Price action
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["range_norm"] = (high - low) / close.replace(0, np.nan)
    df["body_norm"] = (close - df["open"]) / close.replace(0, np.nan)

    return df


def detect_macd_hist_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add buy_signal and sell_signal columns based on MACD(12,26,9) histogram
    sign changes.

    buy_signal  = 1 on the first bar where the histogram turns positive
                  (green-start: prev <= 0 and curr > 0)
    sell_signal = 1 on the first bar where the histogram turns non-positive
                  (red-start:   prev > 0 and curr <= 0)
    """
    hist = df["macd_hist"]
    prev_pos = hist.shift(1) > 0
    curr_pos = hist > 0

    df["buy_signal"] = (~prev_pos & curr_pos).astype(int)
    df["sell_signal"] = (prev_pos & ~curr_pos).astype(int)

    return df
