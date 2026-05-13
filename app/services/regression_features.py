"""
Feature engineering for the XGBoost return-regression model.

Base timeframe: 15-minute candles. Higher-timeframe context (1h, 4h) is
resampled from the same 15m stream so we never look ahead.

All features use ONLY past/current candle data — no future leakage.
"""

import numpy as np
import pandas as pd
import ta

# Base 15m features (same schema for trainer + predictor; symbol one-hots
# are appended dynamically via `symbol_one_hot_cols`).
BASE_FEATURE_COLS = [
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
    "ret_6",
    "ret_12",
    "ret_48",
    "vol_of_ret_20",
    "vol_of_ret_50",
    "atr_pct_rank_200",
    "dist_from_high_50",
    "dist_from_low_50",
    "range_norm",
    "body_norm",
    # Microstructure
    "taker_buy_ratio",
    "taker_buy_ratio_ma20",
    "log_num_trades",
    # Cyclic time-of-day / day-of-week
    "hod_sin",
    "hod_cos",
    "dow_sin",
    "dow_cos",
    # Higher-timeframe context (resampled from 15m → 1h, 4h)
    "ema_diff_1h",
    "rsi_14_1h",
    "atr_ratio_1h",
    "ret_1h",
    "ema_diff_4h",
    "rsi_14_4h",
    "atr_ratio_4h",
    "ret_4h",
]

def symbol_one_hot_cols(symbols: list[str]) -> list[str]:
    """Stable, sorted one-hot column names for a set of symbols."""
    return [f"is_{s.upper()}" for s in sorted({s.upper() for s in symbols})]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _add_htf_features(df: pd.DataFrame, rule: str, suffix: str) -> pd.DataFrame:
    """
    Resample 15m OHLCV → higher-timeframe (rule e.g. '1h', '4h'),
    compute EMA9/EMA21 diff, RSI(14), ATR(14)/close, and 1-bar return,
    then forward-fill back onto the 15m index. Past-only by construction
    because we only forward-fill (never bfill).
    """
    htf = df[["open", "high", "low", "close"]].resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if htf.empty:
        for col in (f"ema_diff_{suffix}", f"rsi_14_{suffix}",
                    f"atr_ratio_{suffix}", f"ret_{suffix}"):
            df[col] = np.nan
        return df

    ema9 = ta.trend.EMAIndicator(htf["close"], window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(htf["close"], window=21).ema_indicator()
    rsi = ta.momentum.RSIIndicator(htf["close"], window=14).rsi()
    atr = ta.volatility.AverageTrueRange(
        htf["high"], htf["low"], htf["close"], window=14
    ).average_true_range()

    htf_feats = pd.DataFrame(index=htf.index)
    htf_feats[f"ema_diff_{suffix}"] = ema9 - ema21
    htf_feats[f"rsi_14_{suffix}"] = rsi
    htf_feats[f"atr_ratio_{suffix}"] = _safe_div(atr, htf["close"])
    htf_feats[f"ret_{suffix}"] = htf["close"].pct_change(1)

    # Forward-fill onto the 15m index — each 15m bar inherits the most
    # recently CLOSED higher-timeframe bar's values.
    aligned = htf_feats.reindex(df.index, method="ffill")
    for col in aligned.columns:
        df[col] = aligned[col]
    return df


def compute_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all regression model features to *df* in-place and return it."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Trend / momentum (15m base) ─────────────────────────────────────────
    ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df["ema9"] = ema9
    df["ema21"] = ema21
    df["ema_diff"] = _safe_div(ema9 - ema21, close)
    df["ema9_slope"] = ema9.diff(3) / close

    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["macd_hist"] = macd.macd_diff() / close

    # ── Volatility ──────────────────────────────────────────────────────────
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr"] = atr
    df["atr_ratio"] = _safe_div(atr, close)
    df["atr_pct_rank_200"] = atr.rolling(200, min_periods=50).rank(pct=True)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_range = (bb_upper - bb_lower)
    df["bb_width"] = _safe_div(bb_range, close)
    df["bb_position"] = _safe_div(close - bb_lower, bb_range)

    # ── Volume ──────────────────────────────────────────────────────────────
    vol_ma = volume.rolling(20).mean()
    df["volume_ratio"] = _safe_div(volume, vol_ma)

    # ── Returns ─────────────────────────────────────────────────────────────
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["ret_12"] = close.pct_change(12)
    df["ret_48"] = close.pct_change(48)

    df["vol_of_ret_20"] = df["ret_1"].rolling(20).std()
    df["vol_of_ret_50"] = df["ret_1"].rolling(50).std()

    # ── Distance to recent extremes (50-bar = ~12.5h on 15m) ────────────────
    rolling_high = high.rolling(50, min_periods=10).max()
    rolling_low = low.rolling(50, min_periods=10).min()
    df["dist_from_high_50"] = _safe_div(close - rolling_high, close)
    df["dist_from_low_50"] = _safe_div(close - rolling_low, close)

    df["range_norm"] = _safe_div(high - low, close)
    df["body_norm"] = _safe_div(close - df["open"], close)

    # ── Microstructure (Binance-native) ─────────────────────────────────────
    taker_buy = df.get("taker_buy_base_volume")
    if taker_buy is not None:
        df["taker_buy_ratio"] = _safe_div(taker_buy, volume)
        df["taker_buy_ratio_ma20"] = df["taker_buy_ratio"].rolling(20).mean()
    else:
        df["taker_buy_ratio"] = np.nan
        df["taker_buy_ratio_ma20"] = np.nan

    num_trades = df.get("num_trades")
    if num_trades is not None:
        df["log_num_trades"] = np.log1p(num_trades.astype(float))
    else:
        df["log_num_trades"] = np.nan

    # ── Cyclic time features ────────────────────────────────────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        hod = df.index.hour + df.index.minute / 60.0
        dow = df.index.dayofweek.astype(float)
        df["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
        df["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    else:
        for col in ("hod_sin", "hod_cos", "dow_sin", "dow_cos"):
            df[col] = 0.0

    # ── Higher-timeframe context resampled from 15m ─────────────────────────
    df = _add_htf_features(df, "1h", "1h")
    df = _add_htf_features(df, "4h", "4h")

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
