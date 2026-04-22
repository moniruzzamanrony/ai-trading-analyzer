"""
Feature engineering pipeline.

Produces exactly the 18 features the XGBoost model was trained on:
  open, high, low, close, volume,
  quote_asset_volume, taker_buy_base_volume, taker_buy_quote_volume,
  ema_short (EMA-9), ema_long (EMA-21),
  rsi (RSI-14),
  MACD_12_26_9, MACDs_12_26_9,
  return (1-bar pct change),
  trend (1 if ema_short > ema_long, else 0),
  volatility (20-bar rolling std of returns),
  buy_pressure (taker_buy_base_volume / volume),
  quote_ratio (taker_buy_quote_volume / quote_asset_volume)

Also computes ADX-14 as a *side-channel* feature for regime classification
(it is NOT fed to the model).
"""

import logging
import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich *df* (raw Binance klines) with all model features and ADX.

    The returned DataFrame contains all original columns plus the derived
    features listed in the module docstring.  NaN rows are NOT dropped here –
    the caller decides when to slice to the clean tail.
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── Trend indicators ──────────────────────────────────────────────────────
    df["ema_short"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    df["ema_long"]  = ta.trend.EMAIndicator(close, window=21).ema_indicator()

    # Binary trend direction: 1 when short EMA is above long EMA (bull), else 0
    df["trend"] = (df["ema_short"] > df["ema_long"]).astype(int)

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd_ind = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD_12_26_9"]  = macd_ind.macd()           # MACD line
    df["MACDs_12_26_9"] = macd_ind.macd_signal()    # signal line

    # ── Returns & volatility ──────────────────────────────────────────────────
    df["return"]     = close.pct_change(1)
    # 20-bar rolling standard deviation of 1-bar returns (used as realised vol)
    df["volatility"] = df["return"].rolling(20).std()

    # ── Order-flow derived features ───────────────────────────────────────────
    # buy_pressure: fraction of volume initiated by buyers (takers buying)
    # Clipped to [0, 1] to guard against occasional data quirks from Binance
    df["buy_pressure"] = (
        df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)
    ).clip(0, 1)

    # quote_ratio: buyer share of total quote turnover
    df["quote_ratio"] = (
        df["taker_buy_quote_volume"] / df["quote_asset_volume"].replace(0, np.nan)
    ).clip(0, 1)

    # ── Regime indicator (side-channel, not fed to model) ─────────────────────
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_ind.adx()

    return df


def detect_regime(df: pd.DataFrame) -> str:
    """
    Classify the current market regime using ADX-14.

    ADX > 25 is the widely-used threshold for a *trending* market.
    Below that, price is oscillating without a clear direction.
    """
    adx = df["adx"].iloc[-1]
    if pd.isna(adx):
        return "UNKNOWN"
    return "TRENDING" if adx > 25 else "SIDEWAYS"


def compute_response_volatility(df: pd.DataFrame) -> float:
    """
    Convert the rolling volatility into the API response value.

    Returns the latest 20-bar realised vol expressed as a percentage (e.g. 0.8
    means ≈0.8% standard deviation per 5-minute bar).
    Rounded to 2 decimal places for clean JSON output.
    """
    vol = df["volatility"].iloc[-1]
    if pd.isna(vol):
        # Fallback: full-series std if the rolling window hasn't warmed up
        vol = df["return"].std()
    return round(float(vol) * 100, 2)
