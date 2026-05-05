"""
Multi-symbol XGBoost regression trainer.

Pipeline:
  1. Fetch *lookback_days* of historical candles per symbol from Binance
  2. Compute regression features (past-only)
  3. Label every candle with target_return = max forward return over the
     next *forward_horizon* candles (no future leakage into features)
  4. Merge all symbols → time-ordered dataset
  5. 80/20 time-based split (no shuffle)
  6. Train quantile XGBRegressor, evaluate, persist to disk
"""

import json
import logging
import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from app.core.config import settings
from app.services.data_fetcher import fetch_klines_history
from app.services.regression_features import (
    FEATURE_COLS,
    compute_regression_features,
)

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS = 60
_DEFAULT_FORWARD_HORIZON = 60   # candles
_MODEL_FILENAME = "regression_xgb_model.pkl"
_ENCODER_FILENAME = "regression_label_encoder.pkl"
_METADATA_FILENAME = "regression_metadata.json"

_lock = threading.Lock()
_cache: dict = {}


def _model_path() -> Path:
    return settings.models_dir / _MODEL_FILENAME


def _encoder_path() -> Path:
    return settings.models_dir / _ENCODER_FILENAME


def _metadata_path() -> Path:
    return settings.models_dir / _METADATA_FILENAME


# ── Label creation ─────────────────────────────────────────────────────────────

def _create_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    target_return[i] = (max(high[i+1 : i+1+horizon]) - close[i]) / close[i]

    Every candle gets a label, so the live predictor's call on the latest
    candle is in-distribution. Last *horizon* rows have NaN labels.
    """
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    n = len(df)
    target = np.full(n, np.nan)

    for i in range(n - horizon):
        c = close[i]
        if c <= 0:
            continue
        future_max = high[i + 1 : i + 1 + horizon].max()
        target[i] = (future_max - c) / c

    return pd.Series(target, index=df.index)


# ── Dataset construction ───────────────────────────────────────────────────────

def build_dataset(
    symbols: list[str],
    lookback_days: int,
    forward_horizon: int,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Return (combined_df_with_labels, fitted_encoder) for all symbols."""
    symbols = [s.upper() for s in symbols]
    encoder = LabelEncoder()
    encoder.fit(symbols)

    parts = []
    for symbol in symbols:
        logger.info("Building dataset for %s (lookback=%dd)", symbol, lookback_days)
        try:
            df = fetch_klines_history(symbol, days=lookback_days)
            if df.empty:
                logger.warning("No candles returned for %s – skipping", symbol)
                continue

            df = compute_regression_features(df)
            df.dropna(subset=["ema9", "ema21", "rsi_14", "macd_hist"], inplace=True)

            df["target_return"] = _create_labels(df, forward_horizon)
            labeled = df.dropna(subset=["target_return"]).copy()
            if labeled.empty:
                logger.warning("No labeled rows for %s – skipping", symbol)
                continue

            labeled["symbol_encoded"] = float(encoder.transform([symbol])[0])
            parts.append(labeled)
            logger.info("  %s: %d labeled rows", symbol, len(labeled))

        except Exception as exc:
            logger.error("Failed to build dataset for %s: %s", symbol, exc)

    if not parts:
        raise ValueError("No labeled data available from any of the requested symbols")

    combined = pd.concat(parts).sort_index()
    return combined, encoder


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    symbols: list[str],
    quantile_alpha: float = 0.5,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    forward_horizon: int = _DEFAULT_FORWARD_HORIZON,
) -> dict:
    """
    Full training pipeline using quantile regression on the forward-window
    max return.

    quantile_alpha:
      - 0.5 (default) → predict the median forward max → ~50% of trades
        reach the predicted price.
      - 0.3 → ~70% reach it (more conservative price, easier to hit).
      - 0.7 → ~30% reach it (aggressive, frequently missed).
    """
    if not 0.0 < quantile_alpha < 1.0:
        raise ValueError("quantile_alpha must be in (0, 1)")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    if forward_horizon <= 0:
        raise ValueError("forward_horizon must be > 0")

    combined, encoder = build_dataset(symbols, lookback_days, forward_horizon)

    available = [c for c in FEATURE_COLS if c in combined.columns]
    X = combined[available].values.astype(np.float64)
    y = combined["target_return"].values.astype(np.float64)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile_alpha,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    logger.info(
        "Training quantile XGBRegressor (alpha=%.2f) on %d samples (horizon=%d) …",
        quantile_alpha, len(X_train), forward_horizon,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
    hit_rate = round(float(np.mean(y_test >= y_pred)), 4) if len(y_test) else 0.0

    with open(_model_path(), "wb") as fh:
        pickle.dump(model, fh)
    with open(_encoder_path(), "wb") as fh:
        pickle.dump(encoder, fh)
    with open(_metadata_path(), "w") as fh:
        json.dump({
            "hit_rate": hit_rate,
            "quantile_alpha": quantile_alpha,
            "lookback_days": lookback_days,
            "forward_horizon": forward_horizon,
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": r2,
        }, fh)
    logger.info("Regression model saved to %s", _model_path())

    with _lock:
        _cache["model"]           = model
        _cache["encoder"]         = encoder
        _cache["features"]        = available
        _cache["hit_rate"]        = hit_rate
        _cache["quantile_alpha"]  = quantile_alpha
        _cache["lookback_days"]   = lookback_days
        _cache["forward_horizon"] = forward_horizon
        _cache["trained"]         = True

    metrics = {
        "symbols":         [s.upper() for s in symbols],
        "features":        available,
        "total_samples":   int(len(X)),
        "train_samples":   int(len(X_train)),
        "test_samples":    int(len(X_test)),
        "lookback_days":   lookback_days,
        "forward_horizon": forward_horizon,
        "quantile_alpha":  quantile_alpha,
        "mae":             round(mae, 6),
        "rmse":            round(rmse, 6),
        "r2":              r2,
        "hit_rate":        hit_rate,
    }
    logger.info("Training complete: %s", metrics)
    return metrics


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model() -> tuple[object, LabelEncoder, list[str]]:
    """Return (model, encoder, feature_cols). Loads from disk on first call."""
    with _lock:
        if _cache.get("trained"):
            return _cache["model"], _cache["encoder"], _cache["features"]

    mp = _model_path()
    ep = _encoder_path()

    if not mp.exists() or not ep.exists():
        raise FileNotFoundError("Regression model has not been trained yet.")

    with open(mp, "rb") as fh:
        model = pickle.load(fh)
    with open(ep, "rb") as fh:
        encoder = pickle.load(fh)

    metadata: dict = {}
    if _metadata_path().exists():
        with open(_metadata_path()) as fh:
            metadata = json.load(fh)

    with _lock:
        _cache["model"]           = model
        _cache["encoder"]         = encoder
        _cache["features"]        = FEATURE_COLS
        _cache["hit_rate"]        = metadata.get("hit_rate")
        _cache["quantile_alpha"]  = metadata.get("quantile_alpha")
        _cache["lookback_days"]   = metadata.get("lookback_days")
        _cache["forward_horizon"] = metadata.get("forward_horizon")
        _cache["trained"]         = True

    return model, encoder, FEATURE_COLS


def _metadata_field(key: str):
    if _cache.get("trained"):
        return _cache.get(key)
    if _metadata_path().exists():
        with open(_metadata_path()) as fh:
            return json.load(fh).get(key)
    return None


def get_hit_rate() -> float | None:
    """Fraction of test rows where actual max reached the prediction."""
    return _metadata_field("hit_rate")


def get_quantile_alpha() -> float | None:
    return _metadata_field("quantile_alpha")


def is_trained() -> bool:
    if _cache.get("trained"):
        return True
    return _model_path().exists() and _encoder_path().exists()
