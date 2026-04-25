"""
Multi-symbol XGBoost regression trainer.

Pipeline:
  1. Fetch 1 000 × 5-min candles per symbol from Binance
  2. Compute regression features (past-only)
  3. Detect EMA crossover signals
  4. Create target_return labels from future data (no leakage into features)
  5. Merge all symbols → time-ordered dataset
  6. 80/20 time-based split (no shuffle)
  7. Train XGBRegressor, evaluate, persist to disk
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
from app.services.data_fetcher import fetch_klines
from app.services.regression_features import (
    FEATURE_COLS,
    compute_regression_features,
    detect_ema_crossovers,
)

logger = logging.getLogger(__name__)

_TRAINING_CANDLES = 1000
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

def _create_labels(df: pd.DataFrame) -> pd.Series:
    """
    For every BUY-signal row compute:

        target_return = (max_high_between_buy_and_next_sell - entry_price) / entry_price

    Rows without a subsequent SELL signal are dropped (return NaN).
    Future data is used ONLY for the label, never for features.
    """
    buy_positions = df.index[df["buy_signal"] == 1]
    sell_positions = df.index[df["sell_signal"] == 1]
    int_index = list(df.index)

    target = pd.Series(np.nan, index=df.index)

    for buy_idx in buy_positions:
        buy_pos = int_index.index(buy_idx)
        entry_price = df["close"].iloc[buy_pos]
        if entry_price <= 0:
            continue

        future_sells = [s for s in sell_positions if s > buy_idx]
        if not future_sells:
            continue

        sell_idx = future_sells[0]
        sell_pos = int_index.index(sell_idx)

        # Highest high from candle after entry up to and including sell candle
        max_price = df["high"].iloc[buy_pos + 1 : sell_pos + 1].max()
        if pd.isna(max_price):
            continue

        target.iloc[buy_pos] = (max_price - entry_price) / entry_price

    return target


# ── Dataset construction ───────────────────────────────────────────────────────

def build_dataset(symbols: list[str]) -> tuple[pd.DataFrame, LabelEncoder]:
    """Return (combined_df_with_labels, fitted_encoder) for all symbols."""
    symbols = [s.upper() for s in symbols]
    encoder = LabelEncoder()
    encoder.fit(symbols)

    parts = []
    for symbol in symbols:
        logger.info("Building dataset for %s", symbol)
        try:
            df = fetch_klines(symbol, limit=_TRAINING_CANDLES)
            df = compute_regression_features(df)
            df = detect_ema_crossovers(df)
            df.dropna(subset=["ema9", "ema21", "rsi_14", "macd_hist"], inplace=True)

            df["target_return"] = _create_labels(df)
            labeled = df.dropna(subset=["target_return"])
            if labeled.empty:
                logger.warning("No labeled rows for %s – skipping", symbol)
                continue

            labeled = labeled.copy()
            labeled["symbol_encoded"] = float(encoder.transform([symbol])[0])
            parts.append(labeled)
            logger.info("  %s: %d labeled BUY rows", symbol, len(labeled))

        except Exception as exc:
            logger.error("Failed to build dataset for %s: %s", symbol, exc)

    if not parts:
        raise ValueError("No labeled data available from any of the requested symbols")

    combined = pd.concat(parts).sort_index()
    return combined, encoder


# ── Training ───────────────────────────────────────────────────────────────────

def train(symbols: list[str]) -> dict:
    """
    Full training pipeline.  Returns a metrics dict.

    Thread-safe: concurrent calls will each run training independently and
    overwrite the model file; this is intentional (last write wins).
    """
    combined, encoder = build_dataset(symbols)

    available = [c for c in FEATURE_COLS if c in combined.columns]
    X = combined[available].values.astype(np.float64)
    y = combined["target_return"].values.astype(np.float64)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    logger.info("Training XGBRegressor on %d samples …", len(X_train))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
    directional_accuracy = round(float(np.mean(np.sign(y_pred) == np.sign(y_test))), 4)

    with open(_model_path(), "wb") as fh:
        pickle.dump(model, fh)
    with open(_encoder_path(), "wb") as fh:
        pickle.dump(encoder, fh)
    with open(_metadata_path(), "w") as fh:
        json.dump({"directional_accuracy": directional_accuracy}, fh)
    logger.info("Regression model saved to %s", _model_path())

    with _lock:
        _cache["model"]               = model
        _cache["encoder"]             = encoder
        _cache["features"]            = available
        _cache["directional_accuracy"] = directional_accuracy
        _cache["trained"]             = True

    metrics = {
        "symbols":        [s.upper() for s in symbols],
        "features":       available,
        "total_samples":  int(len(X)),
        "train_samples":  int(len(X_train)),
        "test_samples":   int(len(X_test)),
        "mae":                  round(mae, 6),
        "rmse":                 round(rmse, 6),
        "r2":                   r2,
        "directional_accuracy": directional_accuracy,
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

    directional_accuracy: float | None = None
    if _metadata_path().exists():
        with open(_metadata_path()) as fh:
            directional_accuracy = json.load(fh).get("directional_accuracy")

    with _lock:
        _cache["model"]               = model
        _cache["encoder"]             = encoder
        _cache["features"]            = FEATURE_COLS
        _cache["directional_accuracy"] = directional_accuracy
        _cache["trained"]             = True

    return model, encoder, FEATURE_COLS


def get_accuracy() -> float | None:
    """Return the directional accuracy of the loaded model, or None if unavailable."""
    if not _cache.get("trained"):
        _metadata = _metadata_path()
        if _metadata.exists():
            with open(_metadata) as fh:
                return json.load(fh).get("directional_accuracy")
        return None
    return _cache.get("directional_accuracy")


def is_trained() -> bool:
    if _cache.get("trained"):
        return True
    return _model_path().exists() and _encoder_path().exists()
