"""
XGBoost model wrapper with per-symbol lazy loading.

Each symbol has its own model and feature list loaded on first use and cached
for the lifetime of the process.

Model output classes:
  0 → SELL
  1 → HOLD
  2 → BUY

Expected file naming convention (case-insensitive symbol):
  <models_dir>/<symbol_lower>_xgb_model.pkl
  <models_dir>/<symbol_lower>_features.pkl
"""

import logging
import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

CLASS_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Per-symbol cache: symbol (upper) → (model, feature_names)
_cache: dict[str, tuple] = {}
_lock = threading.Lock()


def _load_for_symbol(symbol: str) -> tuple:
    """Return (model, feature_names) for *symbol*, loading from disk if needed."""
    key = symbol.upper()
    if key in _cache:
        return _cache[key]

    with _lock:
        if key in _cache:
            return _cache[key]

        base = symbol.lower()
        models_dir: Path = settings.models_dir

        model_path    = models_dir / f"{base}_xgb_model.pkl"
        features_path = models_dir / f"{base}_features.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"No model file for {symbol}: {model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"No features file for {symbol}: {features_path}")

        logger.info("Loading model for %s from %s", symbol, model_path)
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)

        logger.info("Loading features for %s from %s", symbol, features_path)
        with open(features_path, "rb") as fh:
            feature_names = pickle.load(fh)

        _cache[key] = (model, feature_names)
        return _cache[key]


def predict(df: pd.DataFrame, symbol: str) -> tuple[str, float]:
    """
    Run the XGBoost classifier for *symbol* on the last row of *df*.

    Returns:
        signal      – "BUY", "SELL", or "HOLD"
        probability – confidence of the predicted class (0–1)
    """
    model, feature_names = _load_for_symbol(symbol)

    latest = df.iloc[-1]

    feature_values = []
    for name in feature_names:
        val = latest.get(name, np.nan)
        if pd.isna(val):
            logger.warning("[%s] Feature '%s' is NaN – substituting 0.0", symbol, name)
            val = 0.0
        feature_values.append(float(val))

    X = np.array(feature_values, dtype=np.float64).reshape(1, -1)

    probs       = model.predict_proba(X)[0]
    class_idx   = int(np.argmax(probs))
    probability = float(probs[class_idx])
    signal      = CLASS_LABELS[class_idx]

    return signal, round(probability, 4)
