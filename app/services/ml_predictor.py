"""
XGBoost model wrapper with lazy loading and thread-safe singleton.

The model and feature list are loaded once on first use (not at import time)
so that the API server can start even if it takes a moment, and so that tests
can swap them without side-effects.

Model output classes:
  0 → SELL
  1 → HOLD
  2 → BUY
"""

import logging
import pickle
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# Human-readable label for each class index
CLASS_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}

_load_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the XGBoost model (thread-safe via lru_cache + lock)."""
    path: Path = settings.model_path
    logger.info("Loading model from %s", path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    """Load and cache the ordered feature name list."""
    path: Path = settings.features_path
    logger.info("Loading feature list from %s", path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def predict(df: pd.DataFrame) -> tuple[str, float]:
    """
    Run the XGBoost classifier on the *last row* of *df*.

    Returns:
        signal      – "BUY", "SELL", or "HOLD"
        probability – confidence of the predicted class (0–1)

    Any NaN feature values are replaced with 0.0 so the model never crashes.
    A warning is logged when this happens.
    """
    model        = _load_model()
    feature_names = _load_feature_names()

    latest = df.iloc[-1]

    # Build the feature vector in the exact order the model expects
    feature_values = []
    for name in feature_names:
        val = latest.get(name, np.nan)
        if pd.isna(val):
            logger.warning("Feature '%s' is NaN – substituting 0.0", name)
            val = 0.0
        feature_values.append(float(val))

    X = np.array(feature_values, dtype=np.float64).reshape(1, -1)

    probs       = model.predict_proba(X)[0]          # shape: (3,) → [P(sell), P(hold), P(buy)]
    class_idx   = int(np.argmax(probs))
    probability = float(probs[class_idx])
    signal      = CLASS_LABELS[class_idx]

    return signal, round(probability, 4)
