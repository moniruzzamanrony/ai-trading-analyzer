"""
Live take-profit prediction using the trained XGBoost regression model.

Workflow:
  1. Fetch latest 5-min candles for the symbol
  2. Compute regression features on the most recent candle (past-only)
  3. Encode the symbol
  4. Predict expected return
  5. Convert to TP price: take_profit = buy_price × (1 + predicted_return)
  6. Apply trade filter: only take trade if predicted_return > min_return
"""

import logging

import numpy as np
import pandas as pd

from app.services.data_fetcher import fetch_klines
from app.services.regression_features import (
    FEATURE_COLS,
    compute_regression_features,
    detect_ema_crossovers,
)
from app.services.regression_trainer import get_accuracy, load_model

logger = logging.getLogger(__name__)


def predict_take_profit(
    symbol: str,
    buy_price: float,
    min_return: float = 0.01,
) -> dict:
    """
    Predict the expected return and derived take-profit for a BUY entry.

    Args:
        symbol:      Trading pair, e.g. "BTCUSDT".
        buy_price:   Entry price.  Pass 0 to use the latest close.
        min_return:  Minimum predicted return to flag *signal_taken* True.

    Returns a dict matching PredictTPResponse schema.
    """
    model, encoder, feature_cols = load_model()

    df = fetch_klines(symbol.upper())
    df = compute_regression_features(df)
    df = detect_ema_crossovers(df)
    df.dropna(subset=["ema9", "ema21", "rsi_7", "macd_hist"], inplace=True)

    if df.empty:
        raise ValueError(f"Insufficient clean data for {symbol} after indicator warm-up.")

    latest = df.iloc[-1]
    current_price = float(latest["close"])
    effective_buy = buy_price if buy_price > 0 else current_price

    # Encode symbol; fall back to mean class index for unseen symbols
    try:
        sym_encoded = float(encoder.transform([symbol.upper()])[0])
    except ValueError:
        sym_encoded = float((len(encoder.classes_) - 1) / 2)
        logger.warning(
            "Symbol %s not seen during training; using encoded value %.1f",
            symbol, sym_encoded,
        )

    row = []
    for col in feature_cols:
        if col == "symbol_encoded":
            row.append(sym_encoded)
        else:
            val = latest.get(col, np.nan)
            row.append(0.0 if pd.isna(val) else float(val))

    X = np.array(row, dtype=np.float64).reshape(1, -1)
    pred_return = float(model.predict(X)[0])
    pred_return = max(pred_return, 0.0)

    take_profit = round(effective_buy * (1.0 + pred_return), 8)

    return {
        "symbol":               symbol.upper(),
        "current_price":        current_price,
        "buy_price":            effective_buy,
        "predicted_return":     round(pred_return, 6),
        "take_profit":          take_profit,
        "signal_taken":         pred_return > min_return,
        "min_return_threshold": min_return,
        "directional_accuracy": get_accuracy(),
    }
