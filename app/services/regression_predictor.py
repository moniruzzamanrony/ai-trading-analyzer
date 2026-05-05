"""
Live max-sell-price prediction using the trained XGBoost regression model.
"""

import logging

import numpy as np
import pandas as pd

from app.services.data_fetcher import fetch_klines
from app.services.regression_features import (
    FEATURE_COLS,
    compute_regression_features,
)
from app.services.regression_trainer import get_hit_rate, load_model

logger = logging.getLogger(__name__)

# Round-trip trading fee (entry + exit) as a fraction of notional.
TRADING_FEE_PCT = 0.2  # = 0.2 %


def predict_take_profit(symbol: str, buy_price: float) -> dict:
    """
    Predict the take-profit sell price for a BUY entry at *buy_price*.

    Returns:
      - sell_price:        buy_price × (1 + predicted_return)
      - profitPercentage:  (predicted_return × 100) − TRADING_FEE_PCT
                           (net profit after trading fee, can be negative)
      - accuracy:          historical hit-rate of the model
    """
    model, encoder, feature_cols = load_model()

    df = fetch_klines(symbol.upper())
    df = compute_regression_features(df)
    df.dropna(subset=["ema9", "ema21", "rsi_14", "macd_hist"], inplace=True)

    if df.empty:
        raise ValueError(f"Insufficient clean data for {symbol} after indicator warm-up.")

    latest = df.iloc[-1]

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
    pred_return = max(float(model.predict(X)[0]), 0.0)

    sell_price = round(buy_price * (1.0 + pred_return), 8)
    profit_pct = round(pred_return * 100.0 - TRADING_FEE_PCT, 4)

    return {
        "sell_price":       sell_price,
        "profitPercentage": profit_pct,
        "accuracy":         get_hit_rate(),
    }
