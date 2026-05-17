"""
Live take-profit prediction using the multi-horizon × multi-quantile XGB bundle.

Each model predicts forward-window max in ATR units:
    target = (max_high - close) / atr
At inference we denormalize back to a fractional return:
    pred_return = pred_atr_units × atr_now / close_now

The user specifies how many future candles to forecast (`forward_candles`).
We snap that to the nearest *trained* horizon present in the bundle and
return the requested + used horizon so the caller can see the snap.
"""

import logging

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.data_fetcher import fetch_klines
from app.services.regression_features import compute_regression_features
from app.services.regression_trainer import (
    get_hit_rate,
    get_primary_horizon,
    get_quantile_alpha,
    load_bundle,
)

logger = logging.getLogger(__name__)

_QUANTILE_LABEL = {
    0.3: "conservative",
    0.5: "median",
    0.7: "aggressive",
}


def _label_for(alpha: float) -> str:
    return _QUANTILE_LABEL.get(round(alpha, 2), f"q{alpha:.2f}")


def _nearest_horizon(requested: int, available: list[int]) -> int:
    """Snap `requested` to the closest trained horizon (ties go to the smaller)."""
    return min(available, key=lambda h: (abs(h - requested), h))


def predict_take_profit(
    symbol: str,
    buy_price: float,
    forward_candles: int | None = None,
) -> dict:
    bundle = load_bundle()
    models_by_horizon: dict = bundle["models"]   # {horizon: {alpha: model}}
    feature_cols: list[str] = bundle["feature_cols"]
    symbol_cols: list[str] = bundle["symbol_cols"]
    calibration: dict = bundle.get("calibration") or {}  # {horizon: {alpha: offset}}
    available_horizons: list[int] = sorted(int(h) for h in models_by_horizon.keys())

    if forward_candles is None:
        forward_candles = get_primary_horizon() or available_horizons[len(available_horizons) // 2]
    if forward_candles <= 0:
        raise ValueError("forward_candles must be a positive integer")

    chosen_horizon = _nearest_horizon(int(forward_candles), available_horizons)
    if chosen_horizon != forward_candles:
        # Snap is silent profitkiller bait: caller asks for h=24, gets h=16,
        # entirely different time window. Surface at warn level so ops can
        # see the mismatch even though the wire response shape is unchanged.
        logger.warning(
            "forward_candles=%d snapped to trained horizon=%d (available=%s). "
            "Train at the requested horizon to remove this snap.",
            forward_candles, chosen_horizon, available_horizons,
        )

    df = fetch_klines(symbol.upper())
    df = compute_regression_features(df)
    df.dropna(subset=["ema9", "ema21", "rsi_14", "macd_hist", "atr"], inplace=True)
    if df.empty:
        raise ValueError(f"Insufficient clean data for {symbol} after warm-up.")

    latest = df.iloc[-1]
    atr_now = float(latest["atr"])
    close_now = float(latest["close"])
    if atr_now <= 0 or close_now <= 0:
        raise ValueError(f"Invalid ATR/close for {symbol}: atr={atr_now}, close={close_now}")

    sym_col = f"is_{symbol.upper()}"
    if sym_col not in symbol_cols:
        logger.warning(
            "Symbol %s not seen at training time; using zero one-hot vector",
            symbol,
        )

    row = []
    for col in feature_cols:
        if col in symbol_cols:
            row.append(1.0 if col == sym_col else 0.0)
        else:
            val = latest.get(col, np.nan)
            row.append(0.0 if pd.isna(val) else float(val))
    X = np.array(row, dtype=np.float64).reshape(1, -1)

    atr_to_return = atr_now / close_now
    horizon_models: dict = models_by_horizon[chosen_horizon]
    horizon_offsets: dict = calibration.get(chosen_horizon, {}) or {}
    fee_pct = float(settings.trading_fee_pct)

    quantiles: dict = {}
    for alpha in sorted(horizon_models.keys()):
        model = horizon_models[alpha]
        pred_units = float(model.predict(X)[0])
        # Conformal calibration: shift the raw quantile prediction so the
        # empirical hit-rate matches `1 − α`. Older bundles default to 0.0.
        offset = float(horizon_offsets.get(alpha, 0.0))
        pred_units_cal = pred_units + offset
        pred_return = max(pred_units_cal * atr_to_return, 0.0)
        sell_price = round(buy_price * (1.0 + pred_return), 8)
        profit_pct = round(pred_return * 100.0 - fee_pct, 4)
        quantiles[_label_for(alpha)] = {
            "alpha":            float(alpha),
            "sell_price":       sell_price,
            "profitPercentage": profit_pct,
            "accuracy":         get_hit_rate(alpha, chosen_horizon),
        }

    primary_alpha = get_quantile_alpha() or (
        0.5 if 0.5 in horizon_models else next(iter(horizon_models))
    )
    primary = quantiles.get(_label_for(primary_alpha)) or next(iter(quantiles.values()))

    return {
        "sell_price":                primary["sell_price"],
        "profitPercentage":          primary["profitPercentage"],
        "accuracy":                  get_hit_rate(primary_alpha, chosen_horizon),
        "forward_candles_requested": int(forward_candles),
        "forward_candles_used":      int(chosen_horizon),
        "available_horizons":        available_horizons,
        "quantiles":                 quantiles,
    }
