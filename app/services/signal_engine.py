"""
Signal generation orchestrator.

Ties together data fetching → feature engineering → ML prediction → filtering
and produces the final SignalResponse for one symbol.

This module contains only synchronous logic.  The worker wraps calls in
asyncio.to_thread() so the event loop is never blocked.
"""

import logging

from app.core.config import settings
from app.models.schemas import SignalResponse
from app.services import data_fetcher, feature_engineering, ml_predictor

logger = logging.getLogger(__name__)


def _apply_regime_filter(signal: str, probability: float, regime: str) -> tuple[str, float]:
    """
    Dampen directional signals when the market has no trend (SIDEWAYS).

    In a choppy market the ML model's BUY/SELL calls are less reliable, so we
    scale the probability toward 0.5 and fall back to HOLD when confidence
    drops below the configured minimum.

    Returns the (possibly adjusted) signal and probability.
    """
    if regime == "SIDEWAYS" and signal in ("BUY", "SELL"):
        # Shrink the deviation from 0.5 by 40% to reduce false directional calls
        adjusted_prob = 0.5 + (probability - 0.5) * 0.6
        if adjusted_prob < settings.min_signal_confidence + 0.5:
            return "HOLD", round(adjusted_prob, 4)
        return signal, round(adjusted_prob, 4)

    return signal, probability


def generate_signal(symbol: str) -> SignalResponse:
    """
    Full pipeline for one symbol:

      1. Fetch 5-min klines from Binance
      2. Compute all 18 model features + ADX
      3. Drop NaN rows that result from indicator warm-up periods
      4. Run the XGBoost classifier on the latest candle
      5. Apply regime filter
      6. Return a SignalResponse

    Raises ValueError if insufficient clean rows remain after dropping NaNs.
    """
    logger.info("Generating signal for %s", symbol)

    # ── Step 1: Fetch raw market data ─────────────────────────────────────────
    df = data_fetcher.fetch_klines(symbol)

    # ── Step 2: Compute features ──────────────────────────────────────────────
    df = feature_engineering.compute_features(df)

    # ── Step 3: Drop NaN rows from indicator warm-up ──────────────────────────
    df.dropna(inplace=True)

    if len(df) < 2:
        raise ValueError(f"Not enough clean data for {symbol} after indicator warm-up")

    # ── Step 4: ML prediction ─────────────────────────────────────────────────
    signal, probability = ml_predictor.predict(df, symbol)

    # ── Step 5: Regime filter ─────────────────────────────────────────────────
    regime   = feature_engineering.detect_regime(df)
    signal, probability = _apply_regime_filter(signal, probability, regime)

    # ── Step 6: Build response ────────────────────────────────────────────────
    volatility    = feature_engineering.compute_response_volatility(df)
    current_price = float(df["close"].iloc[-1])

    result = SignalResponse(
        symbol=symbol.upper(),
        signal=signal,
        probability=probability,
        current_price=current_price,
        volatility=volatility,
        regime=regime,
    )

    logger.info(
        "Signal ready | %s | %s | price=%.4f | prob=%.3f | vol=%.2f%% | regime=%s",
        symbol, signal, current_price, probability, volatility, regime,
    )
    return result
