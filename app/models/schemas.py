from typing import Any

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        description="Symbols to include in training (e.g. BTCUSDT, ETHUSDT)",
    )
    lookback_days: int = Field(
        default=365,
        ge=1,
        le=730,
        description="How many days of historical candles to pull per symbol.",
    )
    forward_horizon: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description=(
            "Legacy single-horizon shortcut. If both this and `horizons` are null, "
            "the trainer uses its default horizon set."
        ),
    )
    horizons: list[int] | None = Field(
        default=None,
        description=(
            "List of forward horizons (in candles) to train. One model is "
            "produced per (horizon, alpha) pair. Default ≈ [16, 60, 240] = "
            "4h / 15h / 60h on 15m candles."
        ),
    )
    quantile_alpha: float | None = Field(
        default=None,
        ge=0.05,
        le=0.95,
        description=(
            "Single-quantile training override. Leave null to train the default "
            "tuple of quantiles (0.3 / 0.5 / 0.7)."
        ),
    )
    alphas: list[float] | None = Field(
        default=None,
        description=(
            "Optional list of quantiles to train, e.g. [0.3, 0.5, 0.7]. "
            "If both this and quantile_alpha are null, defaults to (0.3, 0.5, 0.7)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                 "lookback_days": 365,
                 "horizons": [16, 60, 240],
                 "alphas": [0.3, 0.5, 0.7]}
            ]
        }
    }


class TrainResponse(BaseModel):
    symbols: list[str]
    features: list[str]
    total_samples: int
    train_samples: int
    test_samples: int
    lookback_days: int
    forward_horizon: int = Field(
        ...,
        description="Primary trained horizon (median of `horizons`).",
    )
    horizons: list[int] = Field(
        ...,
        description="All forward horizons that were trained (in candles).",
    )
    alphas: list[float] = Field(..., description="Quantiles trained")
    label: str = Field(..., description="Label form used (e.g. atr_normalized_forward_max)")
    quantile_alpha: float = Field(
        ...,
        description="Primary quantile (median if 0.5 is in alphas, otherwise the middle one).",
    )
    mae: float = Field(..., description="Mean absolute error on the primary alpha (CV mean)")
    rmse: float = Field(..., description="Root mean squared error on the primary alpha (CV mean)")
    r2: float = Field(..., description="R² on the primary alpha (CV mean across folds)")
    hit_rate: float = Field(
        ...,
        description=(
            "Fraction of test rows whose actual ATR-normalized max forward return "
            "was >= predicted on the primary alpha (CV mean)."
        ),
    )
    pinball: float = Field(
        ...,
        description="Pinball loss for the primary alpha (CV mean) – the actual training objective.",
    )
    cv_metrics: dict[str, Any] = Field(
        ...,
        description="Per-alpha walk-forward CV metrics (per-fold + aggregate mean).",
    )


class PredictTPRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair, e.g. BTCUSDT")
    buy_price: float = Field(..., gt=0, description="Entry price (must be > 0)")
    forward_candles: int | None = Field(
        default=None,
        ge=1,
        le=2000,
        description=(
            "Number of future candles (15m bars) over which to predict the max "
            "price reached after buy_price. Snapped to the nearest trained "
            "horizon. If null, uses the model's primary trained horizon."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"symbol": "BTCUSDT", "buy_price": 65000.0, "forward_candles": 60}
            ]
        }
    }


class PredictTPResponse(BaseModel):
    sell_price: float = Field(
        ...,
        description="Predicted sell price (primary quantile) = buy_price × (1 + predicted_return)",
    )
    profitPercentage: float = Field(
        ...,
        description="Predicted profit (primary quantile) as % of buy_price, net of trading fee.",
    )
    accuracy: float | None = Field(
        None,
        description="Historical hit-rate at the primary quantile + chosen horizon (0–1).",
    )
    forward_candles_requested: int | None = Field(
        None,
        description="The forward_candles value the caller asked for.",
    )
    forward_candles_used: int | None = Field(
        None,
        description="The trained horizon actually used (nearest snap from the request).",
    )
    available_horizons: list[int] | None = Field(
        None,
        description="Horizons available in the loaded bundle (in candles).",
    )
    quantiles: dict[str, dict[str, float | None]] | None = Field(
        None,
        description=(
            "Per-quantile take-profit suggestions: keys are 'conservative' (0.3), "
            "'median' (0.5), 'aggressive' (0.7). Each value contains alpha, sell_price, "
            "profitPercentage, and accuracy (the CV hit-rate of that specific quantile)."
        ),
    )


class RegressionStatusResponse(BaseModel):
    trained: bool = Field(..., description="True when a regression model is available")
    quantile_alpha: float | None = Field(None, description="Quantile alpha used at training time")
    hit_rate: float | None = Field(
        None,
        description="Fraction of historical rows where actual max forward return >= predicted (0–1)",
    )
