from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        description="Symbols to include in training (e.g. BTCUSDT, ETHUSDT)",
    )
    lookback_days: int = Field(
        default=60,
        ge=1,
        le=730,
        description="How many days of historical candles to pull per symbol.",
    )
    forward_horizon: int = Field(
        default=60,
        ge=1,
        le=500,
        description="Number of future candles to scan for the max-return label.",
    )
    quantile_alpha: float = Field(
        default=0.5,
        ge=0.05,
        le=0.95,
        description=(
            "Quantile target. 0.5 = median forward max (~50% hit rate), "
            "0.3 = conservative (~70% hit rate), 0.7 = aggressive (~30% hit rate)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                 "lookback_days": 60,
                 "forward_horizon": 60,
                 "quantile_alpha": 0.5}
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
    forward_horizon: int
    quantile_alpha: float = Field(..., description="Quantile alpha used for training")
    mae: float = Field(..., description="Mean absolute error on test split")
    rmse: float = Field(..., description="Root mean squared error on test split")
    r2: float = Field(..., description="R² score on test split")
    hit_rate: float = Field(
        ...,
        description=(
            "Fraction of test rows whose actual max forward return was >= "
            "predicted (0–1). Higher = more reliable sell-price predictions."
        ),
    )


class PredictTPRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair, e.g. BTCUSDT")
    buy_price: float = Field(..., gt=0, description="Entry price (must be > 0)")

    model_config = {
        "json_schema_extra": {
            "examples": [{"symbol": "BTCUSDT", "buy_price": 65000.0}]
        }
    }


class PredictTPResponse(BaseModel):
    sell_price: float = Field(
        ...,
        description="Predicted sell price = buy_price × (1 + predicted_return)",
    )
    profitPercentage: float = Field(
        ...,
        description="Predicted profit as a percentage of buy_price (e.g. 2.34 = 2.34%)",
    )
    accuracy: float | None = Field(
        None,
        description="Historical hit-rate: fraction of cycles where actual max "
                    "reached the predicted sell price (0–1). None if model "
                    "metadata is unavailable.",
    )


class RegressionStatusResponse(BaseModel):
    trained: bool = Field(..., description="True when a regression model is available")
    quantile_alpha: float | None = Field(None, description="Quantile alpha used at training time")
    hit_rate: float | None = Field(
        None,
        description="Fraction of historical rows where actual max forward return >= predicted (0–1)",
    )
