from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        description="Symbols to include in training (e.g. BTCUSDT, ETHUSDT)",
    )
    min_return_threshold: float = Field(
        default=0.01,
        description="Stored for reference; not used during training",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"], "min_return_threshold": 0.01}
            ]
        }
    }


class TrainResponse(BaseModel):
    symbols: list[str]
    features: list[str]
    total_samples: int
    train_samples: int
    test_samples: int
    mae: float = Field(..., description="Mean absolute error on test split")
    rmse: float = Field(..., description="Root mean squared error on test split")
    r2: float = Field(..., description="R² score on test split")
    directional_accuracy: float = Field(..., description="% of test predictions with correct return direction (0–1)")


class PredictTPRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair, e.g. BTCUSDT")
    buy_price: float = Field(
        default=0.0,
        description="Entry price; pass 0 to use the latest close price",
    )
    min_return_threshold: float = Field(
        default=0.01,
        description="Minimum predicted return required to take the trade",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"symbol": "BTCUSDT", "buy_price": 65000.0, "min_return_threshold": 0.01}
            ]
        }
    }


class PredictTPResponse(BaseModel):
    symbol: str
    current_price: float
    buy_price: float
    predicted_return: float = Field(..., description="Model-predicted return (e.g. 0.023 = 2.3%)")
    take_profit: float = Field(..., description="TP price = buy_price × (1 + predicted_return)")
    signal_taken: bool = Field(..., description="True if predicted_return > min_return_threshold")
    min_return_threshold: float
    directional_accuracy: float | None = Field(None, description="% of test predictions with correct return direction (0–1)")


class RegressionStatusResponse(BaseModel):
    trained: bool = Field(..., description="True when a regression model is available")
    directional_accuracy: float | None = Field(None, description="% of test predictions with correct return direction (0–1)")
