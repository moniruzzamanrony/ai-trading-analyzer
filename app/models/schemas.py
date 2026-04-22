"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field


class SignalResponse(BaseModel):
    """Trading signal for a single symbol."""

    symbol: str = Field(..., description="Trading pair, e.g. BTCUSDT")
    signal: str = Field(..., description="BUY | SELL | HOLD")
    probability: float = Field(..., description="Model confidence for the predicted class (0–1)")
    current_price: float = Field(..., description="Latest close price")
    volatility: float = Field(..., description="Realised volatility over the last 20 candles (%)")
    regime: str = Field(..., description="Market regime: TRENDING | SIDEWAYS")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symbol": "BTCUSDT",
                    "signal": "HOLD",
                    "probability": 0.52,
                    "volatility": 0.8,
                    "regime": "SIDEWAYS",
                }
            ]
        }
    }
