"""
XGBoost return-regression endpoints.

POST /regression/train   – Train a new multi-symbol regression model
POST /regression/predict – Predict take-profit for a BUY entry
GET  /regression/status  – Check whether the model is available
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    PredictTPRequest,
    PredictTPResponse,
    RegressionStatusResponse,
    TrainRequest,
    TrainResponse,
)
from app.services import regression_predictor, regression_trainer
from app.services.regression_trainer import get_hit_rate, get_quantile_alpha

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regression", tags=["regression"])


@router.post(
    "/v2/train",
    response_model=TrainResponse,
    summary="Train multi-symbol regression model",
    description=(
        "Fetches candles per symbol, engineers features (past-only), creates "
        "target_return labels = max_pct_up during each MACD-bullish cycle, "
        "and trains a quantile XGBRegressor with an 80/20 time-based split. "
        "Lower quantile_alpha → more conservative predictions that get hit "
        "more often. Returns evaluation metrics including hit_rate."
    ),
)
async def train_model(req: TrainRequest) -> TrainResponse:
    if not req.symbols:
        raise HTTPException(status_code=422, detail="At least one symbol is required.")
    try:
        metrics = await asyncio.to_thread(
            regression_trainer.train,
            [s.upper() for s in req.symbols],
            req.quantile_alpha,
            req.lookback_days,
            req.forward_horizon,
            tuple(req.alphas) if req.alphas else None,
            tuple(req.horizons) if req.horizons else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Training error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")
    return TrainResponse(**metrics)


@router.post(
    "/v2/predict",
    response_model=PredictTPResponse,
    summary="Predict take-profit for a BUY entry",
    description=(
        "Builds live features from the most recent candle and predicts the "
        "conservative max return during the upcoming MACD bullish cycle "
        "(green-start → red-start). Returns the suggested sell price, the "
        "expected hit-rate, and whether the trade passes the minimum return "
        "filter."
    ),
)
async def predict_take_profit(req: PredictTPRequest) -> PredictTPResponse:
    if not regression_trainer.is_trained():
        raise HTTPException(
            status_code=409,
            detail="Regression model not trained. Call POST /regression/train first.",
        )
    try:
        result = await asyncio.to_thread(
            regression_predictor.predict_take_profit,
            req.symbol,
            req.buy_price,
            req.forward_candles,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error for %s: %s", req.symbol, exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Prediction failed: {exc}")
    return PredictTPResponse(**result)


@router.get(
    "/v2/status",
    response_model=RegressionStatusResponse,
    summary="Regression model status",
    description="Returns whether a trained regression model is available for prediction.",
)
async def model_status() -> RegressionStatusResponse:
    return RegressionStatusResponse(
        trained=regression_trainer.is_trained(),
        quantile_alpha=get_quantile_alpha(),
        hit_rate=get_hit_rate(),
    )
