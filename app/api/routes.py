"""
Trading signal API endpoints.

Both endpoints ALWAYS return a JSON array so consumers never need to branch
on the response shape regardless of how many symbols are requested.

Endpoints:
  GET /signal?symbol=BTCUSDT   → list with 1 item
  GET /signals                 → list with all cached items
  GET /health                  → liveness probe (no auth, no cache dependency)
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import SignalResponse
from app.services.cache import cache
from app.services.signal_engine import generate_signal

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/signal",
    response_model=list[SignalResponse],
    summary="Signal for a single symbol",
    description=(
        "Returns a one-element array containing the latest cached signal "
        "for the requested symbol.  If the symbol has never been processed "
        "by the worker, the signal is computed on-demand (may take ~2 s)."
    ),
)
async def get_signal(
    symbol: str = Query(..., description="Trading pair, e.g. BTCUSDT", examples=["BTCUSDT"])
) -> list[SignalResponse]:
    symbol = symbol.upper()

    # Fast path: cache hit
    cached = cache.get(symbol)
    if cached:
        return [cached]

    # Slow path: compute on-demand for symbols not in the worker's list
    logger.info("Cache miss for %s – computing on-demand", symbol)
    try:
        signal = await asyncio.to_thread(generate_signal, symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("On-demand signal failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail=f"Failed to fetch data for {symbol}: {exc}")

    # Persist so subsequent requests are fast
    cache.set(symbol, signal)
    return [signal]


@router.get(
    "/signals",
    response_model=list[SignalResponse],
    summary="Signals for all watched symbols",
    description=(
        "Returns the cached signals for every symbol in the worker's watch-list. "
        "The list is empty if the worker has not completed its first run yet."
    ),
)
async def get_signals() -> list[SignalResponse]:
    return cache.get_all()


@router.get(
    "/health",
    tags=["ops"],
    summary="Liveness probe",
)
async def health() -> dict:
    """Simple health check – returns 200 as long as the process is alive."""
    return {"status": "ok", "cached_symbols": len(cache)}
