"""
Application entry point.

Responsibilities:
  - Create the FastAPI app.
  - Register the API router.
  - On startup: populate the cache via REST once, then connect to
    Binance WebSocket kline streams for event-driven signal updates.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.services.cache import cache
from app.services.signal_engine import generate_signal
from app.worker import ws_listener

configure_logging()
logger = logging.getLogger(__name__)


async def _initial_populate() -> None:
    """Warm the cache via REST before the first WebSocket kline closes."""
    logger.info("Populating cache for %s", settings.symbol_list)
    for symbol in settings.symbol_list:
        try:
            signal = await asyncio.to_thread(generate_signal, symbol)
            cache.set(symbol, signal)
        except Exception as exc:
            logger.error("Initial signal failed for %s: %s", symbol, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s | symbols=%s",
        settings.app_title,
        settings.app_version,
        settings.symbol_list,
    )

    await _initial_populate()

    ws_task = asyncio.create_task(ws_listener.run())

    yield

    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass
    logger.info("WebSocket listener stopped – shutting down.")


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Real-time 5-minute trading signal API powered by XGBoost. "
        "Signals update on every closed Binance kline via WebSocket."
    ),
    lifespan=lifespan,
)

app.include_router(router)
