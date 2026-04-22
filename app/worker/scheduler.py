"""
Background worker that refreshes trading signals on a fixed schedule.

Design:
  - Runs every N minutes (configured via WORKER_INTERVAL_MINUTES).
  - Processes every symbol in the SYMBOLS list sequentially.
  - Results are written to the shared in-memory cache.
  - Errors for one symbol do not abort the others.
  - Runs immediately at startup so the cache is populated before the first
    API request arrives.
"""

import asyncio
import logging

from app.core.config import settings
from app.services.cache import cache
from app.services.signal_engine import generate_signal

logger = logging.getLogger(__name__)


async def refresh_all_signals() -> None:
    """
    Compute and cache signals for every configured symbol.

    Each symbol's computation is offloaded to a thread pool via
    asyncio.to_thread() so synchronous I/O (httpx, pickle, numpy) never
    blocks the event loop.
    """
    symbols = settings.symbol_list
    logger.info("Worker starting – processing %d symbol(s): %s", len(symbols), symbols)

    for symbol in symbols:
        try:
            signal = await asyncio.to_thread(generate_signal, symbol)
            cache.set(symbol, signal)
        except Exception as exc:
            # Log but continue – a bad symbol should not stall the others
            logger.error("Failed to generate signal for %s: %s", symbol, exc)

    logger.info("Worker finished – cache now holds %d signal(s)", len(cache))
