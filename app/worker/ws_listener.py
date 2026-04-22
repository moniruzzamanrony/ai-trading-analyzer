"""
Binance WebSocket kline listener.

Subscribes to the combined stream for all configured symbols.
When a kline closes (k.x == True) it triggers REST-based signal
generation and updates the cache — exactly once per closed candle.

Reconnects automatically on any network error.
"""

import asyncio
import json
import logging

import websockets

from app.core.config import settings
from app.services.cache import cache
from app.services.signal_engine import generate_signal

logger = logging.getLogger(__name__)

_WS_BASE = "wss://stream.binance.com:9443/stream"


def _stream_url() -> str:
    streams = "/".join(
        f"{s.lower()}@kline_{settings.kline_interval}"
        for s in settings.symbol_list
    )
    return f"{_WS_BASE}?streams={streams}"


async def _on_kline_close(symbol: str) -> None:
    try:
        signal = await asyncio.to_thread(generate_signal, symbol)
        cache.set(symbol, signal)
    except Exception as exc:
        logger.error("Signal generation failed for %s: %s", symbol, exc)


async def run() -> None:
    """Connect to Binance and process kline events forever (with auto-reconnect)."""
    url = _stream_url()
    logger.info("Connecting to Binance WebSocket | streams: %s", settings.symbol_list)

    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                logger.info("WebSocket connected")
                async for raw in ws:
                    msg  = json.loads(raw)
                    k    = msg.get("data", {}).get("k", {})
                    if k.get("x"):   # kline is closed
                        symbol = k["s"]
                        logger.info("Kline closed | %s – generating signal", symbol)
                        asyncio.create_task(_on_kline_close(symbol))

        except Exception as exc:
            logger.warning("WebSocket error: %s – reconnecting in 5s", exc)
            await asyncio.sleep(5)
