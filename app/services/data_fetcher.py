import logging
import time

import httpx
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

_KLINE_COLUMNS = [
    "timestamp",
    "open", "high", "low", "close", "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Binance returns at most 1000 candles per /klines request.
_MAX_BINANCE_LIMIT = 1000

_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}


def fetch_klines(symbol: str, limit: int | None = None) -> pd.DataFrame:
    """
    Fetch the most recent candles for *symbol* from Binance REST in a single
    request. Capped at 1000 candles by the exchange.

    Returns a DataFrame indexed by UTC timestamp with OHLCV float columns.
    Raises httpx.HTTPError on network or non-200 response.
    """
    url = f"{settings.binance_base_url}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": settings.kline_interval,
        "limit": limit if limit is not None else settings.kline_limit,
    }

    logger.debug("Fetching klines: %s", params)

    with httpx.Client(timeout=15) as client:
        response = client.get(url, params=params)
        response.raise_for_status()

    df = pd.DataFrame(response.json(), columns=_KLINE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_klines_history(
    symbol: str,
    days: int,
    interval: str | None = None,
    request_pause: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch *days* of historical candles for *symbol* from Binance, paginating
    1000 candles at a time. Returns the same OHLCV DataFrame layout as
    ``fetch_klines``.

    Args:
        symbol:        Trading pair, e.g. "BTCUSDT".
        days:          Lookback in days, ending now.
        interval:      Kline interval; defaults to ``settings.kline_interval``.
        request_pause: Seconds to sleep between requests (Binance courtesy).
    """
    if days <= 0:
        raise ValueError("days must be positive")

    interval = interval or settings.kline_interval
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    step_ms = _INTERVAL_MS[interval] * _MAX_BINANCE_LIMIT
    url = f"{settings.binance_base_url}/api/v3/klines"

    rows: list[list] = []
    cursor = start_ms
    request_count = 0

    with httpx.Client(timeout=20) as client:
        while cursor < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor,
                "endTime": min(cursor + step_ms, end_ms),
                "limit": _MAX_BINANCE_LIMIT,
            }
            response = client.get(url, params=params)
            response.raise_for_status()
            batch = response.json()
            request_count += 1
            if not batch:
                cursor += step_ms
                continue
            rows.extend(batch)
            # Resume right after the last candle's open time to avoid duplicates.
            cursor = batch[-1][0] + _INTERVAL_MS[interval]
            time.sleep(request_pause)

    logger.info(
        "fetch_klines_history(%s, %dd, %s): %d candles in %d requests",
        symbol, days, interval, len(rows), request_count,
    )

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=_KLINE_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)
