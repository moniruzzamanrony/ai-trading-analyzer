"""
Binance public klines (candlestick) fetcher.

Binance returns 12 columns per kline, but standard OHLCV libraries only expose 6.
We call the REST endpoint directly so we can also capture:
  - quote_asset_volume      (col 7)
  - taker_buy_base_volume   (col 9)
  - taker_buy_quote_volume  (col 10)
These are required model features that measure real buy-side order flow.
"""

import logging
import httpx
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# Column names matching Binance kline response positions
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


def fetch_klines(symbol: str) -> pd.DataFrame:
    """
    Fetch the most recent *kline_limit* 5-minute candles for *symbol* from Binance.

    Returns a DataFrame indexed by UTC timestamp with float columns for all
    price/volume fields that the feature-engineering layer needs.

    Raises httpx.HTTPError on network or non-200 response.
    """
    url = f"{settings.binance_base_url}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": settings.kline_interval,
        "limit": settings.kline_limit,
    }

    logger.debug("Fetching klines: %s", params)

    with httpx.Client(timeout=15) as client:
        response = client.get(url, params=params)
        response.raise_for_status()

    raw = response.json()

    df = pd.DataFrame(raw, columns=_KLINE_COLUMNS)

    # Parse timestamp and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Keep only the columns we actually use; cast everything to float
    keep = ["open", "high", "low", "close", "volume",
            "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    df = df[keep].astype(float)

    return df
