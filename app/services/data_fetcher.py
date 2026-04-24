import logging
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


def fetch_klines(symbol: str, limit: int | None = None) -> pd.DataFrame:
    """
    Fetch the most recent 5-minute candles for *symbol* from Binance REST.
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
