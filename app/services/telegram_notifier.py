"""
Telegram notification service.

Sends a formatted alert to a configured Telegram chat whenever a BUY or SELL
signal is produced.  Uses httpx (already in requirements) for async HTTP.

Configuration (via .env):
    TELEGRAM_BOT_TOKEN  – bot token from @BotFather
    TELEGRAM_CHAT_ID    – target chat / channel / user ID
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_SIGNAL_EMOJI = {"BUY": "🟢", "SELL": "🔴"}
_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def _build_message(symbol: str, signal: str, price: float, probability: float,
                   volatility: float, regime: str) -> str:
    emoji = _SIGNAL_EMOJI.get(signal, "⚪")
    return (
        f"{emoji} *{signal} Signal – {symbol}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price:       `{price:,.4f}`\n"
        f"📊 Confidence:  `{probability:.1%}`\n"
        f"📈 Volatility:  `{volatility:.2f}%`\n"
        f"🏳 Regime:      `{regime}`"
    )


async def send_signal_alert(symbol: str, signal: str, price: float,
                            probability: float, volatility: float,
                            regime: str) -> None:
    """Fire-and-forget Telegram alert; swallows all errors to protect the worker."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.debug("Telegram not configured – skipping alert for %s %s", signal, symbol)
        return

    text = _build_message(symbol, signal, price, probability, volatility, regime)
    url  = _API_URL.format(token=settings.telegram_bot_token)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json={
                "chat_id":    settings.telegram_chat_id,
                "text":       text,
                "parse_mode": "Markdown",
            })
            if resp.status_code != 200:
                logger.warning("Telegram API error %s: %s", resp.status_code, resp.text)
            else:
                logger.info("Telegram alert sent | %s %s", signal, symbol)
    except Exception as exc:
        logger.warning("Failed to send Telegram alert for %s %s: %s", signal, symbol, exc)
