import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_MAILJET_URL = "https://api.mailjet.com/v3.1/send"


async def send_email(to_email: str, subject: str, text: str, html: str | None = None) -> None:
    if not (settings.mailjet_api_key and settings.mailjet_api_secret and settings.mailjet_from_email):
        logger.warning("Mailjet not configured — skipping email")
        return

    payload = {
        "Messages": [
            {
                "From": {"Email": settings.mailjet_from_email},
                "To": [{"Email": to_email}],
                "Subject": subject,
                "TextPart": text,
                **({"HTMLPart": html} if html else {}),
            }
        ]
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            _MAILJET_URL,
            auth=(settings.mailjet_api_key, settings.mailjet_api_secret),
            json=payload,
        )
        resp.raise_for_status()
