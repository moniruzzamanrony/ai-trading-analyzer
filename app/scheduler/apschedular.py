import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.config import settings
from app.services import regression_trainer
from app.services.mailer import send_email

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def async_job():
    result = await asyncio.to_thread(
        regression_trainer.train,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        lookback_days=60,
        forward_horizon=60,
        quantile_alpha=0.3,
    )

    recipient = settings.mailjet_error_recipient
    if recipient:
        try:
            await send_email(
                to_email=recipient,
                subject="Regression training completed",
                text=f"Daily regression training finished successfully.\n\nResult:\n{result}",
            )
        except Exception:
            logger.exception("Failed to send training-success email")


def start_scheduler():
    if not scheduler.running:
        scheduler.add_job(async_job, "interval", hours=24)
        scheduler.start()


def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)