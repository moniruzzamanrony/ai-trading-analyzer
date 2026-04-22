"""
Application entry point.

Responsibilities:
  - Create the FastAPI app.
  - Register the API router.
  - Manage the APScheduler background worker lifecycle via FastAPI's lifespan.

The lifespan context manager ensures the worker runs immediately at startup
(so the cache is populated before the first request) and is cleanly shut down
when the server exits.
"""

import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.worker.scheduler import refresh_all_signals

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context.

    Startup:
      1. Trigger an immediate signal refresh so the cache is ready.
      2. Schedule subsequent refreshes every WORKER_INTERVAL_MINUTES minutes.

    Shutdown:
      - APScheduler is gracefully stopped (waits for any in-flight job to finish).
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info(
        "Starting %s v%s | symbols=%s | interval=%dm",
        settings.app_title,
        settings.app_version,
        settings.symbol_list,
        settings.worker_interval_minutes,
    )

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        refresh_all_signals,
        trigger="interval",
        minutes=settings.worker_interval_minutes,
        id="signal_refresh",
        max_instances=1,          # prevent overlapping runs if one takes too long
        replace_existing=True,
    )
    scheduler.start()

    # Populate cache immediately instead of waiting for the first interval tick
    await refresh_all_signals()

    yield  # ←── application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    scheduler.shutdown(wait=True)
    logger.info("Scheduler stopped – shutting down.")


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Real-time 5-minute trading signal API powered by XGBoost. "
        "All endpoints return a JSON array for consistent consumer handling."
    ),
    lifespan=lifespan,
)

app.include_router(router)
