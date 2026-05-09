import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.regression_routes import router as regression_router
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.scheduler.apschedular import shutdown_scheduler, start_scheduler

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_scheduler()
    try:
        yield
    finally:
        shutdown_scheduler()


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="XGBoost regression API for crypto take-profit prediction.",
    lifespan=lifespan,
)

app.include_router(regression_router)
