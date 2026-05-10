import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.regression_routes import router as regression_router
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.services import regression_trainer

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if regression_trainer.is_trained():
        try:
            bundle = regression_trainer.load_bundle()
            logger.info(
                "Loaded regression model from disk: horizons=%s alphas=%s features=%d",
                bundle.get("horizons"),
                bundle.get("alphas"),
                len(bundle.get("feature_cols", [])),
            )
        except Exception:
            logger.exception("Failed to pre-load regression model on startup")
    else:
        logger.warning(
            "No trained regression model found at startup; "
            "POST /v2/regression/train before calling /v2/regression/predict."
        )
    yield


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="XGBoost regression API for crypto take-profit prediction.",
    lifespan=lifespan,
)

app.include_router(regression_router)
