"""FastAPI application entry-point for the AI Room Classifier microservice."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from loguru import logger

from app.logging import setup_logging
from app.model import init_model, is_model_loaded
from app.routers import predict
from app.schemas import HealthResponse
from app.settings import settings
from app.telemetry import get_metrics_response, setup_telemetry

# Configure logging before anything else
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model and telemetry at startup."""
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")

    setup_telemetry()
    logger.info("Telemetry initialized")

    # Fail-fast: crash if model cannot be loaded — orchestrator will restart
    try:
        init_model()
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model file not found at '{settings.model_path}' — crashing")
        raise
    except Exception:
        logger.exception("Failed to load model — crashing")
        raise

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.service_name}")


app = FastAPI(
    title="AI Room Classifier",
    description="Classify real estate photos into room types using EfficientNet-B0.",
    version=settings.service_version,
    lifespan=lifespan,
)

# Register routers
app.include_router(predict.router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — confirms the service and model are ready."""
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    body, content_type = get_metrics_response()
    return Response(content=body, media_type=content_type)


@app.get("/")
async def root() -> dict[str, str]:
    """Service root — redirects to docs."""
    return {"message": "AI Room Classifier — see /docs for API documentation"}
