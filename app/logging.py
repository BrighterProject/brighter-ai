import sys

from loguru import logger

from app.settings import settings


def setup_logging() -> None:
    """Configure loguru with consistent formatting across BrighterProject services."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} — {message}",
        colorize=True,
    )
