"""Tests for FootCast logging configuration."""

import logging
from pathlib import Path

from footcast.logging_config import configure_logging


def test_configure_logging_creates_log_file(
    tmp_path: Path,
) -> None:
    """Logging configuration should create a writable log file."""
    log_path = tmp_path / "footcast.log"

    configure_logging(
        level="INFO",
        log_file=log_path,
    )

    logger = logging.getLogger("footcast.test")
    logger.info("Test log message")

    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_path.exists()
    assert "Test log message" in log_path.read_text(encoding="utf-8")
