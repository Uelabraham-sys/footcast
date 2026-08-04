"""Logging configuration for FootCast applications."""

from __future__ import annotations

import logging
from pathlib import Path

from footcast.config import PROJECT_ROOT


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure console and file logging."""
    resolved_log_file = (
        log_file if log_file is not None else PROJECT_ROOT / "logs" / "footcast.log"
    )
    resolved_log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        resolved_log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
