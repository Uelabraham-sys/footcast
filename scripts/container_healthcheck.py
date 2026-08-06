"""Run lightweight FootCast container health checks."""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Final

REQUIRED_MODULES: Final[tuple[str, ...]] = (
    "footcast",
    "footcast.features",
    "footcast.ingestion",
    "footcast.modelling",
    "footcast.pipelines",
    "footcast.prediction",
    "footcast.processing",
)

REQUIRED_DIRECTORIES: Final[tuple[Path, ...]] = (
    Path("/app/data"),
    Path("/app/artifacts"),
)


def validate_imports() -> None:
    """Ensure all primary FootCast packages import."""
    for module_name in REQUIRED_MODULES:
        importlib.import_module(module_name)


def validate_directories() -> None:
    """Ensure runtime directories exist and are writable."""
    for directory in REQUIRED_DIRECTORIES:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required runtime directory is missing: {directory}"
            )

        if not os.access(directory, os.W_OK):
            raise PermissionError(f"Runtime directory is not writable: {directory}")


def validate_temporary_storage() -> None:
    """Ensure temporary storage is available."""
    with tempfile.NamedTemporaryFile(
        prefix="footcast-health-",
        delete=True,
    ) as temporary_file:
        temporary_file.write(b"healthy")
        temporary_file.flush()


def main() -> None:
    """Run all container health validations."""
    validate_imports()
    validate_directories()
    validate_temporary_storage()

    print("FootCast container is healthy.")


if __name__ == "__main__":
    main()
