"""Tests for the FootCast container health check."""

from pathlib import Path

import pytest
from scripts.container_healthcheck import (
    validate_directories,
    validate_imports,
    validate_temporary_storage,
)


def test_required_packages_import() -> None:
    """Primary FootCast packages should import."""
    validate_imports()


def test_temporary_storage_is_available() -> None:
    """Temporary storage should permit file creation."""
    validate_temporary_storage()


def test_missing_runtime_directory_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing runtime directories must fail health checks."""
    missing = tmp_path / "missing"

    monkeypatch.setattr(
        "scripts.container_healthcheck.REQUIRED_DIRECTORIES",
        (missing,),
    )

    with pytest.raises(
        FileNotFoundError,
        match="runtime directory",
    ):
        validate_directories()


def test_existing_writable_directory_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Writable runtime directories should pass."""
    monkeypatch.setattr(
        "scripts.container_healthcheck.REQUIRED_DIRECTORIES",
        (tmp_path,),
    )

    validate_directories()
