"""Tests for FootCast application configuration."""

from pathlib import Path

from footcast.config import PROJECT_ROOT, get_settings


def test_project_root_exists() -> None:
    """The resolved project root should exist."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_default_competition_is_premier_league() -> None:
    """FootCast should initially target the Premier League."""
    settings = get_settings()

    assert settings.competition_code == "PL"


def test_required_data_directories_exist() -> None:
    """Configuration should create the medallion data directories."""
    settings = get_settings()

    assert isinstance(settings.bronze_directory, Path)
    assert settings.bronze_directory.exists()
    assert settings.silver_directory.exists()
    assert settings.gold_directory.exists()


def test_api_uses_version_four() -> None:
    """The configured football-data URL should target API v4."""
    settings = get_settings()

    assert settings.football_data_base_url.rstrip("/").endswith("/v4")
