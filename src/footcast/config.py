"""Application configuration for FootCast."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-based FootCast settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    football_data_api_key: str = Field(
        default="",
        description="Authentication token for football-data.org.",
    )
    football_data_base_url: str = Field(
        default="https://api.football-data.org/v4",
        description="Base URL for the football-data.org API.",
    )
    footcast_environment: str = Field(default="development")
    footcast_log_level: str = Field(default="INFO")

    competition_code: str = "PL"

    bronze_directory: Path = PROJECT_ROOT / "data" / "bronze"
    silver_directory: Path = PROJECT_ROOT / "data" / "silver"
    gold_directory: Path = PROJECT_ROOT / "data" / "gold"

    def ensure_directories(self) -> None:
        """Create required local data directories when missing."""
        for directory in (
            self.bronze_directory,
            self.silver_directory,
            self.gold_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
