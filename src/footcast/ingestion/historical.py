"""Historical Premier League match ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import httpx
import polars as pl
import typer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from footcast.config import get_settings
from footcast.ingestion.validation import validate_historical_matches

LOGGER = logging.getLogger(__name__)

FOOTBALL_DATA_BASE_URL: Final[str] = "https://www.football-data.co.uk/mmz4281"
PREMIER_LEAGUE_DIVISION_CODE: Final[str] = "E0"

SOURCE_COLUMN_MAPPING: Final[dict[str, str]] = {
    "Date": "match_date",
    "Time": "kickoff_time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "full_time_result",
    "HTHG": "half_time_home_goals",
    "HTAG": "half_time_away_goals",
    "HTR": "half_time_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
}

REQUIRED_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
)

app = typer.Typer(help="Download and normalise historical Premier League results.")


@dataclass(frozen=True)
class IngestionManifestEntry:
    """Metadata describing one ingested season."""

    season: str
    season_code: str
    source_url: str
    raw_csv_path: str
    parquet_path: str
    row_count: int
    file_sha256: str
    ingested_at: str


def season_label(start_year: int) -> str:
    """Convert a starting year into a football-season label."""
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def season_code(start_year: int) -> str:
    """Convert a starting year into the source's four-digit season code."""
    first_year = str(start_year)[-2:]
    second_year = str(start_year + 1)[-2:]
    return f"{first_year}{second_year}"


def historical_csv_url(start_year: int) -> str:
    """Build the Football-Data CSV URL for one Premier League season."""
    code = season_code(start_year)
    return f"{FOOTBALL_DATA_BASE_URL}/{code}/{PREMIER_LEAGUE_DIVISION_CODE}.csv"


def calculate_sha256(content: bytes) -> str:
    """Calculate the SHA-256 checksum of downloaded content."""
    return hashlib.sha256(content).hexdigest()


@retry(
    retry=retry_if_exception_type(
        (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        )
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def download_csv(url: str, timeout_seconds: float = 30.0) -> bytes:
    """Download a historical season CSV with retry handling."""
    headers = {"User-Agent": ("FootCast/0.1 (historical football analytics project)")}

    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = client.get(url)
        response.raise_for_status()

    if not response.content:
        message = f"Downloaded an empty response from {url}"
        raise ValueError(message)

    return response.content


def save_raw_csv(
    content: bytes,
    start_year: int,
    bronze_directory: Path,
) -> Path:
    """Persist an untouched source CSV in Bronze storage."""
    code = season_code(start_year)
    raw_directory = bronze_directory / "historical_matches" / f"season={code}"
    raw_directory.mkdir(parents=True, exist_ok=True)

    output_path = raw_directory / "source.csv"
    output_path.write_bytes(content)

    return output_path


def read_source_csv(csv_path: Path) -> pl.DataFrame:
    """Read a downloaded historical CSV into Polars."""
    dataframe = pl.read_csv(
        csv_path,
        infer_schema_length=10_000,
        ignore_errors=False,
        null_values=["", "NA", "N/A"],
        truncate_ragged_lines=True,
    )

    missing_columns = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(dataframe.columns))

    if missing_columns:
        message = (
            f"Source CSV {csv_path} is missing required columns: {missing_columns}"
        )
        raise ValueError(message)

    return dataframe


def parse_match_date_expression() -> pl.Expr:
    """Return an expression supporting common source date formats."""
    date_text = pl.col("match_date").cast(pl.String).str.strip_chars()

    return pl.coalesce(
        date_text.str.strptime(pl.Date, "%d/%m/%Y", strict=False),
        date_text.str.strptime(pl.Date, "%d/%m/%y", strict=False),
        date_text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    )


def normalise_historical_matches(
    source: pl.DataFrame,
    start_year: int,
    ingested_at: datetime | None = None,
) -> pl.DataFrame:
    """Normalise a source CSV into the FootCast historical schema."""
    timestamp = ingested_at or datetime.now(UTC)
    available_mapping = {
        source_name: target_name
        for source_name, target_name in SOURCE_COLUMN_MAPPING.items()
        if source_name in source.columns
    }

    dataframe = source.rename(available_mapping)

    required_normalised_columns = (
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "full_time_result",
    )

    missing_columns = sorted(set(required_normalised_columns) - set(dataframe.columns))

    if missing_columns:
        message = f"Normalised data is missing columns: {missing_columns}"
        raise ValueError(message)

    optional_integer_columns = [
        column
        for column in (
            "half_time_home_goals",
            "half_time_away_goals",
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
            "home_fouls",
            "away_fouls",
            "home_corners",
            "away_corners",
            "home_yellow_cards",
            "away_yellow_cards",
            "home_red_cards",
            "away_red_cards",
        )
        if column in dataframe.columns
    ]

    dataframe = dataframe.with_columns(
        parse_match_date_expression().alias("match_date"),
        pl.col("home_team").cast(pl.String).str.strip_chars(),
        pl.col("away_team").cast(pl.String).str.strip_chars(),
        pl.col("home_goals").cast(pl.Int64, strict=False),
        pl.col("away_goals").cast(pl.Int64, strict=False),
        pl.col("full_time_result").cast(pl.String).str.strip_chars().str.to_uppercase(),
        *[
            pl.col(column).cast(pl.Int64, strict=False)
            for column in optional_integer_columns
        ],
    )

    season = season_label(start_year)

    dataframe = dataframe.with_columns(
        pl.lit(season).alias("season"),
        pl.lit("Premier League").alias("competition"),
        pl.lit("football-data.co.uk").alias("source"),
        pl.lit(timestamp).alias("ingested_at"),
    )

    dataframe = dataframe.with_columns(
        pl.concat_str(
            [
                pl.col("season"),
                pl.col("match_date").cast(pl.String),
                pl.col("home_team"),
                pl.col("away_team"),
            ],
            separator="|",
        )
        .hash(seed=42)
        .cast(pl.String)
        .alias("match_id")
    )

    preferred_order = [
        "match_id",
        "season",
        "competition",
        "match_date",
        "kickoff_time",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "full_time_result",
        "half_time_home_goals",
        "half_time_away_goals",
        "half_time_result",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_fouls",
        "away_fouls",
        "home_corners",
        "away_corners",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
        "source",
        "ingested_at",
    ]

    selected_columns = [
        column for column in preferred_order if column in dataframe.columns
    ]

    return dataframe.select(selected_columns).sort(
        ["match_date", "home_team", "away_team"]
    )


def save_bronze_parquet(
    dataframe: pl.DataFrame,
    start_year: int,
    bronze_directory: Path,
) -> Path:
    """Write normalised Bronze data as a partitioned Parquet file."""
    code = season_code(start_year)
    output_directory = bronze_directory / "historical_matches" / f"season={code}"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "matches.parquet"
    dataframe.write_parquet(
        output_path,
        compression="zstd",
        statistics=True,
    )

    return output_path


def ingest_season(
    start_year: int,
    bronze_directory: Path,
    timeout_seconds: float = 30.0,
) -> IngestionManifestEntry:
    """Download, normalise, validate and persist one season."""
    url = historical_csv_url(start_year)
    LOGGER.info(
        "Downloading season %s from %s",
        season_label(start_year),
        url,
    )

    content = download_csv(url, timeout_seconds=timeout_seconds)
    checksum = calculate_sha256(content)

    raw_path = save_raw_csv(
        content=content,
        start_year=start_year,
        bronze_directory=bronze_directory,
    )

    source = read_source_csv(raw_path)
    normalised = normalise_historical_matches(
        source=source,
        start_year=start_year,
    )

    validate_historical_matches(normalised)

    parquet_path = save_bronze_parquet(
        dataframe=normalised,
        start_year=start_year,
        bronze_directory=bronze_directory,
    )

    LOGGER.info(
        "Ingested %s: %s matches",
        season_label(start_year),
        normalised.height,
    )

    return IngestionManifestEntry(
        season=season_label(start_year),
        season_code=season_code(start_year),
        source_url=url,
        raw_csv_path=str(raw_path),
        parquet_path=str(parquet_path),
        row_count=normalised.height,
        file_sha256=checksum,
        ingested_at=datetime.now(UTC).isoformat(),
    )


def write_manifest(
    entries: list[IngestionManifestEntry],
    bronze_directory: Path,
) -> Path:
    """Write ingestion metadata for all downloaded seasons."""
    manifest_directory = bronze_directory / "manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_directory / "historical_ingestion.json"

    payload = {
        "dataset": "Premier League historical matches",
        "provider": "football-data.co.uk",
        "generated_at": datetime.now(UTC).isoformat(),
        "season_count": len(entries),
        "total_matches": sum(entry.row_count for entry in entries),
        "seasons": [asdict(entry) for entry in entries],
    }

    manifest_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return manifest_path


def ingest_historical_range(
    start_year: int,
    end_year: int,
    bronze_directory: Path,
) -> list[IngestionManifestEntry]:
    """Ingest an inclusive range of Premier League seasons."""
    if start_year > end_year:
        message = "start_year must be less than or equal to end_year"
        raise ValueError(message)

    entries = [
        ingest_season(
            start_year=year,
            bronze_directory=bronze_directory,
        )
        for year in range(start_year, end_year + 1)
    ]

    write_manifest(
        entries=entries,
        bronze_directory=bronze_directory,
    )

    return entries


@app.command()
def run(
    start_year: int = typer.Option(
        2019,
        help="First season starting year, for example 2019 for 2019/20.",
    ),
    end_year: int = typer.Option(
        2025,
        help="Final season starting year, inclusive.",
    ),
) -> None:
    """Download and process a range of Premier League seasons."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = get_settings()

    entries = ingest_historical_range(
        start_year=start_year,
        end_year=end_year,
        bronze_directory=settings.bronze_directory,
    )

    total_matches = sum(entry.row_count for entry in entries)

    typer.echo(
        f"Ingested {len(entries)} seasons and {total_matches} historical matches."
    )


if __name__ == "__main__":
    app()
