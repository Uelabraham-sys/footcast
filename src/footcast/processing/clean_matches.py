"""Build the canonical Silver match dataset."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer

from footcast.config import Settings, get_settings
from footcast.processing.silver_validation import (
    validate_silver_matches,
)
from footcast.processing.team_mapping import (
    build_team_dimension,
    canonicalise_match_teams,
    load_team_aliases,
)

app = typer.Typer(help="Build canonical Silver football datasets.")


def load_historical_bronze(
    bronze_directory: Path,
) -> pl.DataFrame:
    """Load all historical Bronze match partitions."""
    paths = sorted(bronze_directory.glob("historical_matches/season=*/matches.parquet"))

    if not paths:
        raise FileNotFoundError("No historical Bronze match partitions were found.")

    return pl.concat(
        [pl.read_parquet(path) for path in paths],
        how="diagonal_relaxed",
    )


def load_optional_current_bronze(
    bronze_directory: Path,
) -> pl.DataFrame | None:
    """Load current API matches when available."""
    path = bronze_directory / "current" / "matches.parquet"

    if not path.exists():
        return None

    return pl.read_parquet(path)


def prepare_historical_matches(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Convert historical records to the canonical match schema."""
    return dataframe.with_columns(
        pl.col("match_date")
        .cast(pl.Date)
        .cast(pl.Datetime)
        .dt.replace_time_zone("UTC")
        .alias("kickoff_utc"),
        pl.lit("FINISHED").alias("status"),
        pl.col("source").cast(pl.String),
        pl.col("ingested_at").cast(
            pl.Datetime(time_zone="UTC"),
            strict=False,
        ),
    ).select(
        "season",
        "competition",
        "kickoff_utc",
        "match_date",
        "status",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "full_time_result",
        "source",
        "ingested_at",
    )


def prepare_current_matches(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Convert current API records to the canonical match schema."""
    return dataframe.with_columns(
        pl.col("kickoff_utc").cast(pl.Datetime(time_zone="UTC"), strict=False),
        pl.col("kickoff_utc").dt.date().alias("match_date"),
        pl.col("competition").fill_null("Premier League").alias("competition"),
        pl.col("source").cast(pl.String),
        pl.col("ingested_at").cast(
            pl.Datetime(time_zone="UTC"),
            strict=False,
        ),
    ).select(
        "season",
        "competition",
        "kickoff_utc",
        "match_date",
        "status",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "full_time_result",
        "source",
        "ingested_at",
    )


def create_match_keys(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Create deterministic canonical identifiers for matches."""
    return dataframe.with_columns(
        pl.concat_str(
            [
                pl.col("competition"),
                pl.col("season"),
                pl.col("match_date").cast(pl.String),
                pl.col("home_team_id"),
                pl.col("away_team_id"),
            ],
            separator="|",
        )
        .hash(seed=42)
        .cast(pl.String)
        .alias("match_key")
    )


def source_priority_expression() -> pl.Expr:
    """Return provider priority for deduplication."""
    return (
        pl.when(pl.col("source") == "football-data.org")
        .then(pl.lit(2))
        .when(pl.col("source") == "football-data.co.uk")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
    )


def deduplicate_matches(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Resolve duplicate matches using source and freshness priority."""
    return (
        dataframe.with_columns(source_priority_expression().alias("_source_priority"))
        .sort(
            [
                "match_key",
                "_source_priority",
                "ingested_at",
            ],
            descending=[False, True, True],
        )
        .unique(
            subset=["match_key"],
            keep="first",
            maintain_order=True,
        )
        .drop("_source_priority")
    )


def add_match_outcomes(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Add points and goal-difference outcomes."""
    return dataframe.with_columns(
        pl.when(pl.col("full_time_result") == "H")
        .then(pl.lit(3))
        .when(pl.col("full_time_result") == "D")
        .then(pl.lit(1))
        .when(pl.col("full_time_result") == "A")
        .then(pl.lit(0))
        .otherwise(pl.lit(None))
        .cast(pl.Int64)
        .alias("home_points"),
        pl.when(pl.col("full_time_result") == "A")
        .then(pl.lit(3))
        .when(pl.col("full_time_result") == "D")
        .then(pl.lit(1))
        .when(pl.col("full_time_result") == "H")
        .then(pl.lit(0))
        .otherwise(pl.lit(None))
        .cast(pl.Int64)
        .alias("away_points"),
        (pl.col("home_goals") - pl.col("away_goals")).alias("home_goal_difference"),
    )


def build_silver_matches(
    settings: Settings,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build canonical Silver match and team datasets."""
    aliases = load_team_aliases()

    historical = canonicalise_match_teams(
        prepare_historical_matches(load_historical_bronze(settings.bronze_directory)),
        aliases,
    )

    frames = [historical]

    current = load_optional_current_bronze(settings.bronze_directory)
    if current is not None and not current.is_empty():
        frames.append(
            canonicalise_match_teams(
                prepare_current_matches(current),
                aliases,
            )
        )

    combined = pl.concat(
        frames,
        how="diagonal_relaxed",
    )

    matches = (
        combined.pipe(create_match_keys)
        .pipe(deduplicate_matches)
        .pipe(add_match_outcomes)
        .sort(["kickoff_utc", "match_key"])
    )

    validate_silver_matches(matches)

    teams = build_team_dimension(matches)

    return matches, teams


def write_silver_datasets(
    matches: pl.DataFrame,
    teams: pl.DataFrame,
    silver_directory: Path,
) -> None:
    """Write canonical Silver datasets and build metadata."""
    silver_directory.mkdir(parents=True, exist_ok=True)

    matches_path = silver_directory / "matches.parquet"
    teams_path = silver_directory / "teams.parquet"
    report_path = silver_directory / "silver_build_report.json"

    matches.write_parquet(
        matches_path,
        compression="zstd",
        statistics=True,
    )
    teams.write_parquet(
        teams_path,
        compression="zstd",
        statistics=True,
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "match_count": matches.height,
        "team_count": teams.height,
        "season_count": matches["season"].n_unique(),
        "source_counts": (matches.group_by("source").len().sort("source").to_dicts()),
        "first_match": str(matches["match_date"].min()),
        "last_match": str(matches["match_date"].max()),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


@app.command()
def run() -> None:
    """Build the FootCast Silver datasets."""
    settings = get_settings()
    matches, teams = build_silver_matches(settings)

    write_silver_datasets(
        matches=matches,
        teams=teams,
        silver_directory=settings.silver_directory,
    )

    typer.echo(
        f"Built Silver datasets with {matches.height} matches and {teams.height} teams."
    )


if __name__ == "__main__":
    app()
