"""Audit utilities for FootCast Bronze datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer

from footcast.config import get_settings
from footcast.ingestion.current_validation import (
    validate_current_matches,
    validate_current_standings,
)
from footcast.ingestion.validation import validate_historical_matches
from footcast.logging_config import configure_logging

app = typer.Typer(help="Audit FootCast Bronze ingestion outputs.")


class BronzeAuditError(RuntimeError):
    """Raised when Bronze datasets fail audit requirements."""


@dataclass(frozen=True)
class HistoricalSeasonAudit:
    """Audit result for one historical season partition."""

    season_partition: str
    row_count: int
    unique_match_ids: int
    first_match_date: str
    last_match_date: str
    team_count: int


@dataclass(frozen=True)
class BronzeAuditReport:
    """Consolidated audit report for Bronze datasets."""

    generated_at: str
    historical_season_count: int
    historical_match_count: int
    current_match_count: int
    current_finished_count: int
    current_scheduled_count: int
    standings_row_count: int
    overall_standings_team_count: int
    historical_seasons: list[HistoricalSeasonAudit]


def find_historical_parquet_files(
    bronze_directory: Path,
) -> list[Path]:
    """Return historical match Parquet paths in partition order."""
    historical_root = bronze_directory / "historical_matches"

    return sorted(historical_root.glob("season=*/matches.parquet"))


def load_historical_matches(
    bronze_directory: Path,
) -> tuple[pl.DataFrame, list[HistoricalSeasonAudit]]:
    """Load and validate every historical season partition."""
    paths = find_historical_parquet_files(bronze_directory)

    if not paths:
        raise BronzeAuditError("No historical match Parquet files were found.")

    frames: list[pl.DataFrame] = []
    season_audits: list[HistoricalSeasonAudit] = []

    for path in paths:
        dataframe = pl.read_parquet(path)
        validate_historical_matches(dataframe)

        partition_name = path.parent.name

        season_audits.append(
            HistoricalSeasonAudit(
                season_partition=partition_name,
                row_count=dataframe.height,
                unique_match_ids=dataframe["match_id"].n_unique(),
                first_match_date=str(dataframe["match_date"].min()),
                last_match_date=str(dataframe["match_date"].max()),
                team_count=pl.concat(
                    [
                        dataframe.select(pl.col("home_team").alias("team")),
                        dataframe.select(pl.col("away_team").alias("team")),
                    ]
                )["team"].n_unique(),
            )
        )
        frames.append(dataframe)

    historical = pl.concat(
        frames,
        how="diagonal_relaxed",
    ).sort(["match_date", "match_id"])

    if historical["match_id"].n_unique() != historical.height:
        duplicate_count = historical.height - historical["match_id"].n_unique()
        raise BronzeAuditError(
            f"Historical datasets contain {duplicate_count} duplicate match IDs."
        )

    return historical, season_audits


def load_current_matches(bronze_directory: Path) -> pl.DataFrame:
    """Load and validate the latest current-match snapshot."""
    path = bronze_directory / "current" / "matches.parquet"

    if not path.exists():
        raise BronzeAuditError(f"Current matches file does not exist: {path}")

    dataframe = pl.read_parquet(path)
    validate_current_matches(dataframe)

    return dataframe


def load_current_standings(
    bronze_directory: Path,
) -> pl.DataFrame:
    """Load and validate the latest standings snapshot."""
    path = bronze_directory / "current" / "standings.parquet"

    if not path.exists():
        raise BronzeAuditError(f"Current standings file does not exist: {path}")

    dataframe = pl.read_parquet(path)
    validate_current_standings(dataframe)

    return dataframe


def build_bronze_audit_report(
    bronze_directory: Path,
) -> BronzeAuditReport:
    """Build a consolidated Bronze audit report."""
    historical, season_audits = load_historical_matches(bronze_directory)
    current = load_current_matches(bronze_directory)
    standings = load_current_standings(bronze_directory)

    overall_standings = standings.filter(pl.col("standing_type") == "TOTAL")

    finished_count = current.filter(pl.col("status") == "FINISHED").height

    scheduled_count = current.filter(pl.col("status") != "FINISHED").height

    return BronzeAuditReport(
        generated_at=datetime.now(UTC).isoformat(),
        historical_season_count=len(season_audits),
        historical_match_count=historical.height,
        current_match_count=current.height,
        current_finished_count=finished_count,
        current_scheduled_count=scheduled_count,
        standings_row_count=standings.height,
        overall_standings_team_count=overall_standings.height,
        historical_seasons=season_audits,
    )


def write_bronze_audit_report(
    report: BronzeAuditReport,
    bronze_directory: Path,
) -> Path:
    """Write the consolidated audit report to JSON."""
    report_directory = bronze_directory / "manifests"
    report_directory.mkdir(parents=True, exist_ok=True)

    output_path = report_directory / "bronze_audit.json"
    output_path.write_text(
        json.dumps(asdict(report), indent=2),
        encoding="utf-8",
    )

    return output_path


def print_bronze_audit_report(
    report: BronzeAuditReport,
) -> None:
    """Print a human-readable Bronze audit summary."""
    typer.echo("")
    typer.echo("FootCast Bronze Audit")
    typer.echo("=" * 40)
    typer.echo(f"Historical seasons: {report.historical_season_count}")
    typer.echo(f"Historical matches: {report.historical_match_count}")
    typer.echo(f"Current matches: {report.current_match_count}")
    typer.echo(f"Current finished: {report.current_finished_count}")
    typer.echo(f"Current scheduled: {report.current_scheduled_count}")
    typer.echo(f"Standings rows: {report.standings_row_count}")
    typer.echo(f"Overall-table teams: {report.overall_standings_team_count}")
    typer.echo("")

    typer.echo("Historical partitions")
    typer.echo("-" * 40)

    for season in report.historical_seasons:
        typer.echo(
            f"{season.season_partition}: "
            f"{season.row_count} matches, "
            f"{season.team_count} teams, "
            f"{season.first_match_date} to "
            f"{season.last_match_date}"
        )


@app.command()
def run() -> None:
    """Audit all Bronze ingestion outputs."""
    settings = get_settings()
    configure_logging(settings.footcast_log_level)

    report = build_bronze_audit_report(settings.bronze_directory)
    output_path = write_bronze_audit_report(
        report,
        settings.bronze_directory,
    )

    print_bronze_audit_report(report)
    typer.echo(f"Audit report written to: {output_path}")


if __name__ == "__main__":
    app()
