"""Tests for consolidated Bronze ingestion audits."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from footcast.ingestion.audit import (
    BronzeAuditError,
    build_bronze_audit_report,
    load_historical_matches,
)


def create_historical_frame() -> pl.DataFrame:
    """Create a valid historical match dataframe."""
    return pl.DataFrame(
        {
            "match_id": ["match-1", "match-2"],
            "season": ["2024/25", "2024/25"],
            "competition": [
                "Premier League",
                "Premier League",
            ],
            "match_date": [
                datetime(2024, 8, 10).date(),
                datetime(2024, 8, 11).date(),
            ],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Everton", "Liverpool"],
            "home_goals": [2, 1],
            "away_goals": [0, 1],
            "full_time_result": ["H", "D"],
            "source": [
                "football-data.co.uk",
                "football-data.co.uk",
            ],
            "ingested_at": [
                datetime(2026, 8, 4, tzinfo=UTC),
                datetime(2026, 8, 4, tzinfo=UTC),
            ],
        }
    )


def create_current_matches_frame() -> pl.DataFrame:
    """Create a valid current-match dataframe."""
    return pl.DataFrame(
        {
            "match_id": [
                "football-data.org:1",
                "football-data.org:2",
            ],
            "api_match_id": [1, 2],
            "competition_code": ["PL", "PL"],
            "season": ["2025/26", "2025/26"],
            "kickoff_utc": [
                datetime(2025, 8, 10, tzinfo=UTC),
                datetime(2025, 8, 11, tzinfo=UTC),
            ],
            "status": ["FINISHED", "TIMED"],
            "home_team_id": [57, 61],
            "home_team": ["Arsenal FC", "Chelsea FC"],
            "away_team_id": [65, 64],
            "away_team": [
                "Manchester City FC",
                "Liverpool FC",
            ],
            "home_goals": [2, None],
            "away_goals": [1, None],
            "source": [
                "football-data.org",
                "football-data.org",
            ],
            "ingested_at": [
                datetime(2026, 8, 4, tzinfo=UTC),
                datetime(2026, 8, 4, tzinfo=UTC),
            ],
        }
    )


def create_standings_frame() -> pl.DataFrame:
    """Create a valid current standings dataframe."""
    return pl.DataFrame(
        {
            "competition_code": ["PL", "PL"],
            "season": ["2025/26", "2025/26"],
            "standing_type": ["TOTAL", "TOTAL"],
            "position": [1, 2],
            "team_id": [57, 65],
            "team": ["Arsenal FC", "Manchester City FC"],
            "played_games": [10, 10],
            "won": [8, 7],
            "draw": [1, 2],
            "lost": [1, 1],
            "points": [25, 23],
            "goals_for": [25, 23],
            "goals_against": [8, 10],
            "goal_difference": [17, 13],
            "source": [
                "football-data.org",
                "football-data.org",
            ],
            "ingested_at": [
                datetime(2026, 8, 4, tzinfo=UTC),
                datetime(2026, 8, 4, tzinfo=UTC),
            ],
        }
    )


def write_test_datasets(root: Path) -> None:
    """Write valid Bronze test datasets."""
    historical_directory = root / "historical_matches" / "season=2425"
    historical_directory.mkdir(parents=True)
    create_historical_frame().write_parquet(historical_directory / "matches.parquet")

    current_directory = root / "current"
    current_directory.mkdir(parents=True)
    create_current_matches_frame().write_parquet(current_directory / "matches.parquet")
    create_standings_frame().write_parquet(current_directory / "standings.parquet")


def test_load_historical_matches(
    tmp_path: Path,
) -> None:
    """Historical partitions should load into one dataframe."""
    write_test_datasets(tmp_path)

    matches, seasons = load_historical_matches(tmp_path)

    assert matches.height == 2
    assert len(seasons) == 1
    assert seasons[0].season_partition == "season=2425"
    assert seasons[0].row_count == 2
    assert seasons[0].unique_match_ids == 2


def test_build_bronze_audit_report(
    tmp_path: Path,
) -> None:
    """The consolidated report should contain dataset counts."""
    write_test_datasets(tmp_path)

    report = build_bronze_audit_report(tmp_path)

    assert report.historical_season_count == 1
    assert report.historical_match_count == 2
    assert report.current_match_count == 2
    assert report.current_finished_count == 1
    assert report.current_scheduled_count == 1
    assert report.standings_row_count == 2
    assert report.overall_standings_team_count == 2


def test_missing_historical_data_fails(
    tmp_path: Path,
) -> None:
    """The audit should fail when historical data is absent."""
    with pytest.raises(
        BronzeAuditError,
        match="No historical match Parquet files",
    ):
        load_historical_matches(tmp_path)


def test_duplicate_historical_ids_fail(
    tmp_path: Path,
) -> None:
    """Duplicate IDs across partitions should fail the audit."""
    first_directory = tmp_path / "historical_matches" / "season=2324"
    second_directory = tmp_path / "historical_matches" / "season=2425"

    first_directory.mkdir(parents=True)
    second_directory.mkdir(parents=True)

    dataframe = create_historical_frame()

    dataframe.write_parquet(first_directory / "matches.parquet")
    dataframe.write_parquet(second_directory / "matches.parquet")

    with pytest.raises(
        BronzeAuditError,
        match="duplicate match IDs",
    ):
        load_historical_matches(tmp_path)
