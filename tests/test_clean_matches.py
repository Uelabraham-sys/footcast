"""Tests for canonical Silver match processing."""

from datetime import UTC, datetime

import polars as pl
import pytest

from footcast.processing.clean_matches import (
    add_match_outcomes,
    create_match_keys,
    deduplicate_matches,
)
from footcast.processing.silver_validation import (
    SilverDataValidationError,
    validate_silver_matches,
)


def create_canonical_matches() -> pl.DataFrame:
    """Create representative canonical match records."""
    return pl.DataFrame(
        {
            "season": ["2024/25", "2024/25"],
            "competition": [
                "Premier League",
                "Premier League",
            ],
            "kickoff_utc": [
                datetime(2024, 8, 10, tzinfo=UTC),
                datetime(2024, 8, 10, tzinfo=UTC),
            ],
            "match_date": [
                datetime(2024, 8, 10).date(),
                datetime(2024, 8, 10).date(),
            ],
            "status": ["FINISHED", "FINISHED"],
            "home_team_id": ["arsenal", "arsenal"],
            "home_team": ["Arsenal", "Arsenal"],
            "away_team_id": ["everton", "everton"],
            "away_team": ["Everton", "Everton"],
            "home_goals": [2, 2],
            "away_goals": [0, 0],
            "full_time_result": ["H", "H"],
            "source": [
                "football-data.co.uk",
                "football-data.org",
            ],
            "ingested_at": [
                datetime(2025, 5, 1, tzinfo=UTC),
                datetime(2025, 5, 2, tzinfo=UTC),
            ],
        }
    )


def test_match_key_is_deterministic() -> None:
    """Equal fixture identities should produce equal keys."""
    result = create_match_keys(create_canonical_matches())

    assert result["match_key"].n_unique() == 1


def test_api_source_wins_deduplication() -> None:
    """The current API source should win overlapping records."""
    result = (
        create_canonical_matches().pipe(create_match_keys).pipe(deduplicate_matches)
    )

    assert result.height == 1
    assert result["source"].item() == "football-data.org"


def test_add_match_outcomes() -> None:
    """Completed scores should produce points and goal difference."""
    result = (
        create_canonical_matches()
        .head(1)
        .pipe(create_match_keys)
        .pipe(add_match_outcomes)
    )

    assert result["home_points"].item() == 3
    assert result["away_points"].item() == 0
    assert result["home_goal_difference"].item() == 2


def test_valid_silver_match_passes() -> None:
    """A complete canonical match should pass validation."""
    result = (
        create_canonical_matches()
        .head(1)
        .pipe(create_match_keys)
        .pipe(add_match_outcomes)
    )

    validate_silver_matches(result)


def test_duplicate_match_keys_fail() -> None:
    """Duplicate canonical match keys must fail validation."""
    result = create_canonical_matches().pipe(create_match_keys).pipe(add_match_outcomes)

    with pytest.raises(
        SilverDataValidationError,
        match="duplicate canonical match keys",
    ):
        validate_silver_matches(result)
