"""Tests for match-level rolling feature construction."""

from datetime import UTC, datetime, timedelta

import polars as pl

from footcast.features.build_features import (
    build_match_form_features,
)
from footcast.features.form import build_team_form_features
from footcast.features.validation import (
    validate_match_form_features,
)


def create_matches() -> pl.DataFrame:
    """Create representative Silver match rows."""
    start = datetime(2024, 8, 1, 15, 0, tzinfo=UTC)

    return pl.DataFrame(
        {
            "match_key": ["m1", "m2"],
            "season": ["2024/25", "2024/25"],
            "competition": [
                "Premier League",
                "Premier League",
            ],
            "kickoff_utc": [
                start,
                start + timedelta(days=7),
            ],
            "match_date": [
                start.date(),
                (start + timedelta(days=7)).date(),
            ],
            "status": ["FINISHED", "FINISHED"],
            "home_team_id": ["arsenal", "chelsea"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team_id": ["chelsea", "arsenal"],
            "away_team": ["Chelsea", "Arsenal"],
            "home_goals": [2, 1],
            "away_goals": [0, 1],
            "full_time_result": ["H", "D"],
            "home_points": [3, 1],
            "away_points": [0, 1],
        }
    )


def test_match_features_have_one_row_per_match() -> None:
    """Long team features should join into one row per match."""
    matches = create_matches()
    team_features = build_team_form_features(matches)

    result = build_match_form_features(
        matches,
        team_features,
    )

    assert result.height == matches.height
    assert result["match_key"].n_unique() == matches.height


def test_home_and_away_features_join_correctly() -> None:
    """Team features should join to the correct fixture side."""
    matches = create_matches()
    team_features = build_team_form_features(matches)

    result = build_match_form_features(
        matches,
        team_features,
    )

    second_match = result.filter(pl.col("match_key") == "m2")

    assert second_match["home_points_last_5"].item() == 0
    assert second_match["away_points_last_5"].item() == 3
    assert second_match["form_points_difference"].item() == -3


def test_match_features_pass_validation() -> None:
    """A correctly joined feature table should validate."""
    matches = create_matches()
    team_features = build_team_form_features(matches)

    result = build_match_form_features(
        matches,
        team_features,
    )

    validate_match_form_features(result)
