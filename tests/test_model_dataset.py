"""Tests for the final FootCast modelling dataset."""

from datetime import UTC, datetime, timedelta

import polars as pl

from footcast.features.model_dataset import (
    add_cold_start_features,
    add_target_column,
    assign_chronological_splits,
    join_form_and_elo_features,
)


def create_form_features() -> pl.DataFrame:
    """Create representative rolling-form rows."""
    start = datetime(2022, 8, 1, tzinfo=UTC)

    return pl.DataFrame(
        {
            "match_key": ["m1", "m2", "m3"],
            "season": [
                "2022/23",
                "2023/24",
                "2024/25",
            ],
            "competition": ["Premier League"] * 3,
            "kickoff_utc": [
                start,
                start + timedelta(days=365),
                start + timedelta(days=730),
            ],
            "match_date": [
                start.date(),
                (start + timedelta(days=365)).date(),
                (start + timedelta(days=730)).date(),
            ],
            "status": ["FINISHED"] * 3,
            "home_team_id": ["arsenal"] * 3,
            "home_team": ["Arsenal"] * 3,
            "away_team_id": ["chelsea"] * 3,
            "away_team": ["Chelsea"] * 3,
            "home_goals": [2, 1, 0],
            "away_goals": [0, 1, 1],
            "full_time_result": ["H", "D", "A"],
            "home_points": [3, 1, 0],
            "away_points": [0, 1, 3],
            "home_matches_played_before": [0, 1, 2],
            "away_matches_played_before": [0, 1, 2],
            "home_points_last_5": [0, 3, 4],
            "away_points_last_5": [0, 0, 1],
            "home_expected_score": [0.5, 0.5, 0.5],
            "away_expected_score": [0.5, 0.5, 0.5],
        }
    )


def create_elo_features() -> pl.DataFrame:
    """Create representative Elo feature rows."""
    return pl.DataFrame(
        {
            "match_key": ["m1", "m2", "m3"],
            "home_elo_pre": [1500.0, 1510.0, 1515.0],
            "away_elo_pre": [1500.0, 1490.0, 1485.0],
            "elo_difference": [0.0, 20.0, 30.0],
            "home_expected_score": [0.58, 0.60, 0.62],
            "away_expected_score": [0.42, 0.40, 0.38],
        }
    )


def test_form_and_elo_join_one_to_one() -> None:
    """Form and Elo records should join on match key."""
    result = join_form_and_elo_features(
        form_features=create_form_features().drop(
            "home_expected_score",
            "away_expected_score",
        ),
        elo_features=create_elo_features(),
    )

    assert result.height == 3
    assert result["home_elo_pre"].to_list() == [
        1500.0,
        1510.0,
        1515.0,
    ]


def test_target_encoding() -> None:
    """Match outcomes should encode as away, draw and home."""
    result = add_target_column(create_form_features())

    assert result["target"].to_list() == [2, 1, 0]


def test_cold_start_flags() -> None:
    """Cold-start indicators should use prior match counts."""
    result = add_cold_start_features(create_form_features())

    assert result["home_is_cold_start"].to_list() == [
        True,
        False,
        False,
    ]
    assert result["away_has_limited_history"].to_list() == [
        True,
        True,
        True,
    ]


def test_chronological_split_assignment() -> None:
    """Complete seasons should map to ordered splits."""
    result = assign_chronological_splits(
        create_form_features(),
        validation_season="2023/24",
        test_season="2024/25",
    )

    assert result["split"].to_list() == [
        "train",
        "validation",
        "test",
    ]
