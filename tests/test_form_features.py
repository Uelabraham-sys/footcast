"""Tests for leakage-safe rolling football form features."""

from datetime import UTC, datetime, timedelta

import polars as pl

from footcast.features.form import (
    add_previous_match_date,
    add_rolling_form_features,
    build_team_form_features,
    build_team_match_history,
)


def create_matches() -> pl.DataFrame:
    """Create a small chronological match dataset."""
    start = datetime(2024, 8, 1, 15, 0, tzinfo=UTC)

    return pl.DataFrame(
        {
            "match_key": ["m1", "m2", "m3"],
            "season": ["2024/25"] * 3,
            "competition": ["Premier League"] * 3,
            "kickoff_utc": [
                start,
                start + timedelta(days=7),
                start + timedelta(days=14),
            ],
            "match_date": [
                start.date(),
                (start + timedelta(days=7)).date(),
                (start + timedelta(days=14)).date(),
            ],
            "status": ["FINISHED"] * 3,
            "home_team_id": [
                "arsenal",
                "chelsea",
                "arsenal",
            ],
            "home_team": [
                "Arsenal",
                "Chelsea",
                "Arsenal",
            ],
            "away_team_id": [
                "chelsea",
                "arsenal",
                "chelsea",
            ],
            "away_team": [
                "Chelsea",
                "Arsenal",
                "Chelsea",
            ],
            "home_goals": [2, 1, 0],
            "away_goals": [0, 1, 3],
            "full_time_result": ["H", "D", "A"],
            "home_points": [3, 1, 0],
            "away_points": [0, 1, 3],
        }
    )


def test_team_history_has_two_rows_per_match() -> None:
    """Every completed match should produce two team rows."""
    result = build_team_match_history(create_matches())

    assert result.height == 6
    assert result.group_by("match_key").len().select(pl.col("len").unique()).item() == 2


def test_home_and_away_perspectives_are_correct() -> None:
    """Team statistics should be expressed from each team's view."""
    result = build_team_match_history(create_matches())

    arsenal_first = result.filter(
        (pl.col("match_key") == "m1") & (pl.col("team_id") == "arsenal")
    )
    chelsea_first = result.filter(
        (pl.col("match_key") == "m1") & (pl.col("team_id") == "chelsea")
    )

    assert arsenal_first["goals_for"].item() == 2
    assert arsenal_first["goals_against"].item() == 0
    assert arsenal_first["points"].item() == 3
    assert arsenal_first["result"].item() == "W"

    assert chelsea_first["goals_for"].item() == 0
    assert chelsea_first["goals_against"].item() == 2
    assert chelsea_first["points"].item() == 0
    assert chelsea_first["result"].item() == "L"


def test_previous_match_date_is_team_specific() -> None:
    """Rest days should use the same team's previous match."""
    history = build_team_match_history(create_matches())
    result = add_previous_match_date(history)

    arsenal = result.filter(pl.col("team_id") == "arsenal").sort("kickoff_utc")

    assert arsenal["days_since_previous_match"].to_list() == [
        None,
        7,
        7,
    ]


def test_first_match_has_zero_previous_form() -> None:
    """A team's first match should have neutral form features."""
    result = build_team_form_features(create_matches())

    arsenal_first = result.filter(
        (pl.col("team_id") == "arsenal") & (pl.col("match_key") == "m1")
    )

    assert arsenal_first["matches_played_before"].item() == 0
    assert arsenal_first["points_last_5"].item() == 0
    assert arsenal_first["goals_for_last_5"].item() == 0
    assert arsenal_first["wins_last_5"].item() == 0


def test_current_result_is_excluded_from_features() -> None:
    """A match must not contribute to its own pre-match features."""
    result = build_team_form_features(create_matches())

    arsenal_second = result.filter(
        (pl.col("team_id") == "arsenal") & (pl.col("match_key") == "m2")
    )

    assert arsenal_second["matches_played_before"].item() == 1
    assert arsenal_second["points_last_5"].item() == 3
    assert arsenal_second["goals_for_last_5"].item() == 2
    assert arsenal_second["goals_against_last_5"].item() == 0
    assert arsenal_second["wins_last_5"].item() == 1

    assert arsenal_second["points"].item() == 1


def test_third_match_uses_first_two_results_only() -> None:
    """The third fixture should aggregate only the first two games."""
    result = build_team_form_features(create_matches())

    arsenal_third = result.filter(
        (pl.col("team_id") == "arsenal") & (pl.col("match_key") == "m3")
    )

    assert arsenal_third["matches_played_before"].item() == 2
    assert arsenal_third["points_last_5"].item() == 4
    assert arsenal_third["goals_for_last_5"].item() == 3
    assert arsenal_third["goals_against_last_5"].item() == 1
    assert arsenal_third["wins_last_5"].item() == 1
    assert arsenal_third["draws_last_5"].item() == 1
    assert arsenal_third["losses_last_5"].item() == 0


def test_custom_window_size() -> None:
    """Rolling calculations should support smaller windows."""
    history = build_team_match_history(create_matches())
    history = add_previous_match_date(history)

    result = add_rolling_form_features(
        history,
        window_size=1,
    )

    arsenal_third = result.filter(
        (pl.col("team_id") == "arsenal") & (pl.col("match_key") == "m3")
    )

    assert arsenal_third["points_last_5"].item() == 1
