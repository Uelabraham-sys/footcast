"""Tests for chronological football Elo ratings."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from footcast.features.elo import (
    EloParameters,
    actual_home_score,
    build_elo_history,
    expected_home_score,
    update_elo_ratings,
)


def create_matches() -> pl.DataFrame:
    """Create a small chronological Silver match dataset."""
    start = datetime(
        2024,
        8,
        1,
        15,
        0,
        tzinfo=UTC,
    )

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
            "status": [
                "FINISHED",
                "FINISHED",
                "TIMED",
            ],
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
            "home_goals": [2, 1, None],
            "away_goals": [0, 1, None],
            "full_time_result": [
                "H",
                "D",
                None,
            ],
        }
    )


def test_equal_teams_give_home_advantage() -> None:
    """Equal teams should favour the home side slightly."""
    probability = expected_home_score(
        home_rating=1500.0,
        away_rating=1500.0,
        home_advantage=60.0,
    )

    assert probability > 0.5
    assert probability < 1.0


def test_expected_scores_are_complementary() -> None:
    """Home and away expectations should sum to one."""
    home_expected = expected_home_score(
        home_rating=1600.0,
        away_rating=1500.0,
        home_advantage=60.0,
    )

    assert home_expected + (1.0 - home_expected) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("home_goals", "away_goals", "expected"),
    [
        (2, 0, 1.0),
        (1, 1, 0.5),
        (0, 3, 0.0),
    ],
)
def test_actual_home_score(
    home_goals: int,
    away_goals: int,
    expected: float,
) -> None:
    """Completed scores should map to Elo outcomes."""
    assert actual_home_score(home_goals, away_goals) == expected


def test_winner_gains_and_loser_loses_rating() -> None:
    """A home victory should increase the home rating."""
    home_expected = expected_home_score(
        home_rating=1500.0,
        away_rating=1500.0,
        home_advantage=60.0,
    )

    home_post, away_post = update_elo_ratings(
        home_rating=1500.0,
        away_rating=1500.0,
        actual_home=1.0,
        expected_home=home_expected,
        k_factor=20.0,
    )

    assert home_post > 1500.0
    assert away_post < 1500.0
    assert home_post + away_post == pytest.approx(3000.0)


def test_first_match_uses_initial_ratings() -> None:
    """The first fixture must use pre-update initial ratings."""
    result = build_elo_history(create_matches())

    first_match = result.filter(pl.col("match_key") == "m1")

    assert first_match["home_elo_pre"].item() == 1500.0
    assert first_match["away_elo_pre"].item() == 1500.0
    assert first_match["home_elo_post"].item() > 1500.0
    assert first_match["away_elo_post"].item() < 1500.0


def test_second_match_uses_previous_post_ratings() -> None:
    """Later fixtures should use ratings after earlier matches."""
    result = build_elo_history(create_matches())

    first_match = result.filter(pl.col("match_key") == "m1")
    second_match = result.filter(pl.col("match_key") == "m2")

    assert second_match["away_elo_pre"].item() == pytest.approx(
        first_match["home_elo_post"].item()
    )
    assert second_match["home_elo_pre"].item() == pytest.approx(
        first_match["away_elo_post"].item()
    )


def test_future_fixture_does_not_update_ratings() -> None:
    """Unfinished fixtures must not alter team ratings."""
    result = build_elo_history(create_matches())

    future_match = result.filter(pl.col("match_key") == "m3")

    assert future_match["actual_home_score"].item() is None
    assert future_match["home_elo_post"].item() == pytest.approx(
        future_match["home_elo_pre"].item()
    )
    assert future_match["away_elo_post"].item() == pytest.approx(
        future_match["away_elo_pre"].item()
    )


def test_invalid_parameters_fail() -> None:
    """Invalid Elo parameters should raise explicit errors."""
    with pytest.raises(
        ValueError,
        match="k_factor",
    ):
        build_elo_history(
            create_matches(),
            parameters=EloParameters(k_factor=0.0),
        )
