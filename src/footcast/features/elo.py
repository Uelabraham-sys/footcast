"""Chronological Elo ratings for football teams."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final

import polars as pl

DEFAULT_INITIAL_RATING: Final[float] = 1500.0
DEFAULT_K_FACTOR: Final[float] = 20.0
DEFAULT_HOME_ADVANTAGE: Final[float] = 60.0


@dataclass(frozen=True)
class EloParameters:
    """Parameters controlling Elo calculations."""

    initial_rating: float = DEFAULT_INITIAL_RATING
    k_factor: float = DEFAULT_K_FACTOR
    home_advantage: float = DEFAULT_HOME_ADVANTAGE


def expected_home_score(
    home_rating: float,
    away_rating: float,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> float:
    """Return the expected home-team score between zero and one."""
    adjusted_difference = home_rating + home_advantage - away_rating
    result = 1.0 / (1.0 + 10.0 ** (-adjusted_difference / 400.0))

    return float(result)


def expected_away_score(home_expected_score: float) -> float:
    """Return the complementary expected away-team score."""
    return 1.0 - home_expected_score


def actual_home_score(
    home_goals: int,
    away_goals: int,
) -> float:
    """Convert a final score to the home team's Elo result."""
    if home_goals > away_goals:
        return 1.0

    if home_goals < away_goals:
        return 0.0

    return 0.5


def update_elo_ratings(
    home_rating: float,
    away_rating: float,
    actual_home: float,
    expected_home: float,
    k_factor: float = DEFAULT_K_FACTOR,
) -> tuple[float, float]:
    """Update home and away Elo ratings after a completed match."""
    rating_change = k_factor * (actual_home - expected_home)

    return (
        home_rating + rating_change,
        away_rating - rating_change,
    )


def validate_elo_parameters(
    parameters: EloParameters,
) -> None:
    """Validate Elo parameter values."""
    if parameters.initial_rating <= 0:
        raise ValueError("initial_rating must be greater than zero.")

    if parameters.k_factor <= 0:
        raise ValueError("k_factor must be greater than zero.")

    if parameters.home_advantage < 0:
        raise ValueError("home_advantage cannot be negative.")


def build_elo_history(
    matches: pl.DataFrame,
    parameters: EloParameters | None = None,
) -> pl.DataFrame:
    """Build chronological pre-match and post-match Elo records."""
    elo_parameters = parameters or EloParameters()
    validate_elo_parameters(elo_parameters)

    required_columns = {
        "match_key",
        "season",
        "competition",
        "kickoff_utc",
        "match_date",
        "status",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "home_goals",
        "away_goals",
        "full_time_result",
    }

    missing = sorted(required_columns - set(matches.columns))

    if missing:
        raise ValueError(f"Missing required Elo columns: {missing}")

    ratings: defaultdict[str, float] = defaultdict(
        lambda: elo_parameters.initial_rating
    )

    records: list[dict[str, object]] = []

    chronological_matches = matches.sort(["kickoff_utc", "match_key"])

    for match in chronological_matches.iter_rows(named=True):
        home_team_id = str(match["home_team_id"])
        away_team_id = str(match["away_team_id"])

        home_pre = ratings[home_team_id]
        away_pre = ratings[away_team_id]

        home_expected = expected_home_score(
            home_rating=home_pre,
            away_rating=away_pre,
            home_advantage=elo_parameters.home_advantage,
        )
        away_expected = expected_away_score(home_expected)

        home_goals = match["home_goals"]
        away_goals = match["away_goals"]
        status = str(match["status"])

        home_post = home_pre
        away_post = away_pre
        actual_home: float | None = None

        if (
            status == "FINISHED"
            and isinstance(home_goals, int)
            and isinstance(away_goals, int)
        ):
            actual_home = actual_home_score(
                home_goals=home_goals,
                away_goals=away_goals,
            )

            home_post, away_post = update_elo_ratings(
                home_rating=home_pre,
                away_rating=away_pre,
                actual_home=actual_home,
                expected_home=home_expected,
                k_factor=elo_parameters.k_factor,
            )

            ratings[home_team_id] = home_post
            ratings[away_team_id] = away_post

        records.append(
            {
                "match_key": match["match_key"],
                "season": match["season"],
                "competition": match["competition"],
                "kickoff_utc": match["kickoff_utc"],
                "match_date": match["match_date"],
                "status": status,
                "home_team_id": home_team_id,
                "home_team": match["home_team"],
                "away_team_id": away_team_id,
                "away_team": match["away_team"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "full_time_result": match["full_time_result"],
                "home_elo_pre": home_pre,
                "away_elo_pre": away_pre,
                "elo_difference": home_pre - away_pre,
                "home_expected_score": home_expected,
                "away_expected_score": away_expected,
                "actual_home_score": actual_home,
                "home_elo_post": home_post,
                "away_elo_post": away_post,
                "elo_change": home_post - home_pre,
            }
        )

    if not records:
        return pl.DataFrame()

    return (
        pl.DataFrame(records)
        .with_columns(
            pl.col("home_elo_pre").cast(pl.Float64),
            pl.col("away_elo_pre").cast(pl.Float64),
            pl.col("elo_difference").cast(pl.Float64),
            pl.col("home_expected_score").cast(pl.Float64),
            pl.col("away_expected_score").cast(pl.Float64),
            pl.col("actual_home_score").cast(
                pl.Float64,
                strict=False,
            ),
            pl.col("home_elo_post").cast(pl.Float64),
            pl.col("away_elo_post").cast(pl.Float64),
            pl.col("elo_change").cast(pl.Float64),
        )
        .sort(["kickoff_utc", "match_key"])
    )
