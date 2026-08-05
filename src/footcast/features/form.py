"""Leakage-safe football form feature engineering."""

from __future__ import annotations

import polars as pl

ROLLING_WINDOW = 5


def build_team_match_history(
    matches: pl.DataFrame,
) -> pl.DataFrame:
    """Convert Silver matches into one row per team and match."""
    completed = matches.filter(
        (pl.col("status") == "FINISHED")
        & pl.col("home_goals").is_not_null()
        & pl.col("away_goals").is_not_null()
    )

    home_rows = completed.select(
        "match_key",
        "season",
        "competition",
        "kickoff_utc",
        "match_date",
        pl.col("home_team_id").alias("team_id"),
        pl.col("home_team").alias("team_name"),
        pl.col("away_team_id").alias("opponent_id"),
        pl.col("away_team").alias("opponent_name"),
        pl.lit("HOME").alias("venue"),
        pl.col("home_goals").alias("goals_for"),
        pl.col("away_goals").alias("goals_against"),
        pl.col("home_points").alias("points"),
        pl.col("full_time_result")
        .replace(
            {
                "H": "W",
                "D": "D",
                "A": "L",
            }
        )
        .alias("result"),
    )

    away_rows = completed.select(
        "match_key",
        "season",
        "competition",
        "kickoff_utc",
        "match_date",
        pl.col("away_team_id").alias("team_id"),
        pl.col("away_team").alias("team_name"),
        pl.col("home_team_id").alias("opponent_id"),
        pl.col("home_team").alias("opponent_name"),
        pl.lit("AWAY").alias("venue"),
        pl.col("away_goals").alias("goals_for"),
        pl.col("home_goals").alias("goals_against"),
        pl.col("away_points").alias("points"),
        pl.col("full_time_result")
        .replace(
            {
                "A": "W",
                "D": "D",
                "H": "L",
            }
        )
        .alias("result"),
    )

    return (
        pl.concat([home_rows, away_rows])
        .with_columns(
            (pl.col("goals_for") - pl.col("goals_against")).alias("goal_difference"),
            (pl.col("result") == "W").cast(pl.Int64).alias("is_win"),
            (pl.col("result") == "D").cast(pl.Int64).alias("is_draw"),
            (pl.col("result") == "L").cast(pl.Int64).alias("is_loss"),
        )
        .sort(["team_id", "kickoff_utc", "match_key"])
    )


def add_previous_match_date(
    team_history: pl.DataFrame,
) -> pl.DataFrame:
    """Add the previous completed match date for each team."""
    return team_history.with_columns(
        pl.col("kickoff_utc").shift(1).over("team_id").alias("previous_match_utc")
    ).with_columns(
        (pl.col("kickoff_utc") - pl.col("previous_match_utc"))
        .dt.total_days()
        .cast(pl.Int64)
        .alias("days_since_previous_match")
    )


def add_rolling_form_features(
    team_history: pl.DataFrame,
    window_size: int = ROLLING_WINDOW,
) -> pl.DataFrame:
    """Calculate form features using only previous matches."""
    if window_size < 1:
        raise ValueError("window_size must be at least one.")

    sorted_history = team_history.sort(["team_id", "kickoff_utc", "match_key"])

    return sorted_history.with_columns(
        pl.int_range(0, pl.len())
        .over("team_id")
        .cast(pl.Int64)
        .alias("matches_played_before"),
        pl.col("points")
        .shift(1)
        .rolling_sum(
            window_size=window_size,
            min_samples=1,
        )
        .over("team_id")
        .fill_null(0)
        .cast(pl.Int64)
        .alias("points_last_5"),
        pl.col("goals_for")
        .shift(1)
        .rolling_sum(
            window_size=window_size,
            min_samples=1,
        )
        .over("team_id")
        .fill_null(0)
        .cast(pl.Int64)
        .alias("goals_for_last_5"),
        pl.col("goals_against")
        .shift(1)
        .rolling_sum(
            window_size=window_size,
            min_samples=1,
        )
        .over("team_id")
        .fill_null(0)
        .cast(pl.Int64)
        .alias("goals_against_last_5"),
        pl.col("goal_difference")
        .shift(1)
        .rolling_sum(
            window_size=window_size,
            min_samples=1,
        )
        .over("team_id")
        .fill_null(0)
        .cast(pl.Int64)
        .alias("goal_difference_last_5"),
        pl.col("is_win")
        .shift(1)
        .rolling_sum(
            window_size=window_size,
            min_samples=1,
        )
        .over("team_id")
        .fill_null(0)
        .cast(pl.Int64)
        .alias("wins_last_5"),
        pl.col("is_draw")
        .shift(1)
        .rolling_sum(
            window_size=window_size,
            min_samples=1,
        )
        .over("team_id")
        .fill_null(0)
        .cast(pl.Int64)
        .alias("draws_last_5"),
        pl.col("is_loss")
        .shift(1)
        .rolling_sum(
            window_size=window_size,
            min_samples=1,
        )
        .over("team_id")
        .fill_null(0)
        .cast(pl.Int64)
        .alias("losses_last_5"),
    ).with_columns(
        (
            pl.col("points_last_5")
            / pl.min_horizontal(
                pl.col("matches_played_before"),
                pl.lit(window_size),
            ).replace(0, None)
        )
        .fill_null(0.0)
        .alias("average_points_last_5")
    )


def build_team_form_features(
    matches: pl.DataFrame,
    window_size: int = ROLLING_WINDOW,
) -> pl.DataFrame:
    """Build long-form, leakage-safe team form features."""
    history = build_team_match_history(matches)
    history = add_previous_match_date(history)

    return add_rolling_form_features(
        history,
        window_size=window_size,
    )
