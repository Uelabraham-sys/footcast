"""Validation rules for canonical Silver match data."""

import polars as pl


class SilverDataValidationError(ValueError):
    """Raised when Silver match data fails validation."""


REQUIRED_SILVER_COLUMNS: tuple[str, ...] = (
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
    "source",
    "ingested_at",
)


def validate_silver_matches(dataframe: pl.DataFrame) -> None:
    """Validate canonical Silver match records."""
    if dataframe.is_empty():
        raise SilverDataValidationError("Silver matches dataset is empty.")

    missing = sorted(set(REQUIRED_SILVER_COLUMNS) - set(dataframe.columns))
    if missing:
        raise SilverDataValidationError(f"Missing required Silver columns: {missing}")

    duplicate_keys = (
        dataframe.group_by("match_key").len().filter(pl.col("len") > 1).height
    )
    if duplicate_keys:
        raise SilverDataValidationError(
            f"Found {duplicate_keys} duplicate canonical match keys."
        )

    invalid_teams = dataframe.filter(
        pl.col("home_team_id").is_null()
        | pl.col("away_team_id").is_null()
        | pl.col("home_team").is_null()
        | pl.col("away_team").is_null()
        | (pl.col("home_team_id") == pl.col("away_team_id"))
    )
    if invalid_teams.height:
        raise SilverDataValidationError(
            f"Found {invalid_teams.height} matches with invalid teams."
        )

    completed = dataframe.filter(pl.col("status") == "FINISHED")
    invalid_scores = completed.filter(
        pl.col("home_goals").is_null()
        | pl.col("away_goals").is_null()
        | (pl.col("home_goals") < 0)
        | (pl.col("away_goals") < 0)
    )
    if invalid_scores.height:
        raise SilverDataValidationError(
            f"Found {invalid_scores.height} completed matches with invalid scores."
        )

    expected_result = (
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
    )

    inconsistent_results = completed.filter(
        pl.col("full_time_result") != expected_result
    )
    if inconsistent_results.height:
        raise SilverDataValidationError(
            f"Found {inconsistent_results.height} inconsistent results."
        )

    invalid_dates = dataframe.filter(
        pl.col("match_date").is_null() | pl.col("kickoff_utc").is_null()
    )
    if invalid_dates.height:
        raise SilverDataValidationError(
            f"Found {invalid_dates.height} matches with invalid dates."
        )
