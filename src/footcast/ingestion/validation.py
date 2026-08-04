"""Validation rules for ingested football data."""

from collections.abc import Iterable

import polars as pl


class HistoricalDataValidationError(ValueError):
    """Raised when historical match data fails validation."""


REQUIRED_HISTORICAL_COLUMNS: tuple[str, ...] = (
    "match_id",
    "season",
    "match_date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "full_time_result",
    "source",
    "ingested_at",
)


def validate_required_columns(
    dataframe: pl.DataFrame,
    required_columns: Iterable[str] = REQUIRED_HISTORICAL_COLUMNS,
) -> None:
    """Ensure that every required column exists."""
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))

    if missing_columns:
        message = f"Missing required columns: {missing_columns}"
        raise HistoricalDataValidationError(message)


def validate_non_empty(dataframe: pl.DataFrame) -> None:
    """Ensure that the historical dataset is not empty."""
    if dataframe.is_empty():
        raise HistoricalDataValidationError("Historical dataset is empty.")


def validate_unique_match_ids(dataframe: pl.DataFrame) -> None:
    """Ensure that match identifiers are unique."""
    duplicate_count = (
        dataframe.group_by("match_id").len().filter(pl.col("len") > 1).height
    )

    if duplicate_count > 0:
        message = f"Found {duplicate_count} duplicated match identifiers."
        raise HistoricalDataValidationError(message)


def validate_team_names(dataframe: pl.DataFrame) -> None:
    """Ensure that every match has two distinct non-empty team names."""
    invalid_rows = dataframe.filter(
        pl.col("home_team").is_null()
        | pl.col("away_team").is_null()
        | (pl.col("home_team").str.strip_chars() == "")
        | (pl.col("away_team").str.strip_chars() == "")
        | (pl.col("home_team") == pl.col("away_team"))
    )

    if invalid_rows.height > 0:
        message = f"Found {invalid_rows.height} rows with invalid team names."
        raise HistoricalDataValidationError(message)


def validate_goal_values(dataframe: pl.DataFrame) -> None:
    """Ensure that completed-match goal values are valid."""
    invalid_rows = dataframe.filter(
        pl.col("home_goals").is_null()
        | pl.col("away_goals").is_null()
        | (pl.col("home_goals") < 0)
        | (pl.col("away_goals") < 0)
    )

    if invalid_rows.height > 0:
        message = f"Found {invalid_rows.height} rows with invalid goal values."
        raise HistoricalDataValidationError(message)


def validate_full_time_results(dataframe: pl.DataFrame) -> None:
    """Ensure that full-time result labels agree with goal totals."""
    expected_result = (
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
    )

    inconsistent_rows = dataframe.filter(pl.col("full_time_result") != expected_result)

    if inconsistent_rows.height > 0:
        message = (
            f"Found {inconsistent_rows.height} rows whose result label "
            "does not agree with the score."
        )
        raise HistoricalDataValidationError(message)


def validate_match_dates(dataframe: pl.DataFrame) -> None:
    """Ensure that every historical match has a parsed date."""
    null_dates = dataframe.filter(pl.col("match_date").is_null()).height

    if null_dates > 0:
        message = f"Found {null_dates} rows with invalid match dates."
        raise HistoricalDataValidationError(message)


def validate_historical_matches(dataframe: pl.DataFrame) -> None:
    """Run all historical-match validation rules."""
    validate_non_empty(dataframe)
    validate_required_columns(dataframe)
    validate_unique_match_ids(dataframe)
    validate_team_names(dataframe)
    validate_goal_values(dataframe)
    validate_full_time_results(dataframe)
    validate_match_dates(dataframe)
