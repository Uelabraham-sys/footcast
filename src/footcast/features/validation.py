"""Validation rules for FootCast Gold feature datasets."""

import polars as pl


class FeatureValidationError(ValueError):
    """Raised when a feature dataset fails validation."""


REQUIRED_MATCH_FEATURE_COLUMNS: tuple[str, ...] = (
    "match_key",
    "season",
    "kickoff_utc",
    "home_team_id",
    "away_team_id",
    "home_matches_played_before",
    "away_matches_played_before",
    "home_points_last_5",
    "away_points_last_5",
    "home_goals_for_last_5",
    "away_goals_for_last_5",
    "home_goals_against_last_5",
    "away_goals_against_last_5",
    "home_wins_last_5",
    "away_wins_last_5",
    "home_draws_last_5",
    "away_draws_last_5",
    "home_losses_last_5",
    "away_losses_last_5",
    "home_days_since_previous_match",
    "away_days_since_previous_match",
)


def validate_team_match_history(dataframe: pl.DataFrame) -> None:
    """Validate the long-form team-match history dataset."""
    if dataframe.is_empty():
        raise FeatureValidationError("Team-match history dataset is empty.")

    required = {
        "match_key",
        "kickoff_utc",
        "team_id",
        "opponent_id",
        "venue",
        "goals_for",
        "goals_against",
        "points",
        "result",
    }

    missing = sorted(required - set(dataframe.columns))

    if missing:
        raise FeatureValidationError(f"Missing team-history columns: {missing}")

    invalid_venues = dataframe.filter(~pl.col("venue").is_in(["HOME", "AWAY"]))
    if invalid_venues.height:
        raise FeatureValidationError(
            f"Found {invalid_venues.height} invalid venue values."
        )

    duplicate_team_matches = (
        dataframe.group_by(["match_key", "team_id"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    if duplicate_team_matches:
        raise FeatureValidationError("Duplicate team-match history records were found.")


def validate_match_form_features(dataframe: pl.DataFrame) -> None:
    """Validate the wide match-level form feature dataset."""
    if dataframe.is_empty():
        raise FeatureValidationError("Match form feature dataset is empty.")

    missing = sorted(set(REQUIRED_MATCH_FEATURE_COLUMNS) - set(dataframe.columns))

    if missing:
        raise FeatureValidationError(f"Missing match feature columns: {missing}")

    duplicate_matches = (
        dataframe.group_by("match_key").len().filter(pl.col("len") > 1).height
    )

    if duplicate_matches:
        raise FeatureValidationError(
            f"Found {duplicate_matches} duplicate match feature rows."
        )

    non_negative_columns = (
        "home_matches_played_before",
        "away_matches_played_before",
        "home_points_last_5",
        "away_points_last_5",
        "home_goals_for_last_5",
        "away_goals_for_last_5",
        "home_goals_against_last_5",
        "away_goals_against_last_5",
        "home_wins_last_5",
        "away_wins_last_5",
        "home_draws_last_5",
        "away_draws_last_5",
        "home_losses_last_5",
        "away_losses_last_5",
    )

    for column in non_negative_columns:
        invalid = dataframe.filter(pl.col(column) < 0)

        if invalid.height:
            raise FeatureValidationError(f"Feature {column} contains negative values.")

    invalid_points = dataframe.filter(
        (pl.col("home_points_last_5") > 15) | (pl.col("away_points_last_5") > 15)
    )

    if invalid_points.height:
        raise FeatureValidationError("Rolling five-match points cannot exceed 15.")

    invalid_result_counts = dataframe.filter(
        (
            pl.col("home_wins_last_5")
            + pl.col("home_draws_last_5")
            + pl.col("home_losses_last_5")
        )
        > 5
    ).vstack(
        dataframe.filter(
            (
                pl.col("away_wins_last_5")
                + pl.col("away_draws_last_5")
                + pl.col("away_losses_last_5")
            )
            > 5
        )
    )

    if invalid_result_counts.height:
        raise FeatureValidationError(
            "Rolling result counts cannot exceed five matches."
        )
