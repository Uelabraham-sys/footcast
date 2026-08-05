"""Validation rules for the final FootCast modelling dataset."""

from __future__ import annotations

from datetime import datetime

import polars as pl


class ModelDatasetValidationError(ValueError):
    """Raised when the modelling dataset fails validation."""


REQUIRED_MODEL_COLUMNS: tuple[str, ...] = (
    "match_key",
    "season",
    "kickoff_utc",
    "home_team_id",
    "away_team_id",
    "home_points_last_5",
    "away_points_last_5",
    "home_elo_pre",
    "away_elo_pre",
    "elo_difference",
    "target",
    "split",
)


def validate_required_columns(
    dataframe: pl.DataFrame,
) -> None:
    """Ensure that the modelling table has its required columns."""
    missing = sorted(set(REQUIRED_MODEL_COLUMNS) - set(dataframe.columns))

    if missing:
        raise ModelDatasetValidationError(f"Missing modelling columns: {missing}")


def validate_unique_matches(
    dataframe: pl.DataFrame,
) -> None:
    """Ensure that every match appears exactly once."""
    duplicate_count = (
        dataframe.group_by("match_key").len().filter(pl.col("len") > 1).height
    )

    if duplicate_count:
        raise ModelDatasetValidationError(
            f"Found {duplicate_count} duplicate match rows."
        )


def validate_targets(
    dataframe: pl.DataFrame,
) -> None:
    """Ensure that model targets use the expected classes."""
    invalid = dataframe.filter(
        pl.col("target").is_not_null() & ~pl.col("target").is_in([0, 1, 2])
    )

    if invalid.height:
        raise ModelDatasetValidationError(
            f"Found {invalid.height} invalid target values."
        )


def validate_probabilistic_features(
    dataframe: pl.DataFrame,
) -> None:
    """Validate Elo expectation values."""
    invalid = dataframe.filter(
        (pl.col("home_expected_score") < 0)
        | (pl.col("home_expected_score") > 1)
        | (pl.col("away_expected_score") < 0)
        | (pl.col("away_expected_score") > 1)
    )

    if invalid.height:
        raise ModelDatasetValidationError(
            "Elo expected scores must be between zero and one."
        )

    invalid_sum = dataframe.filter(
        (pl.col("home_expected_score") + pl.col("away_expected_score") - 1.0).abs()
        > 1e-10
    )

    if invalid_sum.height:
        raise ModelDatasetValidationError(
            "Home and away Elo expectations must sum to one."
        )


def validate_split_order(
    dataframe: pl.DataFrame,
) -> None:
    """Ensure that train, validation and test are chronological."""
    labelled = dataframe.filter(pl.col("split").is_in(["train", "validation", "test"]))

    if labelled.is_empty():
        raise ModelDatasetValidationError("No chronological split labels were found.")

    train = labelled.filter(pl.col("split") == "train")
    validation = labelled.filter(pl.col("split") == "validation")
    test = labelled.filter(pl.col("split") == "test")

    if train.is_empty():
        raise ModelDatasetValidationError("Training split is empty.")

    if validation.is_empty():
        raise ModelDatasetValidationError("Validation split is empty.")

    if test.is_empty():
        raise ModelDatasetValidationError("Test split is empty.")

    train_max_value = train.select(pl.col("kickoff_utc").max().alias("value"))[
        "value"
    ].item()

    validation_min_value = validation.select(
        pl.col("kickoff_utc").min().alias("value")
    )["value"].item()

    validation_max_value = validation.select(
        pl.col("kickoff_utc").max().alias("value")
    )["value"].item()

    test_min_value = test.select(pl.col("kickoff_utc").min().alias("value"))[
        "value"
    ].item()

    timestamp_values = {
        "training maximum": train_max_value,
        "validation minimum": validation_min_value,
        "validation maximum": validation_max_value,
        "test minimum": test_min_value,
    }

    for name, value in timestamp_values.items():
        if not isinstance(value, datetime):
            raise ModelDatasetValidationError(
                f"{name} timestamp is missing or invalid."
            )

    train_max = train_max_value
    validation_min = validation_min_value
    validation_max = validation_max_value
    test_min = test_min_value

    if train_max >= validation_min:
        raise ModelDatasetValidationError("Training data overlaps validation data.")

    if validation_max >= test_min:
        raise ModelDatasetValidationError("Validation data overlaps test data.")


def validate_cold_start_flags(
    dataframe: pl.DataFrame,
) -> None:
    """Ensure cold-start flags agree with match history counts."""
    inconsistent_home = dataframe.filter(
        pl.col("home_is_cold_start") != (pl.col("home_matches_played_before") == 0)
    )

    inconsistent_away = dataframe.filter(
        pl.col("away_is_cold_start") != (pl.col("away_matches_played_before") == 0)
    )

    if inconsistent_home.height or inconsistent_away.height:
        raise ModelDatasetValidationError(
            "Cold-start flags do not agree with prior-match counts."
        )


def validate_model_dataset(
    dataframe: pl.DataFrame,
) -> None:
    """Run all final model-dataset validation rules."""
    if dataframe.is_empty():
        raise ModelDatasetValidationError("Model dataset is empty.")

    validate_required_columns(dataframe)
    validate_unique_matches(dataframe)
    validate_targets(dataframe)
    validate_probabilistic_features(dataframe)
    validate_split_order(dataframe)
    validate_cold_start_flags(dataframe)
