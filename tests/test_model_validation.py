"""Tests for final modelling-dataset validation."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from footcast.features.model_validation import (
    ModelDatasetValidationError,
    validate_model_dataset,
)


def create_valid_dataset() -> pl.DataFrame:
    """Create a valid chronological model dataset."""
    start = datetime(2022, 8, 1, tzinfo=UTC)

    return pl.DataFrame(
        {
            "match_key": ["m1", "m2", "m3"],
            "season": [
                "2022/23",
                "2023/24",
                "2024/25",
            ],
            "kickoff_utc": [
                start,
                start + timedelta(days=365),
                start + timedelta(days=730),
            ],
            "home_team_id": ["arsenal"] * 3,
            "away_team_id": ["chelsea"] * 3,
            "home_matches_played_before": [0, 1, 2],
            "away_matches_played_before": [0, 1, 2],
            "home_is_cold_start": [True, False, False],
            "away_is_cold_start": [True, False, False],
            "home_points_last_5": [0, 3, 4],
            "away_points_last_5": [0, 0, 1],
            "home_elo_pre": [1500.0, 1510.0, 1515.0],
            "away_elo_pre": [1500.0, 1490.0, 1485.0],
            "elo_difference": [0.0, 20.0, 30.0],
            "home_expected_score": [0.58, 0.60, 0.62],
            "away_expected_score": [0.42, 0.40, 0.38],
            "target": [2, 1, 0],
            "split": ["train", "validation", "test"],
        }
    )


def test_valid_dataset_passes() -> None:
    """A valid model dataset should pass all rules."""
    validate_model_dataset(create_valid_dataset())


def test_duplicate_matches_fail() -> None:
    """Duplicate match keys should fail validation."""
    dataframe = create_valid_dataset()

    duplicated = pl.concat([dataframe, dataframe.head(1)])

    with pytest.raises(
        ModelDatasetValidationError,
        match="duplicate match rows",
    ):
        validate_model_dataset(duplicated)


def test_invalid_target_fails() -> None:
    """Targets outside zero, one and two should fail."""
    dataframe = create_valid_dataset().with_columns(
        pl.when(pl.col("match_key") == "m1")
        .then(pl.lit(7))
        .otherwise(pl.col("target"))
        .alias("target")
    )

    with pytest.raises(
        ModelDatasetValidationError,
        match="invalid target",
    ):
        validate_model_dataset(dataframe)


def test_overlapping_splits_fail() -> None:
    """Chronological overlap should fail validation."""
    dataframe = create_valid_dataset().with_columns(
        pl.when(pl.col("match_key") == "m2")
        .then(pl.lit(datetime(2022, 7, 1, tzinfo=UTC)))
        .otherwise(pl.col("kickoff_utc"))
        .alias("kickoff_utc")
    )

    with pytest.raises(
        ModelDatasetValidationError,
        match="overlaps",
    ):
        validate_model_dataset(dataframe)


def test_incorrect_cold_start_flag_fails() -> None:
    """Cold-start flags must match prior match counts."""
    dataframe = create_valid_dataset().with_columns(
        pl.lit(False).alias("home_is_cold_start")
    )

    with pytest.raises(
        ModelDatasetValidationError,
        match="Cold-start flags",
    ):
        validate_model_dataset(dataframe)
