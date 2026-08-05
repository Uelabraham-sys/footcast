"""Tests for modelling-dataset preparation."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from footcast.modelling.dataset import (
    build_model_datasets,
    build_model_split,
    prepare_feature_frame,
)

FEATURE_COLUMNS = (
    "feature_one",
    "feature_two",
)


def create_dataset() -> pl.DataFrame:
    """Create chronological train, validation and test data."""
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
            "full_time_result": ["H", "D", "A"],
            "feature_one": [1.0, 2.0, 3.0],
            "feature_two": [None, 5.0, 6.0],
            "target": [2, 1, 0],
            "split": [
                "train",
                "validation",
                "test",
            ],
        }
    )


def test_feature_frame_fills_missing_values() -> None:
    """Missing model features should receive neutral zeroes."""
    result = prepare_feature_frame(
        create_dataset(),
        feature_columns=FEATURE_COLUMNS,
    )

    assert result["feature_two"].to_list() == [
        0.0,
        5.0,
        6.0,
    ]


def test_model_split_returns_numeric_arrays() -> None:
    """A model split should return numeric features and targets."""
    result = build_model_split(
        create_dataset(),
        split_name="train",
        feature_columns=FEATURE_COLUMNS,
    )

    assert result.features.shape == (1, 2)
    assert result.target.tolist() == [2]
    assert result.features.dtype == np.float64
    assert result.target.dtype == np.int64


def test_model_datasets_are_chronological() -> None:
    """Each named split should be constructed independently."""
    result = build_model_datasets(
        create_dataset(),
        feature_columns=FEATURE_COLUMNS,
    )

    assert result.train.metadata["match_key"].to_list() == ["m1"]
    assert result.validation.metadata["match_key"].to_list() == ["m2"]
    assert result.test.metadata["match_key"].to_list() == ["m3"]


def test_missing_feature_fails() -> None:
    """Missing feature columns should raise an explicit error."""
    with pytest.raises(
        ValueError,
        match="Missing modelling feature columns",
    ):
        prepare_feature_frame(
            create_dataset(),
            feature_columns=("unknown_feature",),
        )


def test_empty_split_fails() -> None:
    """An absent chronological split should fail."""
    with pytest.raises(
        ValueError,
        match="is empty",
    ):
        build_model_split(
            create_dataset(),
            split_name="future",
            feature_columns=FEATURE_COLUMNS,
        )
