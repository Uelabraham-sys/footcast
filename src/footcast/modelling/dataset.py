"""Load and prepare FootCast modelling datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

TARGET_COLUMN: Final[str] = "target"
SPLIT_COLUMN: Final[str] = "split"

CLASS_LABELS: Final[tuple[int, int, int]] = (0, 1, 2)
CLASS_NAMES: Final[tuple[str, str, str]] = (
    "away_win",
    "draw",
    "home_win",
)

MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "home_matches_played_before",
    "away_matches_played_before",
    "home_is_cold_start",
    "away_is_cold_start",
    "home_has_limited_history",
    "away_has_limited_history",
    "home_points_last_5",
    "away_points_last_5",
    "form_points_difference",
    "home_goals_for_last_5",
    "away_goals_for_last_5",
    "recent_attack_difference",
    "home_goals_against_last_5",
    "away_goals_against_last_5",
    "recent_defence_difference",
    "home_goal_difference_last_5",
    "away_goal_difference_last_5",
    "recent_goal_difference_difference",
    "home_wins_last_5",
    "away_wins_last_5",
    "home_draws_last_5",
    "away_draws_last_5",
    "home_losses_last_5",
    "away_losses_last_5",
    "home_average_points_last_5",
    "away_average_points_last_5",
    "average_form_difference",
    "home_days_since_previous_match",
    "away_days_since_previous_match",
    "rest_days_difference",
    "home_elo_pre",
    "away_elo_pre",
    "elo_difference",
    "home_expected_score",
    "away_expected_score",
)


@dataclass(frozen=True)
class ModelSplit:
    """Feature matrix, target and match metadata for one split."""

    features: NDArray[np.float64]
    target: NDArray[np.int64]
    metadata: pl.DataFrame


@dataclass(frozen=True)
class ModelDatasets:
    """Chronological train, validation and test datasets."""

    train: ModelSplit
    validation: ModelSplit
    test: ModelSplit
    feature_names: tuple[str, ...]


def load_model_dataset(
    path: Path,
) -> pl.DataFrame:
    """Load the final Gold modelling dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model dataset was not found: {path}. "
            "Run `make build-model-dataset` first."
        )

    dataframe = pl.read_parquet(path)

    if dataframe.is_empty():
        raise ValueError("Model dataset is empty.")

    return dataframe


def validate_feature_columns(
    dataframe: pl.DataFrame,
    feature_columns: tuple[str, ...],
) -> None:
    """Ensure all requested modelling features are present."""
    missing = sorted(set(feature_columns) - set(dataframe.columns))

    if missing:
        raise ValueError(f"Missing modelling feature columns: {missing}")


def prepare_feature_frame(
    dataframe: pl.DataFrame,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> pl.DataFrame:
    """Select features and convert them to finite floats."""
    validate_feature_columns(dataframe, feature_columns)

    return dataframe.select(
        [
            pl.col(column).cast(pl.Float64).fill_nan(None).fill_null(0.0).alias(column)
            for column in feature_columns
        ]
    )


def build_model_split(
    dataframe: pl.DataFrame,
    split_name: str,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> ModelSplit:
    """Construct one labelled chronological model split."""
    split = dataframe.filter(
        (pl.col(SPLIT_COLUMN) == split_name) & pl.col(TARGET_COLUMN).is_not_null()
    ).sort(["kickoff_utc", "match_key"])

    if split.is_empty():
        raise ValueError(f"Model split {split_name!r} is empty.")

    feature_frame = prepare_feature_frame(
        split,
        feature_columns=feature_columns,
    )

    feature_array = np.asarray(
        feature_frame.to_numpy(),
        dtype=np.float64,
    )

    target_array = np.asarray(
        split[TARGET_COLUMN].to_numpy(),
        dtype=np.int64,
    )

    if not np.isfinite(feature_array).all():
        raise ValueError(f"Split {split_name!r} contains non-finite features.")

    metadata_columns = [
        column
        for column in (
            "match_key",
            "season",
            "kickoff_utc",
            "match_date",
            "home_team_id",
            "home_team",
            "away_team_id",
            "away_team",
            "full_time_result",
            "target",
            "split",
        )
        if column in split.columns
    ]

    return ModelSplit(
        features=feature_array,
        target=target_array,
        metadata=split.select(metadata_columns),
    )


def build_model_datasets(
    dataframe: pl.DataFrame,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> ModelDatasets:
    """Create chronological train, validation and test arrays."""
    train = build_model_split(
        dataframe,
        split_name="train",
        feature_columns=feature_columns,
    )
    validation = build_model_split(
        dataframe,
        split_name="validation",
        feature_columns=feature_columns,
    )
    test = build_model_split(
        dataframe,
        split_name="test",
        feature_columns=feature_columns,
    )

    return ModelDatasets(
        train=train,
        validation=validation,
        test=test,
        feature_names=feature_columns,
    )
