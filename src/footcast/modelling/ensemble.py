"""Probability-ensemble utilities for FootCast."""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from footcast.modelling.calibration_diagnostics import (
    PROBABILITY_COLUMNS,
)

DEFAULT_WEIGHTS: Final[tuple[float, ...]] = tuple(
    round(index / 20, 2) for index in range(21)
)


def validate_weight(
    weight: float,
) -> None:
    """Validate a convex ensemble weight."""
    if not np.isfinite(weight):
        raise ValueError("Ensemble weight must be finite.")

    if not 0.0 <= weight <= 1.0:
        raise ValueError("Ensemble weight must be between zero and one.")


def validate_probability_array(
    probabilities: NDArray[np.float64],
    *,
    name: str,
) -> None:
    """Validate a three-class probability matrix."""
    if probabilities.ndim != 2:
        raise ValueError(f"{name} probabilities must be two-dimensional.")

    if probabilities.shape[1] != 3:
        raise ValueError(f"{name} probabilities must have three columns.")

    if probabilities.shape[0] == 0:
        raise ValueError(f"{name} probabilities cannot be empty.")

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{name} probabilities contain non-finite values.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{name} probabilities must be between zero and one.")

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(f"{name} probability rows must sum to one.")


def blend_probabilities(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    *,
    first_weight: float,
) -> NDArray[np.float64]:
    """Blend two aligned probability matrices."""
    validate_weight(first_weight)

    validate_probability_array(
        first,
        name="First model",
    )
    validate_probability_array(
        second,
        name="Second model",
    )

    if first.shape != second.shape:
        raise ValueError("Probability matrices must have identical shapes.")

    second_weight = 1.0 - first_weight

    blended = (first_weight * first + second_weight * second).astype(np.float64)

    row_sums = blended.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(row_sums <= 0.0):
        raise ValueError("Blended probability rows must have positive mass.")

    return (blended / row_sums).astype(np.float64)


def validate_prediction_frame(
    predictions: pl.DataFrame,
    *,
    frame_name: str,
) -> None:
    """Validate a model prediction frame before alignment."""
    required = {
        "match_key",
        "target",
        *PROBABILITY_COLUMNS,
    }

    missing = sorted(required - set(predictions.columns))

    if missing:
        raise ValueError(f"{frame_name} is missing columns: {missing}")

    if predictions.is_empty():
        raise ValueError(f"{frame_name} cannot be empty.")

    duplicate_count = (
        predictions.group_by("match_key").len().filter(pl.col("len") > 1).height
    )

    if duplicate_count > 0:
        raise ValueError(f"{frame_name} contains duplicate match keys.")


def align_prediction_frames(
    first: pl.DataFrame,
    second: pl.DataFrame,
    *,
    first_name: str,
    second_name: str,
) -> pl.DataFrame:
    """Align two prediction frames using match_key."""
    validate_prediction_frame(
        first,
        frame_name=first_name,
    )
    validate_prediction_frame(
        second,
        frame_name=second_name,
    )

    first_keys = set(first["match_key"].to_list())
    second_keys = set(second["match_key"].to_list())

    if first_keys != second_keys:
        missing_from_first = sorted(second_keys - first_keys)
        missing_from_second = sorted(first_keys - second_keys)

        raise ValueError(
            "Prediction match keys do not align. "
            f"Missing from {first_name}: "
            f"{missing_from_first[:5]}; "
            f"missing from {second_name}: "
            f"{missing_from_second[:5]}."
        )

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
        )
        if column in first.columns
    ]

    first_selected = first.select(
        [
            *metadata_columns,
            pl.col("probability_away_win").alias("first_probability_away_win"),
            pl.col("probability_draw").alias("first_probability_draw"),
            pl.col("probability_home_win").alias("first_probability_home_win"),
        ]
    )

    second_selected = second.select(
        [
            "match_key",
            pl.col("target").alias("second_target"),
            pl.col("probability_away_win").alias("second_probability_away_win"),
            pl.col("probability_draw").alias("second_probability_draw"),
            pl.col("probability_home_win").alias("second_probability_home_win"),
        ]
    )

    aligned = first_selected.join(
        second_selected,
        on="match_key",
        how="inner",
        validate="1:1",
    )

    target_mismatch_count = aligned.filter(
        pl.col("target") != pl.col("second_target")
    ).height

    if target_mismatch_count > 0:
        raise ValueError("Aligned prediction frames contain target mismatches.")

    if aligned.height != first.height:
        raise ValueError("Prediction alignment changed the row count.")

    sort_columns = [
        column
        for column in (
            "kickoff_utc",
            "match_key",
        )
        if column in aligned.columns
    ]

    if sort_columns:
        aligned = aligned.sort(sort_columns)

    return aligned.drop("second_target")


def extract_aligned_arrays(
    aligned: pl.DataFrame,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Extract target and component probabilities."""
    target = np.asarray(
        aligned["target"].to_numpy(),
        dtype=np.int64,
    )

    first = np.asarray(
        aligned.select(
            [
                "first_probability_away_win",
                "first_probability_draw",
                "first_probability_home_win",
            ]
        ).to_numpy(),
        dtype=np.float64,
    )

    second = np.asarray(
        aligned.select(
            [
                "second_probability_away_win",
                "second_probability_draw",
                "second_probability_home_win",
            ]
        ).to_numpy(),
        dtype=np.float64,
    )

    validate_probability_array(
        first,
        name="First model",
    )
    validate_probability_array(
        second,
        name="Second model",
    )

    return target, first, second


def create_ensemble_prediction_frame(
    aligned: pl.DataFrame,
    probabilities: NDArray[np.float64],
    *,
    first_weight: float,
    first_model: str,
    second_model: str,
) -> pl.DataFrame:
    """Create match-level ensemble predictions."""
    validate_probability_array(
        probabilities,
        name="Ensemble",
    )

    if probabilities.shape[0] != aligned.height:
        raise ValueError("Ensemble probability rows do not match metadata.")

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
        )
        if column in aligned.columns
    ]

    return (
        aligned.select(metadata_columns)
        .with_columns(
            pl.Series(
                "probability_away_win",
                probabilities[:, 0],
            ),
            pl.Series(
                "probability_draw",
                probabilities[:, 1],
            ),
            pl.Series(
                "probability_home_win",
                probabilities[:, 2],
            ),
        )
        .with_columns(
            pl.concat_list(
                [
                    "probability_away_win",
                    "probability_draw",
                    "probability_home_win",
                ]
            )
            .list.arg_max()
            .cast(pl.Int64)
            .alias("predicted_class"),
            pl.lit(first_weight).alias("first_model_weight"),
            pl.lit(1.0 - first_weight).alias("second_model_weight"),
            pl.lit(first_model).alias("first_model"),
            pl.lit(second_model).alias("second_model"),
        )
    )
