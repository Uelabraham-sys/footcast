"""Expanding-window backtesting utilities for FootCast."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from footcast.modelling.dataset import (
    MODEL_FEATURE_COLUMNS,
    prepare_feature_frame,
)

MINIMUM_TRAINING_SEASONS: Final[int] = 2


@dataclass(frozen=True)
class BacktestFold:
    """One chronological expanding-window backtest fold."""

    fold_number: int
    training_seasons: tuple[str, ...]
    evaluation_season: str
    train_features: NDArray[np.float64]
    train_target: NDArray[np.int64]
    evaluation_features: NDArray[np.float64]
    evaluation_target: NDArray[np.int64]
    evaluation_metadata: pl.DataFrame


def ordered_seasons(
    dataframe: pl.DataFrame,
) -> tuple[str, ...]:
    """Return seasons ordered by their earliest fixture."""
    required = {
        "season",
        "kickoff_utc",
    }

    missing = sorted(required - set(dataframe.columns))

    if missing:
        raise ValueError(f"Missing season-order columns: {missing}")

    season_frame = (
        dataframe.filter(pl.col("season").is_not_null())
        .group_by("season")
        .agg(pl.col("kickoff_utc").min().alias("season_start"))
        .sort("season_start")
    )

    seasons = tuple(str(value) for value in season_frame["season"].to_list())

    if not seasons:
        raise ValueError("No seasons were found.")

    return seasons


def validate_fold_chronology(
    training: pl.DataFrame,
    evaluation: pl.DataFrame,
) -> None:
    """Ensure every training fixture precedes evaluation."""
    training_max = training.select(pl.col("kickoff_utc").max().alias("value"))[
        "value"
    ].item()

    evaluation_min = evaluation.select(pl.col("kickoff_utc").min().alias("value"))[
        "value"
    ].item()

    if not isinstance(training_max, datetime):
        raise TypeError("Training maximum timestamp is invalid.")

    if not isinstance(evaluation_min, datetime):
        raise TypeError("Evaluation minimum timestamp is invalid.")

    if training_max >= evaluation_min:
        raise ValueError("Backtest training and evaluation periods overlap.")


def create_backtest_fold(
    dataframe: pl.DataFrame,
    *,
    fold_number: int,
    training_seasons: tuple[str, ...],
    evaluation_season: str,
    feature_columns: tuple[
        str,
        ...,
    ] = MODEL_FEATURE_COLUMNS,
) -> BacktestFold:
    """Construct one expanding-window backtest fold."""
    if not training_seasons:
        raise ValueError("A backtest fold requires training seasons.")

    training = dataframe.filter(
        pl.col("season").is_in(training_seasons) & pl.col("target").is_not_null()
    ).sort(["kickoff_utc", "match_key"])

    evaluation = dataframe.filter(
        (pl.col("season") == evaluation_season) & pl.col("target").is_not_null()
    ).sort(["kickoff_utc", "match_key"])

    if training.is_empty():
        raise ValueError(f"Training data is empty for fold {fold_number}.")

    if evaluation.is_empty():
        raise ValueError(f"Evaluation data is empty for season {evaluation_season!r}.")

    validate_fold_chronology(
        training,
        evaluation,
    )

    training_features = np.asarray(
        prepare_feature_frame(
            training,
            feature_columns,
        ).to_numpy(),
        dtype=np.float64,
    )

    evaluation_features = np.asarray(
        prepare_feature_frame(
            evaluation,
            feature_columns,
        ).to_numpy(),
        dtype=np.float64,
    )

    training_target = np.asarray(
        training["target"].to_numpy(),
        dtype=np.int64,
    )

    evaluation_target = np.asarray(
        evaluation["target"].to_numpy(),
        dtype=np.int64,
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
        if column in evaluation.columns
    ]

    return BacktestFold(
        fold_number=fold_number,
        training_seasons=training_seasons,
        evaluation_season=evaluation_season,
        train_features=training_features,
        train_target=training_target,
        evaluation_features=evaluation_features,
        evaluation_target=evaluation_target,
        evaluation_metadata=evaluation.select(metadata_columns),
    )


def build_backtest_folds(
    dataframe: pl.DataFrame,
    *,
    minimum_training_seasons: int = (MINIMUM_TRAINING_SEASONS),
    maximum_evaluation_seasons: int | None = None,
    feature_columns: tuple[
        str,
        ...,
    ] = MODEL_FEATURE_COLUMNS,
) -> tuple[BacktestFold, ...]:
    """Build chronological expanding-window folds."""
    if minimum_training_seasons < 1:
        raise ValueError("minimum_training_seasons must be positive.")

    seasons = ordered_seasons(dataframe)

    if len(seasons) <= minimum_training_seasons:
        raise ValueError("Not enough seasons to construct backtest folds.")

    evaluation_seasons = seasons[minimum_training_seasons:]

    if maximum_evaluation_seasons is not None:
        if maximum_evaluation_seasons < 1:
            raise ValueError("maximum_evaluation_seasons must be positive.")

        evaluation_seasons = evaluation_seasons[-maximum_evaluation_seasons:]

    folds: list[BacktestFold] = []

    for fold_number, evaluation_season in enumerate(
        evaluation_seasons,
        start=1,
    ):
        evaluation_position = seasons.index(evaluation_season)

        training_seasons = seasons[:evaluation_position]

        folds.append(
            create_backtest_fold(
                dataframe,
                fold_number=fold_number,
                training_seasons=training_seasons,
                evaluation_season=evaluation_season,
                feature_columns=feature_columns,
            )
        )

    return tuple(folds)
