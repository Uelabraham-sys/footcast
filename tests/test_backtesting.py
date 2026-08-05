"""Tests for expanding-window football backtesting."""

from datetime import UTC, datetime, timedelta

import polars as pl

from footcast.modelling.backtesting import (
    build_backtest_folds,
    ordered_seasons,
)

FEATURE_COLUMNS = (
    "feature_one",
    "feature_two",
)


def create_dataset() -> pl.DataFrame:
    """Create four chronological football seasons."""
    start = datetime(2021, 8, 1, tzinfo=UTC)

    rows: list[dict[str, object]] = []

    seasons = (
        "2021/22",
        "2022/23",
        "2023/24",
        "2024/25",
    )

    index = 0

    for season_number, season in enumerate(seasons):
        for local_index in range(9):
            target = local_index % 3
            kickoff = start + timedelta(days=season_number * 365 + local_index * 7)

            rows.append(
                {
                    "match_key": f"m{index}",
                    "season": season,
                    "kickoff_utc": kickoff,
                    "home_team_id": "home",
                    "away_team_id": "away",
                    "target": target,
                    "feature_one": float(target),
                    "feature_two": float(local_index),
                }
            )

            index += 1

    return pl.DataFrame(rows)


def test_seasons_are_ordered_by_fixture_date() -> None:
    """Season labels should follow chronology, not text sorting."""
    assert ordered_seasons(create_dataset()) == (
        "2021/22",
        "2022/23",
        "2023/24",
        "2024/25",
    )


def test_backtest_folds_expand_training_window() -> None:
    """Every successive fold should add historical seasons."""
    folds = build_backtest_folds(
        create_dataset(),
        minimum_training_seasons=2,
        feature_columns=FEATURE_COLUMNS,
    )

    assert len(folds) == 2

    assert folds[0].training_seasons == (
        "2021/22",
        "2022/23",
    )
    assert folds[0].evaluation_season == "2023/24"

    assert folds[1].training_seasons == (
        "2021/22",
        "2022/23",
        "2023/24",
    )
    assert folds[1].evaluation_season == "2024/25"


def test_fold_arrays_have_expected_shapes() -> None:
    """Fold feature and target arrays should align."""
    folds = build_backtest_folds(
        create_dataset(),
        minimum_training_seasons=2,
        feature_columns=FEATURE_COLUMNS,
    )

    first = folds[0]

    assert first.train_features.shape == (18, 2)
    assert first.train_target.shape == (18,)
    assert first.evaluation_features.shape == (9, 2)
    assert first.evaluation_target.shape == (9,)


def test_maximum_evaluation_seasons_limits_folds() -> None:
    """Only the most recent requested folds should remain."""
    folds = build_backtest_folds(
        create_dataset(),
        minimum_training_seasons=1,
        maximum_evaluation_seasons=2,
        feature_columns=FEATURE_COLUMNS,
    )

    assert len(folds) == 2
    assert folds[0].evaluation_season == "2023/24"
    assert folds[1].evaluation_season == "2024/25"
