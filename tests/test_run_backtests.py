"""Tests for FootCast backtest orchestration."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from footcast.modelling.gradient_boosting import (
    HGBParameters,
)
from footcast.modelling.run_backtests import (
    run_backtests,
    summarise_backtests,
)

FEATURE_COLUMNS = (
    "feature_one",
    "feature_two",
)


def create_dataset() -> pl.DataFrame:
    """Create model data suitable for repeated fitting."""
    start = datetime(2020, 8, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []

    seasons = (
        "2020/21",
        "2021/22",
        "2022/23",
        "2023/24",
    )

    index = 0

    for season_number, season in enumerate(seasons):
        for local_index in range(30):
            target = local_index % 3
            kickoff = start + timedelta(days=season_number * 365 + local_index * 3)

            rows.append(
                {
                    "match_key": f"m{index}",
                    "season": season,
                    "kickoff_utc": kickoff,
                    "match_date": kickoff.date(),
                    "home_team_id": "home",
                    "home_team": "Home",
                    "away_team_id": "away",
                    "away_team": "Away",
                    "full_time_result": {
                        0: "A",
                        1: "D",
                        2: "H",
                    }[target],
                    "target": target,
                    "feature_one": float(target * 2) + local_index * 0.001,
                    "feature_two": float(target) + local_index * 0.001,
                }
            )

            index += 1

    return pl.DataFrame(rows)


def test_backtest_runner_writes_reports(
    tmp_path: Path,
) -> None:
    """Backtest execution should persist metrics and predictions."""
    dataset_path = tmp_path / "dataset.parquet"
    create_dataset().write_parquet(dataset_path)

    fold_metrics, summary = run_backtests(
        dataset_path=dataset_path,
        report_directory=tmp_path / "reports",
        prediction_directory=tmp_path / "predictions",
        minimum_training_seasons=2,
        maximum_evaluation_seasons=2,
        feature_columns=FEATURE_COLUMNS,
        logistic_c=1.0,
        hgb_parameters=HGBParameters(
            max_iter=20,
            max_leaf_nodes=7,
            min_samples_leaf=5,
        ),
    )

    assert fold_metrics.height == 6
    assert summary.height == 3

    assert (tmp_path / "reports" / "backtest_fold_metrics.parquet").exists()

    assert (tmp_path / "reports" / "backtest_summary.parquet").exists()

    assert (tmp_path / "reports" / "backtest_report.json").exists()

    assert len(list((tmp_path / "predictions").glob("*.parquet"))) == 6


def test_backtest_summary_orders_log_loss() -> None:
    """Summary should rank models by average log loss."""
    metrics = pl.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "accuracy": [0.5, 0.6, 0.4, 0.5],
            "balanced_accuracy": [
                0.4,
                0.5,
                0.3,
                0.4,
            ],
            "macro_f1": [0.4, 0.5, 0.3, 0.4],
            "log_loss": [0.9, 1.0, 1.1, 1.2],
            "brier_score": [0.5, 0.6, 0.7, 0.8],
            "ranked_probability_score": [
                0.2,
                0.3,
                0.4,
                0.5,
            ],
        }
    )

    result = summarise_backtests(metrics)

    assert result["model"].to_list()[0] == "a"
