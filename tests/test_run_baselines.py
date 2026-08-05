"""Tests for baseline evaluation orchestration."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from footcast.modelling.run_baselines import (
    run_baseline_evaluation,
)


def create_model_dataset() -> pl.DataFrame:
    """Create a small complete modelling dataset."""
    start = datetime(2022, 8, 1, tzinfo=UTC)

    rows: list[dict[str, object]] = []

    split_definitions = (
        ("train", "2022/23", 0),
        ("train", "2022/23", 1),
        ("train", "2022/23", 2),
        ("train", "2022/23", 2),
        ("validation", "2023/24", 0),
        ("validation", "2023/24", 1),
        ("validation", "2023/24", 2),
        ("test", "2024/25", 0),
        ("test", "2024/25", 1),
        ("test", "2024/25", 2),
    )

    for index, (
        split,
        season,
        target,
    ) in enumerate(split_definitions):
        rows.append(
            {
                "match_key": f"m{index}",
                "season": season,
                "kickoff_utc": (start + timedelta(days=index * 30)),
                "match_date": (start + timedelta(days=index * 30)).date(),
                "home_team_id": "home",
                "home_team": "Home",
                "away_team_id": "away",
                "away_team": "Away",
                "full_time_result": {
                    0: "A",
                    1: "D",
                    2: "H",
                }[target],
                "home_expected_score": (0.35 + index * 0.03),
                "away_expected_score": (0.65 - index * 0.03),
                "target": target,
                "split": split,
                "feature_one": float(index),
            }
        )

    return pl.DataFrame(rows)


def test_baseline_runner_writes_reports(
    tmp_path: Path,
) -> None:
    """Baseline evaluation should write reports and predictions."""
    dataset_path = tmp_path / "model_dataset.parquet"
    report_directory = tmp_path / "reports"
    prediction_directory = tmp_path / "predictions"

    create_model_dataset().write_parquet(dataset_path)

    comparison = run_baseline_evaluation(
        dataset_path=dataset_path,
        report_directory=report_directory,
        prediction_directory=prediction_directory,
        draw_probability=0.25,
        feature_columns=("feature_one",),
    )

    assert comparison.height == 6
    assert (report_directory / "baseline_comparison.json").exists()
    assert (report_directory / "baseline_comparison.parquet").exists()

    assert (prediction_directory / "majority_validation.parquet").exists()
    assert (prediction_directory / "frequency_test.parquet").exists()
    assert (prediction_directory / "elo_test.parquet").exists()


def test_baseline_comparison_contains_all_models(
    tmp_path: Path,
) -> None:
    """Comparison table should contain every baseline and split."""
    dataset_path = tmp_path / "model_dataset.parquet"

    create_model_dataset().write_parquet(dataset_path)

    result = run_baseline_evaluation(
        dataset_path=dataset_path,
        report_directory=tmp_path / "reports",
        prediction_directory=tmp_path / "predictions",
        draw_probability=0.25,
        feature_columns=("feature_one",),
    )

    assert set(result["model"].to_list()) == {
        "majority",
        "frequency",
        "elo",
    }

    assert set(result["split"].to_list()) == {
        "validation",
        "test",
    }
