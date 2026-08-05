"""Tests for HGB training orchestration."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from footcast.modelling.dataset import (
    build_model_datasets,
)
from footcast.modelling.gradient_boosting import (
    HGBParameters,
)
from footcast.modelling.train_gradient_boosting import (
    chronological_validation_partition,
    train_and_evaluate_hgb,
)

FEATURE_COLUMNS = (
    "feature_one",
    "feature_two",
)


def create_model_dataset() -> pl.DataFrame:
    """Create synthetic chronological model data."""
    start = datetime(2021, 8, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []

    split_sizes = (
        ("train", "2021/22", 90),
        ("validation", "2022/23", 60),
        ("test", "2023/24", 30),
    )

    index = 0

    for split, season, size in split_sizes:
        for local_index in range(size):
            target = local_index % 3
            kickoff = start + timedelta(days=index * 3)

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
                    "split": split,
                    "feature_one": float(target * 2) + local_index * 0.001,
                    "feature_two": float(target) + local_index * 0.001,
                }
            )

            index += 1

    return pl.DataFrame(rows)


def test_validation_partition_is_chronological() -> None:
    """Tuning must precede calibration."""
    datasets = build_model_datasets(
        create_model_dataset(),
        feature_columns=FEATURE_COLUMNS,
    )

    tuning, calibration = chronological_validation_partition(
        datasets.validation,
        tuning_fraction=0.6,
    )

    assert tuning.target.shape[0] == 36
    assert calibration.target.shape[0] == 24

    assert (
        tuning.metadata["kickoff_utc"].max() < calibration.metadata["kickoff_utc"].min()
    )


def test_hgb_training_writes_artifacts(
    tmp_path: Path,
) -> None:
    """Training should write models, reports and predictions."""
    dataset_path = tmp_path / "model_dataset.parquet"

    create_model_dataset().write_parquet(dataset_path)

    selection = train_and_evaluate_hgb(
        dataset_path=dataset_path,
        model_directory=tmp_path / "models",
        report_directory=tmp_path / "reports",
        prediction_directory=tmp_path / "predictions",
        parameter_grid=(
            HGBParameters(
                max_iter=20,
                max_leaf_nodes=7,
                min_samples_leaf=5,
            ),
        ),
        feature_columns=FEATURE_COLUMNS,
        tuning_fraction=0.6,
        importance_repeats=2,
    )

    assert selection.height == 1

    assert (tmp_path / "models" / "hist_gradient_boosting.joblib").exists()

    assert (tmp_path / "models" / "hist_gradient_boosting_calibrated.joblib").exists()

    assert (tmp_path / "reports" / "hgb_selection.parquet").exists()

    assert (tmp_path / "reports" / "hgb_feature_importance.parquet").exists()

    assert (tmp_path / "reports" / "hgb_evaluation.json").exists()

    assert (tmp_path / "predictions" / "hgb_calibrated_test.parquet").exists()
