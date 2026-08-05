"""Tests for production model fitting."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from footcast.prediction.fit_production import (
    fit_production_model,
)

FEATURE_COLUMNS = (
    "feature_one",
    "feature_two",
)


def create_dataset() -> pl.DataFrame:
    """Create sufficient chronological labelled data."""
    start = datetime(
        2020,
        8,
        1,
        tzinfo=UTC,
    )

    rows: list[dict[str, object]] = []

    for index in range(150):
        target = index % 3
        kickoff = start + timedelta(days=index * 3)

        rows.append(
            {
                "match_key": f"m{index}",
                "season": "2020/21",
                "kickoff_utc": kickoff,
                "target": target,
                "feature_one": float(target),
                "feature_two": float(index % 10),
            }
        )

    return pl.DataFrame(rows)


def write_reports(
    directory: Path,
) -> tuple[Path, Path, Path]:
    """Write representative selected-model reports."""
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    ensemble = directory / "ensemble.json"
    hgb = directory / "hgb.json"
    logistic = directory / "logistic.json"

    ensemble.write_text(
        json.dumps(
            {
                "selected_weights": {
                    "logistic_regression": 0.6,
                    "hist_gradient_boosting_calibrated": 0.4,
                }
            }
        ),
        encoding="utf-8",
    )

    hgb.write_text(
        json.dumps(
            {
                "selected_parameters": {
                    "learning_rate": 0.05,
                    "max_iter": 20,
                    "max_leaf_nodes": 7,
                    "min_samples_leaf": 5,
                    "l2_regularization": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    logistic.write_text(
        json.dumps(
            {
                "selected_parameters": {
                    "regularisation_strength": 1.0,
                    "class_weight": "none",
                }
            }
        ),
        encoding="utf-8",
    )

    return ensemble, hgb, logistic


def test_production_fit_writes_bundle_and_reports(
    tmp_path: Path,
) -> None:
    """Production fitting should persist required artifacts."""
    dataset_path = tmp_path / "dataset.parquet"

    create_dataset().write_parquet(dataset_path)

    ensemble, hgb, logistic = write_reports(tmp_path / "input_reports")

    bundle_path = tmp_path / "models" / "bundle.joblib"
    manifest_path = tmp_path / "models" / "manifest.json"
    report_path = tmp_path / "reports" / "training.json"

    fit_production_model(
        dataset_path=dataset_path,
        ensemble_report_path=ensemble,
        hgb_report_path=hgb,
        logistic_report_path=logistic,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        training_report_path=report_path,
        feature_columns=FEATURE_COLUMNS,
        calibration_fraction=0.2,
    )

    assert bundle_path.exists()
    assert manifest_path.exists()
    assert report_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["feature_count"] == 2

    assert manifest["ensemble_weights"] == {
        "logistic_regression": 0.6,
        "hist_gradient_boosting_calibrated": 0.4,
    }
