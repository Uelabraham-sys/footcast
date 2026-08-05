"""Tests for production future predictions."""

from pathlib import Path

import numpy as np
import polars as pl

from footcast.prediction.bundle import (
    save_production_bundle,
)
from footcast.prediction.predict import (
    build_prediction_output,
    generate_future_predictions,
    select_future_rows,
)
from tests.test_prediction_bundle import (
    create_bundle,
)


def create_future_dataset() -> pl.DataFrame:
    """Create one labelled and two future rows."""
    return pl.DataFrame(
        {
            "match_key": [
                "past",
                "future-1",
                "future-2",
            ],
            "season": [
                "2024/25",
                "2025/26",
                "2025/26",
            ],
            "kickoff_utc": [
                "2025-05-01T15:00:00Z",
                "2025-08-10T15:00:00Z",
                "2025-08-11T15:00:00Z",
            ],
            "home_team": [
                "Home A",
                "Home B",
                "Home C",
            ],
            "away_team": [
                "Away A",
                "Away B",
                "Away C",
            ],
            "target": [
                2,
                None,
                None,
            ],
            "feature_one": [
                1.0,
                0.5,
                -0.5,
            ],
            "feature_two": [
                1.0,
                0.2,
                -0.2,
            ],
        }
    ).with_columns(pl.col("kickoff_utc").str.to_datetime(time_zone="UTC"))


def test_future_selection_excludes_labelled_rows() -> None:
    """Only unlabelled fixtures should be predicted."""
    result = select_future_rows(create_future_dataset())

    assert result.height == 2

    assert result["match_key"].to_list() == [
        "future-1",
        "future-2",
    ]


def test_prediction_output_contains_probabilities() -> None:
    """Prediction output should contain outcome probabilities."""
    future = select_future_rows(create_future_dataset())

    probabilities = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.5, 0.3, 0.2],
        ],
        dtype=np.float64,
    )

    result = build_prediction_output(
        future,
        probabilities,
    )

    assert result["predicted_class"].to_list() == [
        2,
        0,
    ]

    assert result["predicted_outcome"].to_list() == [
        "home_win",
        "away_win",
    ]


def test_full_prediction_run_writes_outputs(
    tmp_path: Path,
) -> None:
    """A complete inference run should persist outputs."""
    dataset_path = tmp_path / "dataset.parquet"

    create_future_dataset().write_parquet(dataset_path)

    bundle_path = tmp_path / "bundle.joblib"

    save_production_bundle(
        create_bundle(),
        bundle_path,
    )

    parquet_output = tmp_path / "predictions.parquet"

    csv_output = tmp_path / "predictions.csv"
    audit_output = tmp_path / "audit.json"

    result = generate_future_predictions(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        parquet_output=parquet_output,
        csv_output=csv_output,
        audit_output=audit_output,
    )

    assert result.height == 2
    assert parquet_output.exists()
    assert csv_output.exists()
    assert audit_output.exists()
