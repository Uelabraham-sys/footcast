"""Tests for ensemble training orchestration."""

from pathlib import Path

import polars as pl

from footcast.modelling.train_ensemble import (
    parse_weights,
    train_and_evaluate_ensemble,
)


def create_predictions(
    *,
    stronger_first_model: bool,
) -> pl.DataFrame:
    """Create representative prediction data."""
    targets = [0, 1, 2, 0, 1, 2]

    if stronger_first_model:
        probabilities = [
            [0.75, 0.15, 0.10],
            [0.15, 0.70, 0.15],
            [0.10, 0.15, 0.75],
            [0.70, 0.20, 0.10],
            [0.20, 0.65, 0.15],
            [0.10, 0.20, 0.70],
        ]
    else:
        probabilities = [
            [0.40, 0.35, 0.25],
            [0.30, 0.40, 0.30],
            [0.25, 0.35, 0.40],
            [0.40, 0.35, 0.25],
            [0.30, 0.40, 0.30],
            [0.25, 0.35, 0.40],
        ]

    return pl.DataFrame(
        {
            "match_key": [f"m{index}" for index in range(6)],
            "target": targets,
            "probability_away_win": [row[0] for row in probabilities],
            "probability_draw": [row[1] for row in probabilities],
            "probability_home_win": [row[2] for row in probabilities],
        }
    )


def test_parse_weights_removes_duplicates() -> None:
    """Weight parser should return sorted unique values."""
    assert parse_weights("1,0.5,0,0.5") == (
        0.0,
        0.5,
        1.0,
    )


def test_ensemble_runner_writes_outputs(
    tmp_path: Path,
) -> None:
    """Ensemble selection should persist all outputs."""
    prediction_directory = tmp_path / "predictions"

    report_directory = tmp_path / "reports"

    prediction_directory.mkdir()

    first = create_predictions(stronger_first_model=True)

    second = create_predictions(stronger_first_model=False)

    first.write_parquet(prediction_directory / "logistic_calibration.parquet")

    second.write_parquet(prediction_directory / "hgb_calibration.parquet")

    first.write_parquet(prediction_directory / "logistic_test.parquet")

    second.write_parquet(prediction_directory / "hgb_calibrated_test.parquet")

    selection = train_and_evaluate_ensemble(
        prediction_directory=(prediction_directory),
        report_directory=(report_directory),
        candidate_weights=(
            0.0,
            0.5,
            1.0,
        ),
    )

    assert selection.height == 3

    assert (report_directory / "ensemble_selection.parquet").exists()

    assert (report_directory / "ensemble_evaluation.json").exists()

    assert (prediction_directory / "ensemble_validation.parquet").exists()

    assert (prediction_directory / "ensemble_test.parquet").exists()


def test_selection_prefers_stronger_model(
    tmp_path: Path,
) -> None:
    """Selection should prefer the stronger first model."""
    prediction_directory = tmp_path / "predictions"

    prediction_directory.mkdir()

    first = create_predictions(stronger_first_model=True)

    second = create_predictions(stronger_first_model=False)

    first.write_parquet(prediction_directory / "logistic_calibration.parquet")

    second.write_parquet(prediction_directory / "hgb_calibration.parquet")

    first.write_parquet(prediction_directory / "logistic_test.parquet")

    second.write_parquet(prediction_directory / "hgb_calibrated_test.parquet")

    selection = train_and_evaluate_ensemble(
        prediction_directory=(prediction_directory),
        report_directory=(tmp_path / "reports"),
        candidate_weights=(
            0.0,
            0.5,
            1.0,
        ),
    )

    assert selection["first_model_weight"].item(0) == 1.0
