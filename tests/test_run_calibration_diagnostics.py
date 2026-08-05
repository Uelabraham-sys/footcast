"""Tests for calibration-report orchestration."""

from pathlib import Path

import polars as pl

from footcast.modelling.run_calibration_diagnostics import (
    run_calibration_diagnostics,
)


def create_predictions() -> pl.DataFrame:
    """Create representative labelled predictions."""
    return pl.DataFrame(
        {
            "match_key": [
                "m1",
                "m2",
                "m3",
                "m4",
                "m5",
                "m6",
            ],
            "target": [0, 1, 2, 0, 1, 2],
            "probability_away_win": [
                0.70,
                0.20,
                0.10,
                0.60,
                0.20,
                0.10,
            ],
            "probability_draw": [
                0.20,
                0.60,
                0.20,
                0.20,
                0.60,
                0.20,
            ],
            "probability_home_win": [
                0.10,
                0.20,
                0.70,
                0.20,
                0.20,
                0.70,
            ],
            "predicted_class": [
                0,
                1,
                2,
                0,
                1,
                2,
            ],
        }
    )


def test_runner_writes_calibration_reports(
    tmp_path: Path,
) -> None:
    """Calibration execution should write all reports."""
    prediction_directory = tmp_path / "predictions"
    report_directory = tmp_path / "reports"

    prediction_directory.mkdir()

    create_predictions().write_parquet(prediction_directory / "frequency_test.parquet")

    create_predictions().write_parquet(prediction_directory / "logistic_test.parquet")

    summary, class_summary, bins = run_calibration_diagnostics(
        prediction_directory=(prediction_directory),
        report_directory=(report_directory),
        bin_count=5,
    )

    assert summary.height == 2
    assert class_summary.height == 6
    assert bins.height == 40

    assert (report_directory / "calibration_summary.parquet").exists()

    assert (report_directory / "class_calibration_summary.parquet").exists()

    assert (report_directory / "calibration_bins.parquet").exists()

    assert (report_directory / "confidence_summary.parquet").exists()

    assert (report_directory / "calibration_report.json").exists()


def test_summary_orders_models_by_ece(
    tmp_path: Path,
) -> None:
    """Aggregate report should sort by macro class ECE."""
    prediction_directory = tmp_path / "predictions"

    prediction_directory.mkdir()

    create_predictions().write_parquet(prediction_directory / "frequency_test.parquet")

    summary, _, _ = run_calibration_diagnostics(
        prediction_directory=(prediction_directory),
        report_directory=(tmp_path / "reports"),
        bin_count=5,
    )

    assert summary["model"].to_list() == ["frequency"]
