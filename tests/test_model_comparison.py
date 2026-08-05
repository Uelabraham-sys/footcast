"""Tests for final Day 3 model comparison."""

import json
from pathlib import Path

import polars as pl

from footcast.modelling.model_comparison import (
    build_day_3_comparison,
    write_day_3_comparison,
)

METRICS = {
    "accuracy": 0.50,
    "balanced_accuracy": 0.45,
    "macro_f1": 0.44,
    "log_loss": 1.05,
    "brier_score": 0.63,
    "ranked_probability_score": 0.20,
}


def create_reports(directory: Path) -> None:
    """Create representative model reports."""
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pl.DataFrame(
        [
            {
                "model": "frequency",
                "split": "test",
                **METRICS,
            }
        ]
    ).write_parquet(directory / "baseline_comparison.parquet")

    (directory / "logistic_regression.json").write_text(
        json.dumps(
            {
                "test_metrics": {
                    **METRICS,
                    "log_loss": 1.00,
                }
            }
        ),
        encoding="utf-8",
    )

    (directory / "hgb_evaluation.json").write_text(
        json.dumps(
            {
                "test_metrics": {
                    "uncalibrated": {
                        **METRICS,
                        "log_loss": 0.98,
                    },
                    "calibrated": {
                        **METRICS,
                        "log_loss": 0.95,
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_comparison_orders_by_log_loss(
    tmp_path: Path,
) -> None:
    """Lowest test log loss should rank first."""
    create_reports(tmp_path)

    result = build_day_3_comparison(tmp_path)

    assert result["model"].to_list()[0] == ("hist_gradient_boosting_calibrated")


def test_comparison_outputs_are_written(
    tmp_path: Path,
) -> None:
    """Comparison should write Parquet and JSON."""
    create_reports(tmp_path)

    comparison = build_day_3_comparison(tmp_path)

    write_day_3_comparison(
        comparison,
        tmp_path,
    )

    assert (tmp_path / "day_3_model_comparison.parquet").exists()

    assert (tmp_path / "day_3_model_comparison.json").exists()
