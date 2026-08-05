"""Tests for the Day 4 model comparison."""

import json
from pathlib import Path

import polars as pl

from footcast.modelling.day_4_comparison import (
    build_day_4_comparison,
    write_day_4_comparison,
)

METRICS = {
    "accuracy": 0.50,
    "balanced_accuracy": 0.45,
    "macro_f1": 0.44,
    "log_loss": 1.00,
    "brier_score": 0.60,
    "ranked_probability_score": 0.20,
}


def create_reports(
    directory: Path,
) -> None:
    """Create representative Day 3 and ensemble reports."""
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pl.DataFrame(
        [
            {
                "model": "logistic_regression",
                "split": "test",
                **METRICS,
            }
        ]
    ).write_parquet(directory / "day_3_model_comparison.parquet")

    (directory / "ensemble_evaluation.json").write_text(
        json.dumps(
            {
                "test_metrics": {
                    **METRICS,
                    "log_loss": 0.95,
                }
            }
        ),
        encoding="utf-8",
    )


def test_ensemble_is_added_to_comparison(
    tmp_path: Path,
) -> None:
    """Day 4 comparison should include the ensemble."""
    create_reports(tmp_path)

    result = build_day_4_comparison(tmp_path)

    assert "probability_ensemble" in result["model"].to_list()

    assert result["model"].item(0) == "probability_ensemble"


def test_day_4_outputs_are_written(
    tmp_path: Path,
) -> None:
    """Comparison should write Parquet and JSON."""
    create_reports(tmp_path)

    result = build_day_4_comparison(tmp_path)

    write_day_4_comparison(
        result,
        tmp_path,
    )

    assert (tmp_path / "day_4_model_comparison.parquet").exists()

    assert (tmp_path / "day_4_model_comparison.json").exists()
