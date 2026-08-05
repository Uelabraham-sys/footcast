"""Combine FootCast model evaluation reports."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl
import typer

DEFAULT_REPORT_DIRECTORY: Final[Path] = Path("artifacts/reports")

app = typer.Typer(help="Build the final Day 3 model comparison.")


def load_json(path: Path) -> dict[str, Any]:
    """Load a required JSON report."""
    if not path.exists():
        raise FileNotFoundError(f"Required report was not found: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError(f"Report must contain an object: {path}")

    return value


def metric_row(
    *,
    model: str,
    split: str,
    metrics: dict[str, Any],
) -> dict[str, str | float]:
    """Convert metrics into one comparison row."""
    required = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
    )

    row: dict[str, str | float] = {
        "model": model,
        "split": split,
    }

    for name in required:
        value = metrics.get(name)

        if not isinstance(value, (int, float)):
            raise TypeError(f"Metric {name!r} must be numeric.")

        row[name] = float(value)

    return row


def build_day_3_comparison(
    report_directory: Path,
) -> pl.DataFrame:
    """Combine baseline, logistic and HGB test metrics."""
    baseline_path = report_directory / "baseline_comparison.parquet"

    if not baseline_path.exists():
        raise FileNotFoundError("Run `make evaluate-baselines` first.")

    baselines = pl.read_parquet(baseline_path)

    logistic = load_json(report_directory / "logistic_regression.json")

    hgb = load_json(report_directory / "hgb_evaluation.json")

    logistic_test = logistic.get("test_metrics")
    hgb_test = hgb.get("test_metrics")

    if not isinstance(logistic_test, dict):
        raise TypeError("Logistic test metrics are missing.")

    if not isinstance(hgb_test, dict):
        raise TypeError("HGB test metrics are missing.")

    hgb_uncalibrated = hgb_test.get("uncalibrated")
    hgb_calibrated = hgb_test.get("calibrated")

    if not isinstance(
        hgb_uncalibrated,
        dict,
    ):
        raise TypeError("Uncalibrated HGB metrics are missing.")

    if not isinstance(
        hgb_calibrated,
        dict,
    ):
        raise TypeError("Calibrated HGB metrics are missing.")

    learned_rows = pl.DataFrame(
        [
            metric_row(
                model="logistic_regression",
                split="test",
                metrics=logistic_test,
            ),
            metric_row(
                model="hist_gradient_boosting",
                split="test",
                metrics=hgb_uncalibrated,
            ),
            metric_row(
                model=("hist_gradient_boosting_calibrated"),
                split="test",
                metrics=hgb_calibrated,
            ),
        ]
    )

    test_baselines = baselines.filter(pl.col("split") == "test")

    return pl.concat(
        [
            test_baselines,
            learned_rows,
        ],
        how="diagonal_relaxed",
    ).sort(
        [
            "log_loss",
            "brier_score",
        ]
    )


def write_day_3_comparison(
    comparison: pl.DataFrame,
    report_directory: Path,
) -> None:
    """Write Parquet and JSON comparison outputs."""
    parquet_path = report_directory / "day_3_model_comparison.parquet"
    json_path = report_directory / "day_3_model_comparison.json"

    comparison.write_parquet(
        parquet_path,
        compression="zstd",
        statistics=True,
    )

    best = comparison.row(
        0,
        named=True,
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "comparison_scope": ("held-out chronological test season"),
        "primary_metric": "log_loss",
        "best_test_model": best.get("model"),
        "models": comparison.to_dicts(),
    }

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


@app.command()
def run() -> None:
    """Build the final Day 3 model comparison."""
    comparison = build_day_3_comparison(DEFAULT_REPORT_DIRECTORY)

    write_day_3_comparison(
        comparison,
        DEFAULT_REPORT_DIRECTORY,
    )

    typer.echo("DAY 3 MODEL COMPARISON")
    typer.echo("=" * 78)
    typer.echo(str(comparison))


if __name__ == "__main__":
    app()
