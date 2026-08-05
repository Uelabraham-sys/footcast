"""Build the final Day 4 model comparison."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl
import typer

DEFAULT_REPORT_DIRECTORY: Final[Path] = Path("artifacts/reports")

app = typer.Typer(help="Build the final Day 4 model comparison.")


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required report was not found: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")

    return value


def metric_row(
    *,
    model: str,
    metrics: dict[str, Any],
) -> dict[str, str | float]:
    """Create a comparison row from metric values."""
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
    }

    for metric_name in required:
        value = metrics.get(metric_name)

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(f"Metric {metric_name!r} must be numeric.")

        row[metric_name] = float(value)

    return row


def build_day_4_comparison(
    report_directory: Path,
) -> pl.DataFrame:
    """Add the ensemble to the Day 3 comparison."""
    day_3_path = report_directory / "day_3_model_comparison.parquet"

    if not day_3_path.exists():
        raise FileNotFoundError("Run `make compare-models` first.")

    day_3 = pl.read_parquet(day_3_path)

    ensemble_report = load_json_object(report_directory / "ensemble_evaluation.json")

    ensemble_metrics = ensemble_report.get("test_metrics")

    if not isinstance(
        ensemble_metrics,
        dict,
    ):
        raise TypeError("Ensemble test metrics are missing.")

    ensemble_row = pl.DataFrame(
        [
            metric_row(
                model="probability_ensemble",
                metrics=ensemble_metrics,
            )
        ]
    )

    return pl.concat(
        [
            day_3,
            ensemble_row,
        ],
        how="diagonal_relaxed",
    ).sort(
        [
            "log_loss",
            "brier_score",
        ]
    )


def write_day_4_comparison(
    comparison: pl.DataFrame,
    report_directory: Path,
) -> None:
    """Write Day 4 comparison outputs."""
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.write_parquet(
        report_directory / "day_4_model_comparison.parquet",
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

    (report_directory / "day_4_model_comparison.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


@app.command()
def run() -> None:
    """Build the final Day 4 comparison."""
    comparison = build_day_4_comparison(DEFAULT_REPORT_DIRECTORY)

    write_day_4_comparison(
        comparison,
        DEFAULT_REPORT_DIRECTORY,
    )

    typer.echo("DAY 4 MODEL COMPARISON")
    typer.echo("=" * 78)
    typer.echo(str(comparison))


if __name__ == "__main__":
    app()
