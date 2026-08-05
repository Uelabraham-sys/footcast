"""Run calibration diagnostics across FootCast models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import polars as pl
import typer

from footcast.modelling.calibration_diagnostics import (
    build_class_calibration_diagnostics,
    build_confidence_diagnostics,
)

DEFAULT_PREDICTION_DIRECTORY: Final[Path] = Path("artifacts/predictions")

DEFAULT_REPORT_DIRECTORY: Final[Path] = Path("artifacts/reports")

DEFAULT_BIN_COUNT: Final[int] = 10

PREDICTION_FILES: Final[dict[str, str]] = {
    "frequency": "frequency_test.parquet",
    "elo": "elo_test.parquet",
    "logistic_regression": ("logistic_test.parquet"),
    "hist_gradient_boosting": ("hgb_test.parquet"),
    "hist_gradient_boosting_calibrated": ("hgb_calibrated_test.parquet"),
}

app = typer.Typer(help="Build FootCast probability-calibration reports.")


def discover_prediction_files(
    prediction_directory: Path,
) -> dict[str, Path]:
    """Return model prediction files that currently exist."""
    available = {
        model_name: (prediction_directory / filename)
        for model_name, filename in PREDICTION_FILES.items()
        if (prediction_directory / filename).exists()
    }

    if not available:
        raise FileNotFoundError(
            "No supported prediction files were found. "
            "Run Day 3 model evaluation first."
        )

    return available


def load_prediction_file(
    path: Path,
) -> pl.DataFrame:
    """Load one prediction dataset."""
    dataframe = pl.read_parquet(path)

    if dataframe.is_empty():
        raise ValueError(f"Prediction dataset is empty: {path}")

    return dataframe


def aggregate_model_calibration(
    class_summary: pl.DataFrame,
    confidence_summary: pl.DataFrame,
) -> pl.DataFrame:
    """Create one aggregate calibration row per model."""
    class_aggregate = class_summary.group_by("model").agg(
        pl.col("expected_calibration_error").mean().alias("macro_class_ece"),
        pl.col("maximum_calibration_error").max().alias("worst_class_mce"),
        pl.col("calibration_bias").abs().mean().alias("mean_absolute_class_bias"),
    )

    return class_aggregate.join(
        confidence_summary,
        on="model",
        how="inner",
        validate="1:1",
    ).sort(
        [
            "macro_class_ece",
            "confidence_ece",
        ]
    )


def run_calibration_diagnostics(
    *,
    prediction_directory: Path,
    report_directory: Path,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    """Generate class-specific and confidence calibration reports."""
    prediction_files = discover_prediction_files(prediction_directory)

    class_summary_frames: list[pl.DataFrame] = []
    class_bin_frames: list[pl.DataFrame] = []
    confidence_bin_frames: list[pl.DataFrame] = []

    confidence_records: list[dict[str, str | int | float]] = []

    for model_name, path in prediction_files.items():
        predictions = load_prediction_file(path)

        class_summary, class_bins = build_class_calibration_diagnostics(
            predictions,
            model_name=model_name,
            bin_count=bin_count,
        )

        confidence_metrics, confidence_bins = build_confidence_diagnostics(
            predictions,
            model_name=model_name,
            bin_count=bin_count,
        )

        class_summary_frames.append(class_summary)
        class_bin_frames.append(class_bins)
        confidence_bin_frames.append(confidence_bins)

        confidence_records.append(
            {
                "model": model_name,
                "confidence_ece": (confidence_metrics.expected_calibration_error),
                "confidence_mce": (confidence_metrics.maximum_calibration_error),
                "mean_confidence": (confidence_metrics.mean_confidence),
                "observed_accuracy": (confidence_metrics.observed_accuracy),
                "overconfidence_gap": (confidence_metrics.overconfidence_gap),
                "confidence_populated_bins": (confidence_metrics.populated_bins),
            }
        )

    class_summary = pl.concat(class_summary_frames)

    class_bins = pl.concat(class_bin_frames)

    confidence_bins = pl.concat(confidence_bin_frames)

    confidence_summary = pl.DataFrame(confidence_records)

    aggregate_summary = aggregate_model_calibration(
        class_summary,
        confidence_summary,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_summary.write_parquet(
        report_directory / "calibration_summary.parquet",
        compression="zstd",
        statistics=True,
    )

    class_summary.write_parquet(
        report_directory / "class_calibration_summary.parquet",
        compression="zstd",
        statistics=True,
    )

    combined_bins = pl.concat(
        [
            class_bins.with_columns(pl.lit("class").alias("diagnostic_type")),
            confidence_bins,
        ],
        how="diagonal_relaxed",
    )

    combined_bins.write_parquet(
        report_directory / "calibration_bins.parquet",
        compression="zstd",
        statistics=True,
    )

    confidence_summary.write_parquet(
        report_directory / "confidence_summary.parquet",
        compression="zstd",
        statistics=True,
    )

    best = aggregate_summary.row(
        0,
        named=True,
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "bin_count": bin_count,
        "binning_strategy": ("uniform equal-width probability bins"),
        "models_analysed": list(prediction_files),
        "selection_metric": ("macro class expected calibration error"),
        "best_calibrated_model": (best.get("model")),
        "aggregate_summary": (aggregate_summary.to_dicts()),
        "class_summary": (class_summary.to_dicts()),
        "confidence_summary": (confidence_summary.to_dicts()),
        "probability_ensemble": ("ensemble_test.parquet"),
    }

    (report_directory / "calibration_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return (
        aggregate_summary,
        class_summary,
        combined_bins,
    )


@app.command()
def run(
    prediction_directory: Annotated[
        Path,
        typer.Option(
            help="Directory containing model predictions.",
        ),
    ] = DEFAULT_PREDICTION_DIRECTORY,
    report_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for calibration reports.",
        ),
    ] = DEFAULT_REPORT_DIRECTORY,
    bin_count: Annotated[
        int,
        typer.Option(
            min=2,
            max=50,
            help="Number of equal-width probability bins.",
        ),
    ] = DEFAULT_BIN_COUNT,
) -> None:
    """Generate calibration diagnostics."""
    summary, _, _ = run_calibration_diagnostics(
        prediction_directory=(prediction_directory),
        report_directory=(report_directory),
        bin_count=bin_count,
    )

    typer.echo("CALIBRATION SUMMARY")
    typer.echo("=" * 78)
    typer.echo(str(summary))


if __name__ == "__main__":
    app()
