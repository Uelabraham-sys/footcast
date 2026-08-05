"""Evaluate FootCast probability baselines."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import numpy as np
import polars as pl
import typer
from numpy.typing import NDArray

from footcast.modelling.baselines import (
    DEFAULT_DRAW_PROBABILITY,
    elo_outcome_probabilities,
    frequency_probabilities,
    majority_class_probabilities,
)
from footcast.modelling.dataset import (
    MODEL_FEATURE_COLUMNS,
    ModelDatasets,
    ModelSplit,
    build_model_datasets,
    load_model_dataset,
)
from footcast.modelling.evaluation import (
    create_prediction_frame,
    write_prediction_frame,
)
from footcast.modelling.metrics import (
    ClassificationMetrics,
    evaluate_probabilities,
)

DEFAULT_MODEL_DATASET: Final[Path] = Path("data/gold/model_dataset.parquet")
DEFAULT_REPORT_DIRECTORY: Final[Path] = Path("artifacts/reports")
DEFAULT_PREDICTION_DIRECTORY: Final[Path] = Path("artifacts/predictions")

app = typer.Typer(help="Evaluate simple chronological football baselines.")


def extract_home_elo_expectation(
    split: ModelSplit,
) -> NDArray[np.float64]:
    """Read home Elo expectations from split metadata and features."""
    if "home_expected_score" in split.metadata.columns:
        return np.asarray(
            split.metadata["home_expected_score"].to_numpy(),
            dtype=np.float64,
        )

    raise ValueError("home_expected_score is missing from split metadata.")


def metric_record(
    model_name: str,
    split_name: str,
    metrics: ClassificationMetrics,
) -> dict[str, str | float]:
    """Convert one evaluation into a flat comparison record."""
    return {
        "model": model_name,
        "split": split_name,
        **metrics.to_dict(),
    }


def evaluate_and_write_predictions(
    *,
    model_name: str,
    split_name: str,
    split: ModelSplit,
    probabilities: NDArray[np.float64],
    prediction_directory: Path,
) -> ClassificationMetrics:
    """Evaluate probabilities and write prediction rows."""
    metrics = evaluate_probabilities(
        target=split.target,
        probabilities=probabilities,
    )

    predictions = create_prediction_frame(
        metadata=split.metadata,
        probabilities=probabilities,
    )

    write_prediction_frame(
        predictions,
        prediction_directory / f"{model_name}_{split_name}.parquet",
    )

    return metrics


def build_baseline_probability_sets(
    datasets: ModelDatasets,
    draw_probability: float,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Build validation and test probabilities for all baselines."""
    validation_rows = datasets.validation.target.shape[0]
    test_rows = datasets.test.target.shape[0]

    validation_home_elo = extract_home_elo_expectation(datasets.validation)
    test_home_elo = extract_home_elo_expectation(datasets.test)

    return {
        "majority": {
            "validation": majority_class_probabilities(
                datasets.train.target,
                row_count=validation_rows,
            ),
            "test": majority_class_probabilities(
                datasets.train.target,
                row_count=test_rows,
            ),
        },
        "frequency": {
            "validation": frequency_probabilities(
                datasets.train.target,
                row_count=validation_rows,
            ),
            "test": frequency_probabilities(
                datasets.train.target,
                row_count=test_rows,
            ),
        },
        "elo": {
            "validation": elo_outcome_probabilities(
                validation_home_elo,
                draw_probability=draw_probability,
            ),
            "test": elo_outcome_probabilities(
                test_home_elo,
                draw_probability=draw_probability,
            ),
        },
    }


def write_baseline_reports(
    *,
    records: list[dict[str, str | float]],
    report_directory: Path,
    draw_probability: float,
    datasets: ModelDatasets,
) -> None:
    """Write baseline comparison reports."""
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = pl.DataFrame(records).sort(["split", "log_loss"])

    comparison.write_parquet(
        report_directory / "baseline_comparison.parquet",
        compression="zstd",
        statistics=True,
    )

    validation_ranking = (
        comparison.filter(pl.col("split") == "validation")
        .sort("log_loss")
        .select("model", "log_loss")
        .to_dicts()
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": ("chronological complete-season evaluation"),
        "training_rows": int(datasets.train.target.shape[0]),
        "validation_rows": int(datasets.validation.target.shape[0]),
        "test_rows": int(datasets.test.target.shape[0]),
        "elo_draw_probability": draw_probability,
        "selection_metric": "validation log_loss",
        "validation_ranking": validation_ranking,
        "metrics": records,
    }

    (report_directory / "baseline_comparison.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def run_baseline_evaluation(
    *,
    dataset_path: Path,
    report_directory: Path,
    prediction_directory: Path,
    draw_probability: float,
    feature_columns: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> pl.DataFrame:
    """Evaluate all baselines on validation and test periods."""
    dataframe = load_model_dataset(dataset_path)

    datasets = build_model_datasets(
        dataframe,
        feature_columns=feature_columns,
    )

    records: list[dict[str, str | float]] = []

    split_mapping = {
        "validation": datasets.validation,
        "test": datasets.test,
    }
    probability_sets = build_baseline_probability_sets(
        datasets,
        draw_probability=draw_probability,
    )

    for model_name, split_probabilities in probability_sets.items():
        for split_name, split in split_mapping.items():
            probabilities = split_probabilities[split_name]

            metrics = evaluate_and_write_predictions(
                model_name=model_name,
                split_name=split_name,
                split=split,
                probabilities=probabilities,
                prediction_directory=prediction_directory,
            )

            records.append(
                metric_record(
                    model_name=model_name,
                    split_name=split_name,
                    metrics=metrics,
                )
            )

    write_baseline_reports(
        records=records,
        report_directory=report_directory,
        draw_probability=draw_probability,
        datasets=datasets,
    )

    return pl.DataFrame(records).sort(["split", "log_loss"])


@app.command()
def run(
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the Gold model dataset.",
        ),
    ] = DEFAULT_MODEL_DATASET,
    report_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for baseline evaluation reports.",
        ),
    ] = DEFAULT_REPORT_DIRECTORY,
    prediction_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for match-level predictions.",
        ),
    ] = DEFAULT_PREDICTION_DIRECTORY,
    draw_probability: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=0.99,
            help=("Fixed draw probability used by the Elo baseline."),
        ),
    ] = DEFAULT_DRAW_PROBABILITY,
) -> None:
    """Run and compare all baseline models."""
    comparison = run_baseline_evaluation(
        dataset_path=dataset_path,
        report_directory=report_directory,
        prediction_directory=prediction_directory,
        draw_probability=draw_probability,
    )

    typer.echo("BASELINE COMPARISON")
    typer.echo("=" * 78)
    typer.echo(str(comparison))


if __name__ == "__main__":
    app()
