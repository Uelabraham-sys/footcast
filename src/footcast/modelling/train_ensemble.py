"""Select and evaluate a FootCast probability ensemble."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import numpy as np
import polars as pl
import typer

from footcast.modelling.ensemble import (
    DEFAULT_WEIGHTS,
    align_prediction_frames,
    blend_probabilities,
    create_ensemble_prediction_frame,
    extract_aligned_arrays,
)
from footcast.modelling.metrics import (
    ClassificationMetrics,
    evaluate_probabilities,
)

DEFAULT_PREDICTION_DIRECTORY: Final[Path] = Path("artifacts/predictions")

DEFAULT_REPORT_DIRECTORY: Final[Path] = Path("artifacts/reports")

DEFAULT_FIRST_MODEL: Final[str] = "logistic_regression"

DEFAULT_SECOND_MODEL: Final[str] = "hist_gradient_boosting_calibrated"

app = typer.Typer(help="Select and evaluate a probability ensemble.")


def parse_weights(
    value: str,
) -> tuple[float, ...]:
    """Parse comma-separated ensemble weights."""
    try:
        weights = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise ValueError("Weights must be comma-separated numbers.") from error

    if not weights:
        raise ValueError("At least one ensemble weight is required.")

    for weight in weights:
        if not np.isfinite(weight):
            raise ValueError("Every ensemble weight must be finite.")

        if not 0.0 <= weight <= 1.0:
            raise ValueError("Every ensemble weight must be between zero and one.")

    return tuple(sorted(set(weights)))


def metric_record(
    *,
    first_weight: float,
    metrics: ClassificationMetrics,
) -> dict[str, float]:
    """Create one blend-selection row."""
    return {
        "first_model_weight": first_weight,
        "second_model_weight": (1.0 - first_weight),
        **metrics.to_dict(),
    }


def select_ensemble_weight(
    validation_aligned: pl.DataFrame,
    *,
    candidate_weights: tuple[
        float,
        ...,
    ] = DEFAULT_WEIGHTS,
) -> tuple[
    float,
    pl.DataFrame,
]:
    """Select blend weight using validation log loss."""
    if not candidate_weights:
        raise ValueError("Candidate weights cannot be empty.")

    target, first, second = extract_aligned_arrays(validation_aligned)

    records: list[dict[str, float]] = []

    for first_weight in candidate_weights:
        probabilities = blend_probabilities(
            first,
            second,
            first_weight=first_weight,
        )

        metrics = evaluate_probabilities(
            target,
            probabilities,
        )

        records.append(
            metric_record(
                first_weight=first_weight,
                metrics=metrics,
            )
        )

    selection = pl.DataFrame(records).sort(
        [
            "log_loss",
            "brier_score",
            "first_model_weight",
        ]
    )

    best_weight = selection["first_model_weight"].item(0)

    if not isinstance(
        best_weight,
        (int, float),
    ):
        raise TypeError("Selected ensemble weight must be numeric.")

    return float(best_weight), selection


def evaluate_fixed_ensemble(
    aligned: pl.DataFrame,
    *,
    first_weight: float,
    first_model: str,
    second_model: str,
) -> tuple[
    ClassificationMetrics,
    pl.DataFrame,
]:
    """Evaluate a fixed ensemble on aligned predictions."""
    target, first, second = extract_aligned_arrays(aligned)

    probabilities = blend_probabilities(
        first,
        second,
        first_weight=first_weight,
    )

    metrics = evaluate_probabilities(
        target,
        probabilities,
    )

    predictions = create_ensemble_prediction_frame(
        aligned,
        probabilities,
        first_weight=first_weight,
        first_model=first_model,
        second_model=second_model,
    )

    return metrics, predictions


def write_ensemble_reports(
    *,
    selection: pl.DataFrame,
    selected_weight: float,
    validation_metrics: ClassificationMetrics,
    test_metrics: ClassificationMetrics,
    validation_predictions: pl.DataFrame,
    test_predictions: pl.DataFrame,
    report_directory: Path,
    prediction_directory: Path,
    first_model: str,
    second_model: str,
) -> None:
    """Persist ensemble reports and predictions."""
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection.write_parquet(
        report_directory / "ensemble_selection.parquet",
        compression="zstd",
        statistics=True,
    )

    validation_predictions.write_parquet(
        prediction_directory / "ensemble_validation.parquet",
        compression="zstd",
        statistics=True,
    )

    test_predictions.write_parquet(
        prediction_directory / "ensemble_test.parquet",
        compression="zstd",
        statistics=True,
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": "probability_ensemble",
        "selection_rule": (
            "lowest log loss on shared chronological calibration window"
        ),
        "first_model": first_model,
        "second_model": second_model,
        "selected_weights": {
            first_model: selected_weight,
            second_model: 1.0 - selected_weight,
        },
        "candidate_weight_count": (selection.height),
        "validation_metrics": (validation_metrics.to_dict()),
        "test_metrics": test_metrics.to_dict(),
    }

    (report_directory / "ensemble_evaluation.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def train_and_evaluate_ensemble(
    *,
    prediction_directory: Path,
    report_directory: Path,
    candidate_weights: tuple[
        float,
        ...,
    ] = DEFAULT_WEIGHTS,
    first_model: str = DEFAULT_FIRST_MODEL,
    second_model: str = DEFAULT_SECOND_MODEL,
) -> pl.DataFrame:
    """Select the blend and evaluate it on test data."""
    validation_first_path = prediction_directory / "logistic_calibration.parquet"

    validation_second_path = prediction_directory / "hgb_calibration.parquet"

    test_first_path = prediction_directory / "logistic_test.parquet"

    test_second_path = prediction_directory / "hgb_calibrated_test.parquet"

    required_paths = (
        validation_first_path,
        validation_second_path,
        test_first_path,
        test_second_path,
    )

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        formatted = ", ".join(str(path) for path in missing_paths)

        raise FileNotFoundError(
            "Required ensemble predictions are missing: "
            f"{formatted}. Run `make train-hgb` and "
            "`make train-logistic` first."
        )

    validation_first = pl.read_parquet(validation_first_path)

    validation_second = pl.read_parquet(validation_second_path)

    test_first = pl.read_parquet(test_first_path)

    test_second = pl.read_parquet(test_second_path)

    validation_aligned = align_prediction_frames(
        validation_first,
        validation_second,
        first_name=first_model,
        second_name=second_model,
    )

    test_aligned = align_prediction_frames(
        test_first,
        test_second,
        first_name=first_model,
        second_name=second_model,
    )

    selected_weight, selection = select_ensemble_weight(
        validation_aligned,
        candidate_weights=(candidate_weights),
    )

    (
        validation_metrics,
        validation_predictions,
    ) = evaluate_fixed_ensemble(
        validation_aligned,
        first_weight=selected_weight,
        first_model=first_model,
        second_model=second_model,
    )

    test_metrics, test_predictions = evaluate_fixed_ensemble(
        test_aligned,
        first_weight=selected_weight,
        first_model=first_model,
        second_model=second_model,
    )

    write_ensemble_reports(
        selection=selection,
        selected_weight=selected_weight,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_predictions=(validation_predictions),
        test_predictions=test_predictions,
        report_directory=report_directory,
        prediction_directory=(prediction_directory),
        first_model=first_model,
        second_model=second_model,
    )

    return selection


@app.command()
def run(
    prediction_directory: Annotated[
        Path,
        typer.Option(
            help="Directory containing component predictions.",
        ),
    ] = DEFAULT_PREDICTION_DIRECTORY,
    report_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for ensemble reports.",
        ),
    ] = DEFAULT_REPORT_DIRECTORY,
    weights: Annotated[
        str,
        typer.Option(
            help=("Comma-separated weights for the first model."),
        ),
    ] = ",".join(str(value) for value in DEFAULT_WEIGHTS),
) -> None:
    """Select and evaluate the probability ensemble."""
    candidate_weights = parse_weights(weights)

    selection = train_and_evaluate_ensemble(
        prediction_directory=(prediction_directory),
        report_directory=report_directory,
        candidate_weights=candidate_weights,
    )

    typer.echo("ENSEMBLE SELECTION")
    typer.echo("=" * 78)
    typer.echo(str(selection))


if __name__ == "__main__":
    app()
