"""Train and evaluate FootCast logistic regression."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import joblib
import polars as pl
import typer
from sklearn.pipeline import Pipeline

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
from footcast.modelling.logistic import (
    DEFAULT_C_VALUES,
    extract_logistic_coefficients,
    fit_logistic_pipeline,
    ordered_predict_proba,
)
from footcast.modelling.metrics import (
    ClassificationMetrics,
    evaluate_probabilities,
)

DEFAULT_DATASET_PATH: Final[Path] = Path("data/gold/model_dataset.parquet")
DEFAULT_MODEL_PATH: Final[Path] = Path("artifacts/models/logistic_regression.joblib")
DEFAULT_REPORT_DIRECTORY: Final[Path] = Path("artifacts/reports")
DEFAULT_PREDICTION_DIRECTORY: Final[Path] = Path("artifacts/predictions")

app = typer.Typer(help="Train and evaluate multinomial logistic regression.")


def parse_c_values(
    value: str,
) -> tuple[float, ...]:
    """Parse comma-separated inverse regularisation values."""
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("C values must be comma-separated numbers.") from error

    if not values:
        raise ValueError("At least one C value is required.")

    if any(item <= 0.0 for item in values):
        raise ValueError("Every C value must be greater than zero.")

    return values


def evaluate_model_on_split(
    model: Pipeline,
    split: ModelSplit,
) -> tuple[
    ClassificationMetrics,
    pl.DataFrame,
]:
    """Evaluate a fitted model on one chronological split."""
    probabilities = ordered_predict_proba(
        model,
        split.features,
    )

    metrics = evaluate_probabilities(
        target=split.target,
        probabilities=probabilities,
    )

    predictions = create_prediction_frame(
        metadata=split.metadata,
        probabilities=probabilities,
    )

    return metrics, predictions


def selection_record(
    *,
    regularisation_strength: float,
    class_weight: str | None,
    metrics: ClassificationMetrics,
) -> dict[str, str | float]:
    """Create one validation-selection result."""
    return {
        "regularisation_strength": (regularisation_strength),
        "class_weight": (class_weight if class_weight is not None else "none"),
        **metrics.to_dict(),
    }


def select_logistic_configuration(
    datasets: ModelDatasets,
    c_values: tuple[float, ...],
    class_weights: tuple[str | None, ...] = (
        None,
        "balanced",
    ),
) -> tuple[
    float,
    str | None,
    pl.DataFrame,
]:
    """Select configuration using validation log loss only."""
    records: list[dict[str, str | float]] = []

    for class_weight in class_weights:
        for regularisation_strength in c_values:
            model = fit_logistic_pipeline(
                features=datasets.train.features,
                target=datasets.train.target,
                regularisation_strength=(regularisation_strength),
                class_weight=class_weight,
            )

            validation_probabilities = ordered_predict_proba(
                model,
                datasets.validation.features,
            )

            metrics = evaluate_probabilities(
                target=datasets.validation.target,
                probabilities=validation_probabilities,
            )

            records.append(
                selection_record(
                    regularisation_strength=(regularisation_strength),
                    class_weight=class_weight,
                    metrics=metrics,
                )
            )

    results = pl.DataFrame(records).sort(
        [
            "log_loss",
            "brier_score",
            "regularisation_strength",
        ]
    )

    best = results.row(
        0,
        named=True,
    )

    best_c_value = best["regularisation_strength"]
    best_class_weight = best["class_weight"]

    if not isinstance(
        best_c_value,
        (int, float),
    ):
        raise TypeError("Selected C value must be numeric.")

    if not isinstance(best_class_weight, str):
        raise TypeError("Selected class weight must be a string.")

    return (
        float(best_c_value),
        (None if best_class_weight == "none" else best_class_weight),
        results,
    )


def fit_final_logistic_model(
    datasets: ModelDatasets,
    regularisation_strength: float,
    class_weight: str | None,
) -> Pipeline:
    """Fit the selected model using training data only."""
    return fit_logistic_pipeline(
        features=datasets.train.features,
        target=datasets.train.target,
        regularisation_strength=(regularisation_strength),
        class_weight=class_weight,
    )


def write_logistic_artifacts(
    *,
    model: Pipeline,
    datasets: ModelDatasets,
    best_c_value: float,
    best_class_weight: str | None,
    selection_results: pl.DataFrame,
    validation_metrics: ClassificationMetrics,
    test_metrics: ClassificationMetrics,
    validation_predictions: pl.DataFrame,
    test_predictions: pl.DataFrame,
    model_path: Path,
    report_directory: Path,
    prediction_directory: Path,
) -> None:
    """Persist the model, predictions and evaluation reports."""
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    prediction_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_names": datasets.feature_names,
            "class_labels": (0, 1, 2),
            "class_names": (
                "away_win",
                "draw",
                "home_win",
            ),
            "regularisation_strength": best_c_value,
            "class_weight": best_class_weight,
            "trained_at": datetime.now(UTC).isoformat(),
        },
        model_path,
    )

    selection_results.write_parquet(
        report_directory / "logistic_selection.parquet",
        compression="zstd",
        statistics=True,
    )

    coefficient_frame = extract_logistic_coefficients(
        model,
        datasets.feature_names,
    )

    coefficient_frame.write_parquet(
        report_directory / "logistic_coefficients.parquet",
        compression="zstd",
        statistics=True,
    )

    write_prediction_frame(
        validation_predictions,
        prediction_directory / "logistic_validation.parquet",
    )

    write_prediction_frame(
        test_predictions,
        prediction_directory / "logistic_test.parquet",
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": "multinomial_logistic_regression",
        "selection_rule": (
            "lowest validation log loss; test excluded from model selection"
        ),
        "training_rows": int(datasets.train.target.shape[0]),
        "validation_rows": int(datasets.validation.target.shape[0]),
        "test_rows": int(datasets.test.target.shape[0]),
        "feature_count": len(datasets.feature_names),
        "feature_names": list(datasets.feature_names),
        "selected_parameters": {
            "regularisation_strength": best_c_value,
            "class_weight": (
                best_class_weight if best_class_weight is not None else "none"
            ),
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 5_000,
        },
        "validation_metrics": (validation_metrics.to_dict()),
        "test_metrics": test_metrics.to_dict(),
    }

    (report_directory / "logistic_regression.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def train_and_evaluate_logistic(
    *,
    dataset_path: Path,
    model_path: Path,
    report_directory: Path,
    prediction_directory: Path,
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
    feature_columns: tuple[
        str,
        ...,
    ] = MODEL_FEATURE_COLUMNS,
) -> pl.DataFrame:
    """Select, train and evaluate logistic regression."""
    dataframe = load_model_dataset(dataset_path)

    datasets = build_model_datasets(
        dataframe,
        feature_columns=feature_columns,
    )

    (
        best_c_value,
        best_class_weight,
        selection_results,
    ) = select_logistic_configuration(
        datasets=datasets,
        c_values=c_values,
    )

    model = fit_final_logistic_model(
        datasets=datasets,
        regularisation_strength=best_c_value,
        class_weight=best_class_weight,
    )

    (
        validation_metrics,
        validation_predictions,
    ) = evaluate_model_on_split(
        model,
        datasets.validation,
    )

    test_metrics, test_predictions = evaluate_model_on_split(
        model,
        datasets.test,
    )

    write_logistic_artifacts(
        model=model,
        datasets=datasets,
        best_c_value=best_c_value,
        best_class_weight=best_class_weight,
        selection_results=selection_results,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        model_path=model_path,
        report_directory=report_directory,
        prediction_directory=prediction_directory,
    )

    return selection_results


@app.command()
def run(
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the Gold model dataset.",
        ),
    ] = DEFAULT_DATASET_PATH,
    model_path: Annotated[
        Path,
        typer.Option(
            help="Path for the fitted model artifact.",
        ),
    ] = DEFAULT_MODEL_PATH,
    report_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for evaluation reports.",
        ),
    ] = DEFAULT_REPORT_DIRECTORY,
    prediction_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for match predictions.",
        ),
    ] = DEFAULT_PREDICTION_DIRECTORY,
    c_values: Annotated[
        str,
        typer.Option(
            help=("Comma-separated inverse regularisation strengths."),
        ),
    ] = ",".join(str(value) for value in DEFAULT_C_VALUES),
) -> None:
    """Train and evaluate logistic regression."""
    parsed_c_values = parse_c_values(c_values)

    selection_results = train_and_evaluate_logistic(
        dataset_path=dataset_path,
        model_path=model_path,
        report_directory=report_directory,
        prediction_directory=prediction_directory,
        c_values=parsed_c_values,
    )

    typer.echo("LOGISTIC REGRESSION SELECTION")
    typer.echo("=" * 78)
    typer.echo(str(selection_results))


if __name__ == "__main__":
    app()
