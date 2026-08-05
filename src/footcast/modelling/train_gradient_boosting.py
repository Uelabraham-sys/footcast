"""Train, calibrate and evaluate gradient boosting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import joblib
import numpy as np
import polars as pl
import typer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

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
from footcast.modelling.gradient_boosting import (
    DEFAULT_PARAMETER_GRID,
    HGBParameters,
    build_permutation_importance_frame,
    fit_hgb_classifier,
    fit_sigmoid_calibrator,
    ordered_predict_proba,
    validation_record,
)
from footcast.modelling.logistic import (
    fit_logistic_pipeline,
)
from footcast.modelling.logistic import (
    ordered_predict_proba as logistic_predict_proba,
)
from footcast.modelling.metrics import (
    ClassificationMetrics,
    evaluate_probabilities,
)

DEFAULT_DATASET_PATH: Final[Path] = Path("data/gold/model_dataset.parquet")
DEFAULT_MODEL_DIRECTORY: Final[Path] = Path("artifacts/models")
DEFAULT_REPORT_DIRECTORY: Final[Path] = Path("artifacts/reports")
DEFAULT_PREDICTION_DIRECTORY: Final[Path] = Path("artifacts/predictions")
DEFAULT_TUNING_FRACTION: Final[float] = 0.60
DEFAULT_ENSEMBLE_LOGISTIC_C: Final[float] = 0.1

app = typer.Typer(help="Train and calibrate histogram gradient boosting.")


def chronological_validation_partition(
    validation: ModelSplit,
    tuning_fraction: float = DEFAULT_TUNING_FRACTION,
) -> tuple[ModelSplit, ModelSplit]:
    """Split validation chronologically into tuning and calibration."""
    if not 0.5 <= tuning_fraction <= 0.8:
        raise ValueError("tuning_fraction must be between 0.5 and 0.8.")

    row_count = validation.target.shape[0]

    if row_count < 30:
        raise ValueError(
            "Validation requires at least 30 rows for tuning and calibration."
        )

    boundary = int(np.floor(row_count * tuning_fraction))

    if boundary < 1 or boundary >= row_count:
        raise ValueError("Invalid validation partition boundary.")

    tuning = ModelSplit(
        features=validation.features[:boundary],
        target=validation.target[:boundary],
        metadata=validation.metadata.head(boundary),
    )

    calibration = ModelSplit(
        features=validation.features[boundary:],
        target=validation.target[boundary:],
        metadata=validation.metadata.slice(
            boundary,
            row_count - boundary,
        ),
    )

    tuning_max = tuning.metadata["kickoff_utc"].max()
    calibration_min = calibration.metadata["kickoff_utc"].min()

    if not isinstance(tuning_max, datetime) or not isinstance(
        calibration_min, datetime
    ):
        raise TypeError("Validation partition timestamps are invalid.")

    if tuning_max >= calibration_min:
        raise ValueError("Tuning and calibration windows overlap.")

    return tuning, calibration


def select_hgb_parameters(
    datasets: ModelDatasets,
    tuning_split: ModelSplit,
    parameter_grid: tuple[
        HGBParameters,
        ...,
    ] = DEFAULT_PARAMETER_GRID,
) -> tuple[
    HGBParameters,
    pl.DataFrame,
]:
    """Select HGB parameters using tuning-window log loss."""
    if not parameter_grid:
        raise ValueError("Parameter grid cannot be empty.")

    records: list[dict[str, int | float]] = []

    for parameters in parameter_grid:
        model = fit_hgb_classifier(
            features=datasets.train.features,
            target=datasets.train.target,
            parameters=parameters,
        )

        probabilities = ordered_predict_proba(
            model,
            tuning_split.features,
        )

        records.append(
            validation_record(
                parameters,
                tuning_split.target,
                probabilities,
            )
        )

    results = pl.DataFrame(records).sort(
        [
            "log_loss",
            "brier_score",
            "max_leaf_nodes",
            "max_iter",
        ]
    )

    best = results.row(
        0,
        named=True,
    )

    numeric_fields = {
        "learning_rate": best["learning_rate"],
        "max_iter": best["max_iter"],
        "max_leaf_nodes": best["max_leaf_nodes"],
        "min_samples_leaf": best["min_samples_leaf"],
        "l2_regularization": best["l2_regularization"],
    }

    for name, value in numeric_fields.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Selected {name} must be numeric.")

    selected = HGBParameters(
        learning_rate=float(numeric_fields["learning_rate"]),
        max_iter=int(numeric_fields["max_iter"]),
        max_leaf_nodes=int(numeric_fields["max_leaf_nodes"]),
        min_samples_leaf=int(numeric_fields["min_samples_leaf"]),
        l2_regularization=float(numeric_fields["l2_regularization"]),
    )

    return selected, results


def evaluate_model(
    model: HistGradientBoostingClassifier | CalibratedClassifierCV,
    split: ModelSplit,
) -> tuple[
    ClassificationMetrics,
    pl.DataFrame,
]:
    """Evaluate one model on a chronological split."""
    probabilities = ordered_predict_proba(
        model,
        split.features,
    )

    metrics = evaluate_probabilities(
        split.target,
        probabilities,
    )

    predictions = create_prediction_frame(
        split.metadata,
        probabilities,
    )

    return metrics, predictions


def save_model_bundle(
    *,
    path: Path,
    model: HistGradientBoostingClassifier | CalibratedClassifierCV,
    feature_names: tuple[str, ...],
    parameters: HGBParameters,
    calibrated: bool,
) -> None:
    """Persist a model and its required metadata."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "model_family": ("hist_gradient_boosting"),
            "calibrated": calibrated,
            "feature_names": feature_names,
            "class_labels": (0, 1, 2),
            "class_names": (
                "away_win",
                "draw",
                "home_win",
            ),
            "parameters": {
                "learning_rate": (parameters.learning_rate),
                "max_iter": parameters.max_iter,
                "max_leaf_nodes": (parameters.max_leaf_nodes),
                "min_samples_leaf": (parameters.min_samples_leaf),
                "l2_regularization": (parameters.l2_regularization),
            },
            "trained_at": datetime.now(UTC).isoformat(),
        },
        path,
    )


def write_hgb_reports(
    *,
    datasets: ModelDatasets,
    selected_parameters: HGBParameters,
    selection_results: pl.DataFrame,
    importance: pl.DataFrame,
    tuning_metrics: ClassificationMetrics,
    calibration_metrics_uncalibrated: (ClassificationMetrics),
    calibration_metrics_calibrated: (ClassificationMetrics),
    test_metrics_uncalibrated: ClassificationMetrics,
    test_metrics_calibrated: ClassificationMetrics,
    report_directory: Path,
) -> None:
    """Write selection, importance and evaluation reports."""
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_results.write_parquet(
        report_directory / "hgb_selection.parquet",
        compression="zstd",
        statistics=True,
    )

    importance.write_parquet(
        report_directory / "hgb_feature_importance.parquet",
        compression="zstd",
        statistics=True,
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": "hist_gradient_boosting",
        "selection_rule": (
            "lowest log loss on first chronological 60% of validation season"
        ),
        "calibration_strategy": (
            "sigmoid calibration on final chronological "
            "40% of validation season using FrozenEstimator"
        ),
        "training_rows": int(datasets.train.target.shape[0]),
        "validation_rows": int(datasets.validation.target.shape[0]),
        "test_rows": int(datasets.test.target.shape[0]),
        "selected_parameters": {
            "learning_rate": (selected_parameters.learning_rate),
            "max_iter": selected_parameters.max_iter,
            "max_leaf_nodes": (selected_parameters.max_leaf_nodes),
            "min_samples_leaf": (selected_parameters.min_samples_leaf),
            "l2_regularization": (selected_parameters.l2_regularization),
        },
        "tuning_metrics": tuning_metrics.to_dict(),
        "calibration_window_metrics": {
            "uncalibrated": (calibration_metrics_uncalibrated.to_dict()),
            "calibrated": (calibration_metrics_calibrated.to_dict()),
        },
        "test_metrics": {
            "uncalibrated": (test_metrics_uncalibrated.to_dict()),
            "calibrated": (test_metrics_calibrated.to_dict()),
        },
    }

    (report_directory / "hgb_evaluation.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def train_and_evaluate_hgb(
    *,
    dataset_path: Path,
    model_directory: Path,
    report_directory: Path,
    prediction_directory: Path,
    parameter_grid: tuple[
        HGBParameters,
        ...,
    ] = DEFAULT_PARAMETER_GRID,
    feature_columns: tuple[
        str,
        ...,
    ] = MODEL_FEATURE_COLUMNS,
    tuning_fraction: float = DEFAULT_TUNING_FRACTION,
    importance_repeats: int = 20,
) -> pl.DataFrame:
    """Train, calibrate and evaluate the HGB model."""
    dataframe = load_model_dataset(dataset_path)

    datasets = build_model_datasets(
        dataframe,
        feature_columns=feature_columns,
    )

    tuning, calibration = chronological_validation_partition(
        datasets.validation,
        tuning_fraction=tuning_fraction,
    )
    logistic_model = fit_logistic_pipeline(
        features=datasets.train.features,
        target=datasets.train.target,
        regularisation_strength=DEFAULT_ENSEMBLE_LOGISTIC_C,
        class_weight=None,
    )

    logistic_calibration_probabilities = logistic_predict_proba(
        logistic_model,
        calibration.features,
    )

    logistic_calibration_predictions = create_prediction_frame(
        calibration.metadata,
        logistic_calibration_probabilities,
    )

    selected_parameters, selection_results = select_hgb_parameters(
        datasets=datasets,
        tuning_split=tuning,
        parameter_grid=parameter_grid,
    )

    model = fit_hgb_classifier(
        features=datasets.train.features,
        target=datasets.train.target,
        parameters=selected_parameters,
    )

    calibrated_model = fit_sigmoid_calibrator(
        fitted_model=model,
        calibration_features=calibration.features,
        calibration_target=calibration.target,
    )

    tuning_metrics, tuning_predictions = evaluate_model(
        model,
        tuning,
    )

    (
        calibration_metrics_uncalibrated,
        calibration_predictions_uncalibrated,
    ) = evaluate_model(
        model,
        calibration,
    )

    (
        calibration_metrics_calibrated,
        calibration_predictions_calibrated,
    ) = evaluate_model(
        calibrated_model,
        calibration,
    )

    (
        test_metrics_uncalibrated,
        test_predictions_uncalibrated,
    ) = evaluate_model(
        model,
        datasets.test,
    )

    (
        test_metrics_calibrated,
        test_predictions_calibrated,
    ) = evaluate_model(
        calibrated_model,
        datasets.test,
    )

    importance = build_permutation_importance_frame(
        model=model,
        features=tuning.features,
        target=tuning.target,
        feature_names=datasets.feature_names,
        n_repeats=importance_repeats,
    )

    save_model_bundle(
        path=model_directory / "hist_gradient_boosting.joblib",
        model=model,
        feature_names=datasets.feature_names,
        parameters=selected_parameters,
        calibrated=False,
    )

    save_model_bundle(
        path=model_directory / "hist_gradient_boosting_calibrated.joblib",
        model=calibrated_model,
        feature_names=datasets.feature_names,
        parameters=selected_parameters,
        calibrated=True,
    )

    prediction_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_outputs = {
        "hgb_validation_tuning.parquet": (tuning_predictions),
        "hgb_calibration_uncalibrated.parquet": (calibration_predictions_uncalibrated),
        "hgb_calibration.parquet": (calibration_predictions_calibrated),
        "logistic_calibration.parquet": (logistic_calibration_predictions),
        "hgb_test.parquet": (test_predictions_uncalibrated),
        "hgb_calibrated_test.parquet": (test_predictions_calibrated),
    }

    for filename, predictions in prediction_outputs.items():
        write_prediction_frame(
            predictions,
            prediction_directory / filename,
        )

    write_hgb_reports(
        datasets=datasets,
        selected_parameters=selected_parameters,
        selection_results=selection_results,
        importance=importance,
        tuning_metrics=tuning_metrics,
        calibration_metrics_uncalibrated=(calibration_metrics_uncalibrated),
        calibration_metrics_calibrated=(calibration_metrics_calibrated),
        test_metrics_uncalibrated=(test_metrics_uncalibrated),
        test_metrics_calibrated=(test_metrics_calibrated),
        report_directory=report_directory,
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
    model_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for fitted models.",
        ),
    ] = DEFAULT_MODEL_DIRECTORY,
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
    tuning_fraction: Annotated[
        float,
        typer.Option(
            min=0.5,
            max=0.8,
            help=("Fraction of validation used for hyperparameter tuning."),
        ),
    ] = DEFAULT_TUNING_FRACTION,
) -> None:
    """Train and evaluate histogram gradient boosting."""
    results = train_and_evaluate_hgb(
        dataset_path=dataset_path,
        model_directory=model_directory,
        report_directory=report_directory,
        prediction_directory=prediction_directory,
        tuning_fraction=tuning_fraction,
    )

    typer.echo("HGB PARAMETER SELECTION")
    typer.echo("=" * 78)
    typer.echo(str(results))


if __name__ == "__main__":
    app()
