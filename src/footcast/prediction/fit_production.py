"""Fit and persist the final FootCast production model."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import numpy as np
import polars as pl
import typer

from footcast.modelling.dataset import (
    MODEL_FEATURE_COLUMNS,
    load_model_dataset,
    prepare_feature_frame,
)
from footcast.modelling.gradient_boosting import (
    HGBParameters,
    fit_hgb_classifier,
    fit_sigmoid_calibrator,
)
from footcast.modelling.logistic import (
    fit_logistic_pipeline,
)
from footcast.prediction.bundle import (
    create_production_bundle,
    feature_contract_hash,
    save_production_bundle,
)

DEFAULT_DATASET_PATH: Final[Path] = Path("data/gold/model_dataset.parquet")

DEFAULT_ENSEMBLE_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/ensemble_evaluation.json"
)

DEFAULT_HGB_REPORT_PATH: Final[Path] = Path("artifacts/reports/hgb_evaluation.json")

DEFAULT_LOGISTIC_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/logistic_regression.json"
)

DEFAULT_BUNDLE_PATH: Final[Path] = Path(
    "artifacts/models/production/footcast_bundle.joblib"
)

DEFAULT_MANIFEST_PATH: Final[Path] = Path("artifacts/models/production/manifest.json")

DEFAULT_TRAINING_REPORT_PATH: Final[Path] = Path(
    "artifacts/reports/production_training.json"
)

DEFAULT_CALIBRATION_FRACTION: Final[float] = 0.15

app = typer.Typer(help="Fit the final FootCast production model.")


def current_git_commit() -> str | None:
    """Return the current Git commit when available."""
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    """Load one required JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"Required report was not found: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")

    return value


def extract_selected_weights(
    report: dict[str, Any],
) -> tuple[float, float]:
    """Extract logistic and HGB weights from ensemble report."""
    selected = report.get("selected_weights")

    if not isinstance(selected, dict):
        raise TypeError("Ensemble report is missing selected_weights.")

    logistic = selected.get("logistic_regression")

    hgb = selected.get("hist_gradient_boosting_calibrated")

    if not isinstance(logistic, (int, float)):
        raise TypeError("Logistic ensemble weight must be numeric.")

    if not isinstance(hgb, (int, float)):
        raise TypeError("HGB ensemble weight must be numeric.")

    return float(logistic), float(hgb)


def extract_logistic_configuration(
    report: dict[str, Any],
) -> tuple[float, str | None]:
    """Extract selected logistic configuration."""
    selected = report.get("selected_parameters")

    if not isinstance(selected, dict):
        raise TypeError("Logistic report is missing selected_parameters.")

    c_value = selected.get("regularisation_strength")

    class_weight = selected.get("class_weight")

    if not isinstance(c_value, (int, float)):
        raise TypeError("Logistic regularisation strength must be numeric.")

    if class_weight == "none":
        parsed_class_weight: str | None = None
    elif class_weight == "balanced":
        parsed_class_weight = "balanced"
    else:
        raise ValueError("Unsupported logistic class weight.")

    return float(c_value), parsed_class_weight


def extract_hgb_configuration(
    report: dict[str, Any],
) -> HGBParameters:
    """Extract selected HGB hyperparameters."""
    selected = report.get("selected_parameters")

    if not isinstance(selected, dict):
        raise TypeError("HGB report is missing selected_parameters.")

    required = (
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    )

    missing = [name for name in required if name not in selected]

    if missing:
        raise ValueError(f"HGB report is missing parameters: {missing}")

    return HGBParameters(
        learning_rate=float(selected["learning_rate"]),
        max_iter=int(selected["max_iter"]),
        max_leaf_nodes=int(selected["max_leaf_nodes"]),
        min_samples_leaf=int(selected["min_samples_leaf"]),
        l2_regularization=float(selected["l2_regularization"]),
    )


def chronological_calibration_partition(
    labelled: pl.DataFrame,
    *,
    calibration_fraction: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split labelled data into disjoint chronological windows."""
    if not 0.05 <= calibration_fraction <= 0.30:
        raise ValueError("calibration_fraction must be between 0.05 and 0.30.")

    ordered = labelled.sort(
        [
            "kickoff_utc",
            "match_key",
        ]
    )

    row_count = ordered.height

    if row_count < 100:
        raise ValueError(
            "At least 100 labelled rows are required for production fitting."
        )

    requested_calibration_rows = max(
        30,
        int(np.ceil(row_count * calibration_fraction)),
    )

    if requested_calibration_rows >= row_count:
        raise ValueError("Calibration window consumes all labelled rows.")

    provisional_boundary = row_count - requested_calibration_rows

    boundary_timestamp = ordered["kickoff_utc"].item(provisional_boundary)

    if not isinstance(
        boundary_timestamp,
        datetime,
    ):
        raise TypeError("Production calibration boundary timestamp is invalid.")

    model_fit = ordered.filter(pl.col("kickoff_utc") < boundary_timestamp)

    calibration = ordered.filter(pl.col("kickoff_utc") >= boundary_timestamp)

    if model_fit.is_empty():
        raise ValueError("Production model-fit window is empty.")

    if calibration.is_empty():
        raise ValueError("Production calibration window is empty.")

    if calibration.height < 30:
        raise ValueError("Production calibration window requires at least 30 rows.")

    fit_max = model_fit["kickoff_utc"].max()

    calibration_min = calibration["kickoff_utc"].min()

    if not isinstance(fit_max, datetime):
        raise TypeError("Production model-fit maximum timestamp is invalid.")

    if not isinstance(
        calibration_min,
        datetime,
    ):
        raise TypeError("Production calibration minimum timestamp is invalid.")

    if fit_max >= calibration_min:
        raise ValueError("Production calibration window overlaps model fitting.")

    return model_fit, calibration


def extract_training_arrays(
    dataframe: pl.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Extract model-ready features and target."""
    features = np.asarray(
        prepare_feature_frame(
            dataframe,
            feature_columns,
        ).to_numpy(),
        dtype=np.float64,
    )

    target = np.asarray(
        dataframe["target"].to_numpy(),
        dtype=np.int64,
    )

    return features, target


def fit_production_model(
    *,
    dataset_path: Path,
    ensemble_report_path: Path,
    hgb_report_path: Path,
    logistic_report_path: Path,
    bundle_path: Path,
    manifest_path: Path,
    training_report_path: Path,
    feature_columns: tuple[
        str,
        ...,
    ] = MODEL_FEATURE_COLUMNS,
    calibration_fraction: float = (DEFAULT_CALIBRATION_FRACTION),
) -> None:
    """Fit and save the complete production bundle."""
    dataframe = load_model_dataset(dataset_path)

    labelled = dataframe.filter(pl.col("target").is_not_null())

    if labelled.is_empty():
        raise ValueError("No labelled rows are available for production training.")

    ensemble_report = load_json_object(ensemble_report_path)

    hgb_report = load_json_object(hgb_report_path)

    logistic_report = load_json_object(logistic_report_path)

    logistic_weight, hgb_weight = extract_selected_weights(ensemble_report)

    logistic_c, logistic_class_weight = extract_logistic_configuration(logistic_report)

    hgb_parameters = extract_hgb_configuration(hgb_report)

    model_fit, calibration = chronological_calibration_partition(
        labelled,
        calibration_fraction=calibration_fraction,
    )

    all_features, all_target = extract_training_arrays(
        labelled,
        feature_columns,
    )

    fit_features, fit_target = extract_training_arrays(
        model_fit,
        feature_columns,
    )

    calibration_features, calibration_target = extract_training_arrays(
        calibration,
        feature_columns,
    )

    logistic_model = fit_logistic_pipeline(
        features=all_features,
        target=all_target,
        regularisation_strength=logistic_c,
        class_weight=logistic_class_weight,
    )

    hgb_model = fit_hgb_classifier(
        features=fit_features,
        target=fit_target,
        parameters=hgb_parameters,
    )

    calibrated_hgb_model = fit_sigmoid_calibrator(
        fitted_model=hgb_model,
        calibration_features=calibration_features,
        calibration_target=calibration_target,
    )

    training_cutoff = labelled["kickoff_utc"].max()

    metadata: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "labelled_rows": labelled.height,
        "hgb_fit_rows": model_fit.height,
        "calibration_rows": calibration.height,
        "training_cutoff": str(training_cutoff),
        "logistic_configuration": {
            "regularisation_strength": logistic_c,
            "class_weight": (
                logistic_class_weight if logistic_class_weight is not None else "none"
            ),
        },
        "hgb_configuration": {
            "learning_rate": (hgb_parameters.learning_rate),
            "max_iter": hgb_parameters.max_iter,
            "max_leaf_nodes": (hgb_parameters.max_leaf_nodes),
            "min_samples_leaf": (hgb_parameters.min_samples_leaf),
            "l2_regularization": (hgb_parameters.l2_regularization),
        },
    }

    bundle = create_production_bundle(
        logistic_model=logistic_model,
        calibrated_hgb_model=calibrated_hgb_model,
        feature_names=feature_columns,
        logistic_weight=logistic_weight,
        hgb_weight=hgb_weight,
        metadata=metadata,
    )

    save_production_bundle(
        bundle,
        bundle_path,
    )

    bundle_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "schema_version": bundle.schema_version,
        "created_at": bundle.created_at,
        "bundle_path": str(bundle_path),
        "git_commit": current_git_commit(),
        "feature_count": len(feature_columns),
        "feature_names": list(feature_columns),
        "feature_contract_sha256": feature_contract_hash(feature_columns),
        "class_labels": list(bundle.class_labels),
        "class_names": list(bundle.class_names),
        "ensemble_weights": {
            "logistic_regression": logistic_weight,
            "hist_gradient_boosting_calibrated": hgb_weight,
        },
        "training_cutoff": str(training_cutoff),
        "source_reports": {
            "ensemble": str(ensemble_report_path),
            "logistic": str(logistic_report_path),
            "hgb": str(hgb_report_path),
        },
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    training_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "success",
        **manifest,
        "training": metadata,
    }

    training_report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


@app.command()
def run(
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the Gold model dataset.",
        ),
    ] = DEFAULT_DATASET_PATH,
    bundle_path: Annotated[
        Path,
        typer.Option(
            help="Path for the production bundle.",
        ),
    ] = DEFAULT_BUNDLE_PATH,
    calibration_fraction: Annotated[
        float,
        typer.Option(
            min=0.05,
            max=0.30,
            help=("Fraction of recent labelled rows reserved for HGB calibration."),
        ),
    ] = DEFAULT_CALIBRATION_FRACTION,
) -> None:
    """Fit and persist the production model."""
    fit_production_model(
        dataset_path=dataset_path,
        ensemble_report_path=(DEFAULT_ENSEMBLE_REPORT_PATH),
        hgb_report_path=(DEFAULT_HGB_REPORT_PATH),
        logistic_report_path=(DEFAULT_LOGISTIC_REPORT_PATH),
        bundle_path=bundle_path,
        manifest_path=DEFAULT_MANIFEST_PATH,
        training_report_path=(DEFAULT_TRAINING_REPORT_PATH),
        calibration_fraction=(calibration_fraction),
    )

    typer.echo(f"Production bundle written to {bundle_path}")


if __name__ == "__main__":
    app()
