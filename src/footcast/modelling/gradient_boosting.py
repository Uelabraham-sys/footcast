"""Histogram gradient-boosting utilities for FootCast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch

from footcast.modelling.dataset import CLASS_LABELS
from footcast.modelling.metrics import evaluate_probabilities

RANDOM_STATE: Final[int] = 42


@dataclass(frozen=True)
class HGBParameters:
    """Hyperparameters for histogram gradient boosting."""

    learning_rate: float = 0.05
    max_iter: int = 200
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0


DEFAULT_PARAMETER_GRID: Final[tuple[HGBParameters, ...]] = (
    HGBParameters(
        learning_rate=0.03,
        max_iter=200,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
    ),
    HGBParameters(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
    ),
    HGBParameters(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
    ),
    HGBParameters(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=5.0,
    ),
    HGBParameters(
        learning_rate=0.03,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=5.0,
    ),
    HGBParameters(
        learning_rate=0.02,
        max_iter=500,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=10.0,
    ),
)


def validate_hgb_parameters(
    parameters: HGBParameters,
) -> None:
    """Validate gradient-boosting hyperparameters."""
    if parameters.learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than zero.")

    if parameters.max_iter < 1:
        raise ValueError("max_iter must be at least one.")

    if parameters.max_leaf_nodes < 2:
        raise ValueError("max_leaf_nodes must be at least two.")

    if parameters.min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least one.")

    if parameters.l2_regularization < 0.0:
        raise ValueError("l2_regularization cannot be negative.")


def create_hgb_classifier(
    parameters: HGBParameters,
) -> HistGradientBoostingClassifier:
    """Create a deterministic multiclass HGB classifier."""
    validate_hgb_parameters(parameters)

    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=parameters.learning_rate,
        max_iter=parameters.max_iter,
        max_leaf_nodes=parameters.max_leaf_nodes,
        min_samples_leaf=parameters.min_samples_leaf,
        l2_regularization=parameters.l2_regularization,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def validate_training_arrays(
    features: NDArray[np.float64],
    target: NDArray[np.int64],
) -> None:
    """Validate arrays used to fit a classifier."""
    if features.ndim != 2:
        raise ValueError("Features must be two-dimensional.")

    if target.ndim != 1:
        raise ValueError("Target must be one-dimensional.")

    if features.shape[0] != target.shape[0]:
        raise ValueError("Feature and target row counts differ.")

    if features.shape[0] == 0:
        raise ValueError("Training data cannot be empty.")

    if not np.isfinite(features).all():
        raise ValueError("Training features contain non-finite values.")

    observed_classes = {int(value) for value in np.unique(target).tolist()}

    if observed_classes != set(CLASS_LABELS):
        raise ValueError("Training target must contain classes 0, 1 and 2.")


def fit_hgb_classifier(
    features: NDArray[np.float64],
    target: NDArray[np.int64],
    parameters: HGBParameters,
) -> HistGradientBoostingClassifier:
    """Fit a histogram gradient-boosting classifier."""
    validate_training_arrays(features, target)

    model = create_hgb_classifier(parameters)
    fitted = model.fit(features, target)

    if not isinstance(
        fitted,
        HistGradientBoostingClassifier,
    ):
        raise TypeError("HGB fitting did not return the expected estimator.")

    return fitted


def ordered_predict_proba(
    model: HistGradientBoostingClassifier | CalibratedClassifierCV,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return probabilities ordered as away, draw and home."""
    if features.ndim != 2:
        raise ValueError("Prediction features must be two-dimensional.")

    probabilities = np.asarray(
        model.predict_proba(features),
        dtype=np.float64,
    )

    classes = np.asarray(
        model.classes_,
        dtype=np.int64,
    )

    class_positions = {
        int(class_label): position
        for position, class_label in enumerate(classes.tolist())
    }

    missing = sorted(set(CLASS_LABELS) - set(class_positions))

    if missing:
        raise ValueError(f"Model is missing classes: {missing}")

    ordered = np.column_stack(
        [
            probabilities[
                :,
                class_positions[class_label],
            ]
            for class_label in CLASS_LABELS
        ]
    ).astype(np.float64)

    row_sums = ordered.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(row_sums <= 0.0):
        raise ValueError("Probability rows must have positive mass.")

    return (ordered / row_sums).astype(np.float64)


def fit_sigmoid_calibrator(
    fitted_model: HistGradientBoostingClassifier,
    calibration_features: NDArray[np.float64],
    calibration_target: NDArray[np.int64],
) -> CalibratedClassifierCV:
    """Fit sigmoid calibration on data disjoint from model fitting."""
    validate_training_arrays(
        calibration_features,
        calibration_target,
    )

    frozen_model = FrozenEstimator(fitted_model)

    calibrator = CalibratedClassifierCV(
        estimator=frozen_model,
        method="sigmoid",
    )

    fitted_calibrator = calibrator.fit(
        calibration_features,
        calibration_target,
    )

    if not isinstance(
        fitted_calibrator,
        CalibratedClassifierCV,
    ):
        raise TypeError("Calibration did not return the expected estimator.")

    return fitted_calibrator


def build_permutation_importance_frame(
    model: HistGradientBoostingClassifier | CalibratedClassifierCV,
    features: NDArray[np.float64],
    target: NDArray[np.int64],
    feature_names: tuple[str, ...],
    *,
    n_repeats: int = 20,
) -> pl.DataFrame:
    """Calculate validation permutation feature importance."""
    if features.shape[1] != len(feature_names):
        raise ValueError("Feature width does not match feature names.")

    result = permutation_importance(
        estimator=model,
        X=features,
        y=target,
        scoring="neg_log_loss",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if not isinstance(result, Bunch):
        raise TypeError("Expected a single permutation-importance result.")

    importance_mean = np.asarray(
        result.importances_mean,
        dtype=np.float64,
    )
    importance_std = np.asarray(
        result.importances_std,
        dtype=np.float64,
    )

    return (
        pl.DataFrame(
            {
                "feature": list(feature_names),
                "importance_mean": importance_mean,
                "importance_std": importance_std,
            }
        )
        .with_columns(pl.col("importance_mean").abs().alias("absolute_importance"))
        .sort(
            "importance_mean",
            descending=True,
        )
    )


def validation_record(
    parameters: HGBParameters,
    target: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> dict[str, int | float]:
    """Create one hyperparameter-selection record."""
    metrics = evaluate_probabilities(
        target,
        probabilities,
    )

    return {
        "learning_rate": parameters.learning_rate,
        "max_iter": parameters.max_iter,
        "max_leaf_nodes": parameters.max_leaf_nodes,
        "min_samples_leaf": parameters.min_samples_leaf,
        "l2_regularization": (parameters.l2_regularization),
        **metrics.to_dict(),
    }
