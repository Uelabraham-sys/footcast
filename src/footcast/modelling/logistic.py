"""Multinomial logistic-regression utilities for FootCast."""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from footcast.modelling.dataset import CLASS_LABELS

DEFAULT_C_VALUES: Final[tuple[float, ...]] = (
    0.001,
    0.01,
    0.1,
    0.5,
    1.0,
    2.0,
    10.0,
    100.0,
)

RANDOM_STATE: Final[int] = 42


def validate_regularisation_strength(
    regularisation_strength: float,
) -> None:
    """Validate inverse regularisation strength C."""
    if not np.isfinite(regularisation_strength):
        raise ValueError("regularisation_strength must be finite.")

    if regularisation_strength <= 0.0:
        raise ValueError("regularisation_strength must be greater than zero.")


def create_logistic_pipeline(
    regularisation_strength: float,
    class_weight: str | None = None,
) -> Pipeline:
    """Create a scaled multinomial logistic classifier."""
    validate_regularisation_strength(regularisation_strength)

    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight must be None or 'balanced'.")

    classifier = LogisticRegression(
        C=regularisation_strength,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=5_000,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def fit_logistic_pipeline(
    features: NDArray[np.float64],
    target: NDArray[np.int64],
    regularisation_strength: float,
    class_weight: str | None = None,
) -> Pipeline:
    """Fit a logistic pipeline on training data."""
    if features.ndim != 2:
        raise ValueError("Training features must be two-dimensional.")

    if target.ndim != 1:
        raise ValueError("Training target must be one-dimensional.")

    if features.shape[0] != target.shape[0]:
        raise ValueError("Training features and targets have different rows.")

    if features.shape[0] == 0:
        raise ValueError("Training data cannot be empty.")

    if not np.isfinite(features).all():
        raise ValueError("Training features contain non-finite values.")

    observed_classes = set(int(value) for value in np.unique(target).tolist())

    if observed_classes != set(CLASS_LABELS):
        raise ValueError("Training target must contain classes 0, 1 and 2.")

    pipeline = create_logistic_pipeline(
        regularisation_strength=regularisation_strength,
        class_weight=class_weight,
    )

    pipeline.fit(features, target)

    return pipeline


def ordered_predict_proba(
    model: Pipeline,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict probabilities ordered as away, draw and home."""
    if features.ndim != 2:
        raise ValueError("Prediction features must be two-dimensional.")

    raw_probabilities = np.asarray(
        model.predict_proba(features),
        dtype=np.float64,
    )

    classifier = model.named_steps["classifier"]

    if not isinstance(
        classifier,
        LogisticRegression,
    ):
        raise TypeError("Pipeline classifier is not LogisticRegression.")

    fitted_classes = np.asarray(
        classifier.classes_,
        dtype=np.int64,
    )

    class_to_position = {
        int(class_label): index
        for index, class_label in enumerate(fitted_classes.tolist())
    }

    missing_classes = sorted(set(CLASS_LABELS) - set(class_to_position))

    if missing_classes:
        raise ValueError(f"Model is missing classes: {missing_classes}")

    ordered = np.column_stack(
        [
            raw_probabilities[
                :,
                class_to_position[class_label],
            ]
            for class_label in CLASS_LABELS
        ]
    ).astype(np.float64)

    row_sums = ordered.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(row_sums <= 0.0):
        raise ValueError("Predicted probability rows must have positive mass.")

    return (ordered / row_sums).astype(np.float64)


def extract_logistic_coefficients(
    model: Pipeline,
    feature_names: tuple[str, ...],
) -> pl.DataFrame:
    """Return standardised logistic coefficients in long form."""
    classifier = model.named_steps["classifier"]

    if not isinstance(
        classifier,
        LogisticRegression,
    ):
        raise TypeError("Pipeline classifier is not LogisticRegression.")

    coefficients = np.asarray(
        classifier.coef_,
        dtype=np.float64,
    )

    classes = np.asarray(
        classifier.classes_,
        dtype=np.int64,
    )

    if coefficients.shape[1] != len(feature_names):
        raise ValueError("Coefficient width does not match feature names.")

    records: list[dict[str, str | int | float]] = []

    class_names = {
        0: "away_win",
        1: "draw",
        2: "home_win",
    }

    for class_position, class_label in enumerate(classes.tolist()):
        for feature_position, feature_name in enumerate(feature_names):
            coefficient = float(
                coefficients[
                    class_position,
                    feature_position,
                ]
            )

            records.append(
                {
                    "class_label": int(class_label),
                    "class_name": class_names[int(class_label)],
                    "feature": feature_name,
                    "coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                }
            )

    return pl.DataFrame(records).sort(
        [
            "class_label",
            "absolute_coefficient",
        ],
        descending=[False, True],
    )
