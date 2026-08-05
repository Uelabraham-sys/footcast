"""Tests for FootCast multinomial logistic regression."""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression

from footcast.modelling.logistic import (
    create_logistic_pipeline,
    extract_logistic_coefficients,
    fit_logistic_pipeline,
    ordered_predict_proba,
)


def create_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Create a separable three-class training dataset."""
    features = np.array(
        [
            [-3.0, -2.0],
            [-2.5, -2.5],
            [-2.0, -3.0],
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.2, 0.1],
            [2.0, 3.0],
            [2.5, 2.5],
            [3.0, 2.0],
        ],
        dtype=np.float64,
    )

    target = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
        ],
        dtype=np.int64,
    )

    return features, target


def test_pipeline_contains_scaler_and_classifier() -> None:
    """Pipeline should contain scaling and classification."""
    pipeline = create_logistic_pipeline(regularisation_strength=1.0)

    assert list(pipeline.named_steps) == [
        "scaler",
        "classifier",
    ]
    assert isinstance(
        pipeline.named_steps["classifier"],
        LogisticRegression,
    )


def test_fitted_probabilities_sum_to_one() -> None:
    """Predicted class probabilities must sum to one."""
    features, target = create_training_data()

    model = fit_logistic_pipeline(
        features=features,
        target=target,
        regularisation_strength=1.0,
    )

    probabilities = ordered_predict_proba(
        model,
        features,
    )

    assert probabilities.shape == (9, 3)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0] * 9)


def test_probability_order_matches_class_labels() -> None:
    """Columns should correspond to classes zero, one and two."""
    features, target = create_training_data()

    model = fit_logistic_pipeline(
        features=features,
        target=target,
        regularisation_strength=10.0,
    )

    probabilities = ordered_predict_proba(
        model,
        features,
    )

    predicted = np.argmax(
        probabilities,
        axis=1,
    )

    assert predicted.tolist() == target.tolist()


def test_missing_training_class_fails() -> None:
    """Training requires all three outcome classes."""
    features, target = create_training_data()

    mask = target != 1

    with pytest.raises(
        ValueError,
        match="classes 0, 1 and 2",
    ):
        fit_logistic_pipeline(
            features=features[mask],
            target=target[mask],
            regularisation_strength=1.0,
        )


def test_invalid_regularisation_strength_fails() -> None:
    """Inverse regularisation strength must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        create_logistic_pipeline(regularisation_strength=0.0)


def test_coefficient_extraction() -> None:
    """Coefficient report should contain each class-feature pair."""
    features, target = create_training_data()

    model = fit_logistic_pipeline(
        features=features,
        target=target,
        regularisation_strength=1.0,
    )

    result = extract_logistic_coefficients(
        model,
        feature_names=(
            "feature_one",
            "feature_two",
        ),
    )

    assert isinstance(result, pl.DataFrame)
    assert result.height == 6

    assert set(result["class_label"].to_list()) == {0, 1, 2}

    assert set(result["feature"].to_list()) == {
        "feature_one",
        "feature_two",
    }
