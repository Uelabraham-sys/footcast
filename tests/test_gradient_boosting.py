"""Tests for FootCast histogram gradient boosting."""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from footcast.modelling.gradient_boosting import (
    HGBParameters,
    build_permutation_importance_frame,
    create_hgb_classifier,
    fit_hgb_classifier,
    fit_sigmoid_calibrator,
    ordered_predict_proba,
)


def create_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Create representative three-class data."""
    rng = np.random.default_rng(42)

    class_zero = rng.normal(
        loc=-2.0,
        scale=0.3,
        size=(30, 3),
    )
    class_one = rng.normal(
        loc=0.0,
        scale=0.3,
        size=(30, 3),
    )
    class_two = rng.normal(
        loc=2.0,
        scale=0.3,
        size=(30, 3),
    )

    features = np.vstack(
        [
            class_zero,
            class_one,
            class_two,
        ]
    ).astype(np.float64)

    target = np.repeat(
        np.array([0, 1, 2], dtype=np.int64),
        repeats=30,
    )

    return features, target


def test_create_classifier() -> None:
    """Factory should return the expected classifier."""
    model = create_hgb_classifier(
        HGBParameters(
            max_iter=10,
            min_samples_leaf=5,
        )
    )

    assert isinstance(
        model,
        HistGradientBoostingClassifier,
    )


def test_hgb_probabilities_sum_to_one() -> None:
    """HGB predictions should form valid distributions."""
    features, target = create_training_data()

    model = fit_hgb_classifier(
        features,
        target,
        HGBParameters(
            max_iter=30,
            min_samples_leaf=5,
        ),
    )

    probabilities = ordered_predict_proba(
        model,
        features,
    )

    assert probabilities.shape == (90, 3)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0] * 90)


def test_calibrated_probabilities_sum_to_one() -> None:
    """Calibrated predictions should remain valid."""
    features, target = create_training_data()

    # The training slice above lacks class 2, so instead use
    # alternating rows to preserve all classes in each subset.
    training_indices = np.concatenate(
        [
            np.arange(0, 20),
            np.arange(30, 50),
            np.arange(60, 80),
        ]
    )
    calibration_indices = np.concatenate(
        [
            np.arange(20, 30),
            np.arange(50, 60),
            np.arange(80, 90),
        ]
    )

    model = fit_hgb_classifier(
        features[training_indices],
        target[training_indices],
        HGBParameters(
            max_iter=20,
            min_samples_leaf=5,
        ),
    )

    calibrated = fit_sigmoid_calibrator(
        model,
        features[calibration_indices],
        target[calibration_indices],
    )

    probabilities = ordered_predict_proba(
        calibrated,
        features[calibration_indices],
    )

    assert isinstance(
        calibrated,
        CalibratedClassifierCV,
    )
    assert probabilities.shape == (30, 3)
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0] * 30)


def test_invalid_parameters_fail() -> None:
    """Invalid learning rates should fail."""
    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        create_hgb_classifier(HGBParameters(learning_rate=0.0))


def test_permutation_importance_has_one_row_per_feature() -> None:
    """Importance output should match the feature set."""
    features, target = create_training_data()

    model = fit_hgb_classifier(
        features,
        target,
        HGBParameters(
            max_iter=20,
            min_samples_leaf=5,
        ),
    )

    result = build_permutation_importance_frame(
        model,
        features,
        target,
        (
            "feature_one",
            "feature_two",
            "feature_three",
        ),
        n_repeats=2,
    )

    assert result.height == 3
    assert set(result["feature"].to_list()) == {
        "feature_one",
        "feature_two",
        "feature_three",
    }
