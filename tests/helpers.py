"""Shared test factories for FootCast."""

import numpy as np

from footcast.modelling.gradient_boosting import (
    HGBParameters,
    fit_hgb_classifier,
    fit_sigmoid_calibrator,
)
from footcast.modelling.logistic import (
    fit_logistic_pipeline,
)
from footcast.prediction.bundle import (
    ProductionBundle,
    create_production_bundle,
)


def create_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Create balanced synthetic training data."""
    rng = np.random.default_rng(42)

    features = np.vstack(
        [
            rng.normal(-2.0, 0.2, size=(30, 2)),
            rng.normal(0.0, 0.2, size=(30, 2)),
            rng.normal(2.0, 0.2, size=(30, 2)),
        ]
    ).astype(np.float64)

    target = np.repeat(
        np.array([0, 1, 2], dtype=np.int64),
        30,
    )

    return features, target


def create_bundle() -> ProductionBundle:
    """Create a fitted production bundle."""
    features, target = create_training_data()

    fit_indices = np.concatenate(
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

    logistic = fit_logistic_pipeline(
        features,
        target,
        regularisation_strength=1.0,
    )

    hgb = fit_hgb_classifier(
        features[fit_indices],
        target[fit_indices],
        HGBParameters(
            max_iter=20,
            min_samples_leaf=5,
        ),
    )

    calibrated = fit_sigmoid_calibrator(
        hgb,
        features[calibration_indices],
        target[calibration_indices],
    )

    return create_production_bundle(
        logistic_model=logistic,
        calibrated_hgb_model=calibrated,
        feature_names=(
            "feature_one",
            "feature_two",
        ),
        logistic_weight=0.6,
        hgb_weight=0.4,
        metadata={},
    )
