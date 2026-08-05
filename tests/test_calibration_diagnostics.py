"""Tests for FootCast probability calibration diagnostics."""

import numpy as np
import polars as pl
import pytest

from footcast.modelling.calibration_diagnostics import (
    assign_probability_bins,
    binary_reliability_table,
    build_class_calibration_diagnostics,
    build_confidence_diagnostics,
    calculate_calibration_metrics,
)


def test_probability_bins_include_zero_and_one() -> None:
    """Boundary probabilities should stay in valid bins."""
    probabilities = np.array(
        [0.0, 0.1, 0.5, 0.999, 1.0],
        dtype=np.float64,
    )

    result = assign_probability_bins(
        probabilities,
        bin_count=10,
    )

    assert result.tolist() == [
        0,
        1,
        5,
        9,
        9,
    ]


def test_perfect_binary_calibration_has_zero_error() -> None:
    """Perfect deterministic probabilities should have zero error."""
    target = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    probabilities = np.array(
        [0.0, 0.0, 1.0, 1.0],
        dtype=np.float64,
    )

    metrics = calculate_calibration_metrics(
        target,
        probabilities,
        bin_count=5,
    )

    assert metrics.expected_calibration_error == pytest.approx(0.0)

    assert metrics.maximum_calibration_error == pytest.approx(0.0)


def test_reliability_table_preserves_all_rows() -> None:
    """Populated-bin counts should sum to the input count."""
    target = np.array(
        [0, 1, 0, 1, 1],
        dtype=np.int64,
    )

    probabilities = np.array(
        [0.1, 0.3, 0.4, 0.7, 0.9],
        dtype=np.float64,
    )

    result = binary_reliability_table(
        target,
        probabilities,
        bin_count=5,
    )

    assert result["sample_count"].sum() == 5


def create_predictions() -> pl.DataFrame:
    """Create valid three-class predictions."""
    return pl.DataFrame(
        {
            "target": [0, 1, 2, 2, 0, 1],
            "probability_away_win": [
                0.70,
                0.20,
                0.10,
                0.15,
                0.60,
                0.20,
            ],
            "probability_draw": [
                0.20,
                0.60,
                0.20,
                0.15,
                0.20,
                0.60,
            ],
            "probability_home_win": [
                0.10,
                0.20,
                0.70,
                0.70,
                0.20,
                0.20,
            ],
        }
    )


def test_class_diagnostics_have_three_rows() -> None:
    """One summary row should be produced for each class."""
    summary, bins = build_class_calibration_diagnostics(
        create_predictions(),
        model_name="test_model",
        bin_count=5,
    )

    assert summary.height == 3
    assert set(summary["class_name"].to_list()) == {
        "away_win",
        "draw",
        "home_win",
    }

    assert bins.height == 15


def test_confidence_diagnostics_are_valid() -> None:
    """Maximum-confidence diagnostics should be bounded."""
    metrics, bins = build_confidence_diagnostics(
        create_predictions(),
        model_name="test_model",
        bin_count=5,
    )

    assert 0.0 <= (metrics.expected_calibration_error) <= 1.0

    assert 0.0 <= metrics.mean_confidence <= 1.0
    assert 0.0 <= metrics.observed_accuracy <= 1.0
    assert bins.height == 5
