"""Tests for probabilistic football evaluation metrics."""

import numpy as np
import pytest

from footcast.modelling.metrics import (
    evaluate_probabilities,
    multiclass_brier_score,
    ranked_probability_score,
    validate_probabilities,
)


def test_perfect_predictions_have_zero_loss() -> None:
    """Perfect class probabilities should have zero error."""
    target = np.array([0, 1, 2], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float64)

    result = evaluate_probabilities(
        target,
        probabilities,
    )

    assert result.accuracy == 1.0
    assert result.log_loss == pytest.approx(0.0)
    assert result.brier_score == pytest.approx(0.0)
    assert result.ranked_probability_score == pytest.approx(0.0)


def test_uniform_predictions_have_expected_brier_score() -> None:
    """Uniform three-class predictions have a known score."""
    target = np.array([0, 1, 2], dtype=np.int64)
    probabilities = np.full(
        (3, 3),
        1.0 / 3.0,
        dtype=np.float64,
    )

    score = multiclass_brier_score(
        target,
        probabilities,
    )

    assert score == pytest.approx(2.0 / 3.0)


def test_ranked_probability_score_is_non_negative() -> None:
    """Ranked probability score cannot be negative."""
    target = np.array([0, 1, 2], dtype=np.int64)
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )

    assert (
        ranked_probability_score(
            target,
            probabilities,
        )
        >= 0.0
    )


def test_probabilities_must_sum_to_one() -> None:
    """Invalid row totals should fail validation."""
    probabilities = np.array(
        [[0.5, 0.5, 0.5]],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        validate_probabilities(probabilities)


def test_probability_shape_is_validated() -> None:
    """Exactly three probability columns are required."""
    probabilities = np.array(
        [[0.5, 0.5]],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="exactly three classes",
    ):
        validate_probabilities(probabilities)
