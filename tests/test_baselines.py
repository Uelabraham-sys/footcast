"""Tests for football probability baselines."""

import numpy as np
import pytest

from footcast.modelling.baselines import (
    elo_outcome_probabilities,
    frequency_probabilities,
    majority_class_probabilities,
    repeat_probability_vector,
)


def test_repeat_probability_vector() -> None:
    """One probability vector should repeat by row."""
    vector = np.array(
        [0.2, 0.3, 0.5],
        dtype=np.float64,
    )

    result = repeat_probability_vector(
        vector,
        row_count=2,
    )

    assert result.shape == (2, 3)
    assert result[0].tolist() == pytest.approx([0.2, 0.3, 0.5])
    assert result[1].tolist() == pytest.approx([0.2, 0.3, 0.5])


def test_majority_baseline_predicts_largest_class() -> None:
    """Majority baseline should assign one to the largest class."""
    target = np.array(
        [0, 2, 2, 2, 1],
        dtype=np.int64,
    )

    result = majority_class_probabilities(
        target,
        row_count=3,
    )

    assert np.argmax(result[0]) == 2
    assert result[0].sum() == pytest.approx(1.0)
    assert np.all(result > 0.0)


def test_frequency_baseline_uses_training_distribution() -> None:
    """Frequency baseline should estimate training proportions."""
    target = np.array(
        [0, 1, 2, 2],
        dtype=np.int64,
    )

    result = frequency_probabilities(
        target,
        row_count=1,
        smoothing=0.0,
    )

    assert result[0].tolist() == pytest.approx([0.25, 0.25, 0.50])


def test_frequency_smoothing_avoids_zero_probabilities() -> None:
    """Smoothing should assign mass to unseen classes."""
    target = np.array(
        [2, 2, 2],
        dtype=np.int64,
    )

    result = frequency_probabilities(
        target,
        row_count=1,
        smoothing=1.0,
    )

    assert np.all(result > 0.0)
    assert result.sum() == pytest.approx(1.0)


def test_elo_probabilities_have_expected_shape() -> None:
    """Elo expectations should become three-class probabilities."""
    expectations = np.array(
        [0.5, 0.8],
        dtype=np.float64,
    )

    result = elo_outcome_probabilities(
        expectations,
        draw_probability=0.25,
    )

    assert result.shape == (2, 3)
    assert result.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_equal_elo_expectation_is_symmetric() -> None:
    """An expectation of 0.5 should balance away and home wins."""
    result = elo_outcome_probabilities(
        np.array([0.5], dtype=np.float64),
        draw_probability=0.25,
    )

    assert result[0, 0] == pytest.approx(result[0, 2])
    assert result[0, 1] == pytest.approx(0.25)


def test_higher_home_expectation_increases_home_probability() -> None:
    """A stronger home Elo should increase home-win probability."""
    result = elo_outcome_probabilities(
        np.array([0.4, 0.8], dtype=np.float64),
        draw_probability=0.25,
    )

    assert result[1, 2] > result[0, 2]
    assert result[1, 0] < result[0, 0]


def test_invalid_draw_probability_fails() -> None:
    """Draw probability must lie in the valid interval."""
    with pytest.raises(
        ValueError,
        match="draw_probability",
    ):
        elo_outcome_probabilities(
            np.array([0.5], dtype=np.float64),
            draw_probability=1.0,
        )
