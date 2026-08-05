"""Probability baselines for football match outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

CLASS_COUNT: Final[int] = 3
AWAY_WIN_CLASS: Final[int] = 0
DRAW_CLASS: Final[int] = 1
HOME_WIN_CLASS: Final[int] = 2

DEFAULT_DRAW_PROBABILITY: Final[float] = 0.25
MINIMUM_PROBABILITY: Final[float] = 1e-9


@dataclass(frozen=True)
class BaselineProbabilities:
    """Named probability predictions from one baseline."""

    model_name: str
    probabilities: NDArray[np.float64]


def validate_target(target: NDArray[np.int64]) -> None:
    """Validate a three-class outcome target."""
    if target.ndim != 1:
        raise ValueError("Target must be a one-dimensional array.")

    if target.size == 0:
        raise ValueError("Target cannot be empty.")

    valid_classes = np.isin(
        target,
        np.arange(CLASS_COUNT),
    )

    if not bool(np.all(valid_classes)):
        raise ValueError("Target contains classes outside 0, 1 and 2.")


def repeat_probability_vector(
    probability_vector: NDArray[np.float64],
    row_count: int,
) -> NDArray[np.float64]:
    """Repeat one class-probability vector for many matches."""
    if row_count < 1:
        raise ValueError("row_count must be at least one.")

    if probability_vector.shape != (CLASS_COUNT,):
        raise ValueError("Probability vector must contain three values.")

    if not np.isfinite(probability_vector).all():
        raise ValueError("Probability vector contains non-finite values.")

    if np.any(probability_vector < 0.0):
        raise ValueError("Probability vector cannot contain negatives.")

    if not np.isclose(
        probability_vector.sum(),
        1.0,
        atol=1e-10,
    ):
        raise ValueError("Probability vector must sum to one.")

    return np.tile(
        probability_vector,
        (row_count, 1),
    ).astype(np.float64)


def majority_class_probabilities(
    training_target: NDArray[np.int64],
    row_count: int,
) -> NDArray[np.float64]:
    """Predict probability one for the majority training class."""
    validate_target(training_target)

    class_counts = np.bincount(
        training_target,
        minlength=CLASS_COUNT,
    )

    majority_class = int(np.argmax(class_counts))

    probability_vector = np.full(
        CLASS_COUNT,
        MINIMUM_PROBABILITY,
        dtype=np.float64,
    )
    probability_vector[majority_class] = 1.0 - MINIMUM_PROBABILITY * (CLASS_COUNT - 1)

    return repeat_probability_vector(
        probability_vector,
        row_count=row_count,
    )


def frequency_probabilities(
    training_target: NDArray[np.int64],
    row_count: int,
    smoothing: float = 1.0,
) -> NDArray[np.float64]:
    """Predict smoothed training class frequencies."""
    validate_target(training_target)

    if smoothing < 0.0:
        raise ValueError("smoothing cannot be negative.")

    class_counts = np.bincount(
        training_target,
        minlength=CLASS_COUNT,
    ).astype(np.float64)

    smoothed_counts = class_counts + smoothing
    probability_vector = smoothed_counts / smoothed_counts.sum()

    return repeat_probability_vector(
        probability_vector,
        row_count=row_count,
    )


def elo_outcome_probabilities(
    home_expected_score: NDArray[np.float64],
    draw_probability: float = DEFAULT_DRAW_PROBABILITY,
) -> NDArray[np.float64]:
    """Convert Elo home expectations to three outcome probabilities.

    Elo supplies an expected home score over win/draw/loss rather than
    an explicit three-class distribution. This baseline allocates a
    fixed draw probability and distributes the remaining probability
    mass between home and away outcomes according to the Elo expected
    score.
    """
    if home_expected_score.ndim != 1:
        raise ValueError("home_expected_score must be one-dimensional.")

    if home_expected_score.size == 0:
        raise ValueError("home_expected_score cannot be empty.")

    if not np.isfinite(home_expected_score).all():
        raise ValueError("Elo expectations contain non-finite values.")

    if np.any(home_expected_score < 0.0) or np.any(home_expected_score > 1.0):
        raise ValueError("Elo expectations must be between zero and one.")

    if not 0.0 <= draw_probability < 1.0:
        raise ValueError("draw_probability must be at least zero and less than one.")

    decisive_probability = 1.0 - draw_probability

    home_probability = home_expected_score * decisive_probability
    away_probability = (1.0 - home_expected_score) * decisive_probability

    probabilities = np.column_stack(
        (
            away_probability,
            np.full(
                home_expected_score.shape[0],
                draw_probability,
                dtype=np.float64,
            ),
            home_probability,
        )
    ).astype(np.float64)

    probabilities = np.clip(
        probabilities,
        MINIMUM_PROBABILITY,
        1.0,
    )

    row_sums = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    return (probabilities / row_sums).astype(np.float64)
