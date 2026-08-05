"""Evaluation metrics for probabilistic football predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

CLASS_LABELS: Final[tuple[int, int, int]] = (0, 1, 2)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Summary metrics for a multiclass classifier."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    log_loss: float
    brier_score: float
    ranked_probability_score: float

    def to_dict(self) -> dict[str, float]:
        """Return metrics as a serialisable dictionary."""
        return asdict(self)


def validate_probabilities(
    probabilities: NDArray[np.float64],
) -> None:
    """Validate multiclass probability predictions."""
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a two-dimensional array.")

    if probabilities.shape[1] != len(CLASS_LABELS):
        raise ValueError("Probabilities must contain exactly three classes.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values.")

    if np.any(probabilities < 0.0):
        raise ValueError("Probabilities cannot be negative.")

    if np.any(probabilities > 1.0):
        raise ValueError("Probabilities cannot exceed one.")

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Each probability row must sum to one.")


def multiclass_brier_score(
    target: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    """Calculate the mean multiclass Brier score."""
    validate_probabilities(probabilities)

    one_hot = np.eye(
        len(CLASS_LABELS),
        dtype=np.float64,
    )[target]

    squared_error = np.sum(
        (probabilities - one_hot) ** 2,
        axis=1,
    )

    return float(np.mean(squared_error))


def ranked_probability_score(
    target: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> float:
    """Calculate mean ranked probability score."""
    validate_probabilities(probabilities)

    one_hot = np.eye(
        len(CLASS_LABELS),
        dtype=np.float64,
    )[target]

    predicted_cumulative = np.cumsum(
        probabilities,
        axis=1,
    )[:, :-1]

    observed_cumulative = np.cumsum(
        one_hot,
        axis=1,
    )[:, :-1]

    per_row_score = np.mean(
        (predicted_cumulative - observed_cumulative) ** 2,
        axis=1,
    )

    return float(np.mean(per_row_score))


def evaluate_probabilities(
    target: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> ClassificationMetrics:
    """Evaluate probabilistic three-way predictions."""
    validate_probabilities(probabilities)

    if target.ndim != 1:
        raise ValueError("Target must be a one-dimensional array.")

    if target.shape[0] != probabilities.shape[0]:
        raise ValueError("Targets and probabilities have different row counts.")

    predicted_class = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    return ClassificationMetrics(
        accuracy=float(
            accuracy_score(
                target,
                predicted_class,
            )
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(
                target,
                predicted_class,
            )
        ),
        macro_f1=float(
            f1_score(
                target,
                predicted_class,
                average="macro",
                zero_division=0,
            )
        ),
        log_loss=float(
            log_loss(
                target,
                probabilities,
                labels=list(CLASS_LABELS),
            )
        ),
        brier_score=multiclass_brier_score(
            target,
            probabilities,
        ),
        ranked_probability_score=ranked_probability_score(
            target,
            probabilities,
        ),
    )
