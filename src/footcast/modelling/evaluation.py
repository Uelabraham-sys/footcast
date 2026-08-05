"""Utilities for saving model evaluations and predictions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from footcast.modelling.metrics import (
    ClassificationMetrics,
    evaluate_probabilities,
)


def create_prediction_frame(
    metadata: pl.DataFrame,
    probabilities: NDArray[np.float64],
) -> pl.DataFrame:
    """Combine match metadata with class probabilities."""
    if metadata.height != probabilities.shape[0]:
        raise ValueError("Metadata and probability row counts do not match.")

    predicted_class = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    probability_frame = pl.DataFrame(
        {
            "probability_away_win": probabilities[:, 0],
            "probability_draw": probabilities[:, 1],
            "probability_home_win": probabilities[:, 2],
            "predicted_class": predicted_class,
        }
    )

    return metadata.hstack(probability_frame)


def evaluate_prediction_frame(
    target: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> ClassificationMetrics:
    """Evaluate one model's probability predictions."""
    return evaluate_probabilities(
        target=target,
        probabilities=probabilities,
    )


def write_prediction_frame(
    dataframe: pl.DataFrame,
    output_path: Path,
) -> None:
    """Write prediction records as Parquet."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.write_parquet(
        output_path,
        compression="zstd",
        statistics=True,
    )


def write_evaluation_report(
    *,
    model_name: str,
    validation_metrics: ClassificationMetrics,
    test_metrics: ClassificationMetrics | None,
    output_path: Path,
    parameters: dict[str, Any] | None = None,
) -> None:
    """Write model metrics and configuration as JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": model_name,
        "validation_metrics": (validation_metrics.to_dict()),
        "test_metrics": (test_metrics.to_dict() if test_metrics is not None else None),
        "parameters": parameters or {},
    }

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
