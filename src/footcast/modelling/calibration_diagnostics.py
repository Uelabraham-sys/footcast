"""Probability-calibration diagnostics for FootCast models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

CLASS_LABELS: Final[tuple[int, int, int]] = (0, 1, 2)

CLASS_NAMES: Final[tuple[str, str, str]] = (
    "away_win",
    "draw",
    "home_win",
)

PROBABILITY_COLUMNS: Final[tuple[str, str, str]] = (
    "probability_away_win",
    "probability_draw",
    "probability_home_win",
)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Summary calibration statistics for one probability series."""

    expected_calibration_error: float
    maximum_calibration_error: float
    mean_predicted_probability: float
    observed_frequency: float
    calibration_bias: float
    populated_bins: int

    def to_dict(self) -> dict[str, float | int]:
        """Return calibration metrics as a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceMetrics:
    """Calibration statistics for maximum model confidence."""

    expected_calibration_error: float
    maximum_calibration_error: float
    mean_confidence: float
    observed_accuracy: float
    overconfidence_gap: float
    populated_bins: int

    def to_dict(self) -> dict[str, float | int]:
        """Return confidence metrics as a dictionary."""
        return asdict(self)


def validate_binary_calibration_inputs(
    target: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> None:
    """Validate binary one-vs-rest calibration inputs."""
    if target.ndim != 1:
        raise ValueError("Binary calibration target must be one-dimensional.")

    if probabilities.ndim != 1:
        raise ValueError("Binary probabilities must be one-dimensional.")

    if target.shape[0] != probabilities.shape[0]:
        raise ValueError("Target and probability row counts differ.")

    if target.shape[0] == 0:
        raise ValueError("Calibration inputs cannot be empty.")

    if not np.isin(target, [0, 1]).all():
        raise ValueError("Binary calibration target must contain only zero and one.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Calibration probabilities contain non-finite values.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Calibration probabilities must be between zero and one.")


def create_uniform_bin_edges(
    bin_count: int,
) -> NDArray[np.float64]:
    """Create equal-width probability-bin edges."""
    if bin_count < 2:
        raise ValueError("bin_count must be at least two.")

    return np.linspace(
        0.0,
        1.0,
        num=bin_count + 1,
        dtype=np.float64,
    )


def assign_probability_bins(
    probabilities: NDArray[np.float64],
    bin_count: int,
) -> NDArray[np.int64]:
    """Assign probabilities to equal-width bins."""
    edges = create_uniform_bin_edges(bin_count)

    bins = (
        np.searchsorted(
            edges,
            probabilities,
            side="right",
        )
        - 1
    )

    return np.clip(
        bins,
        0,
        bin_count - 1,
    ).astype(np.int64)


def binary_reliability_table(
    target: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    *,
    bin_count: int = 10,
) -> pl.DataFrame:
    """Build a binary reliability table using uniform bins."""
    validate_binary_calibration_inputs(
        target,
        probabilities,
    )

    bins = assign_probability_bins(
        probabilities,
        bin_count,
    )

    edges = create_uniform_bin_edges(bin_count)

    records: list[dict[str, int | float | None]] = []

    for bin_index in range(bin_count):
        mask = bins == bin_index
        sample_count = int(mask.sum())

        lower_bound = float(edges[bin_index])
        upper_bound = float(edges[bin_index + 1])

        if sample_count == 0:
            records.append(
                {
                    "bin_index": bin_index,
                    "bin_lower": lower_bound,
                    "bin_upper": upper_bound,
                    "sample_count": 0,
                    "mean_probability": None,
                    "observed_frequency": None,
                    "absolute_gap": None,
                }
            )
            continue

        mean_probability = float(probabilities[mask].mean())

        observed_frequency = float(target[mask].mean())

        records.append(
            {
                "bin_index": bin_index,
                "bin_lower": lower_bound,
                "bin_upper": upper_bound,
                "sample_count": sample_count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "absolute_gap": abs(mean_probability - observed_frequency),
            }
        )

    return pl.DataFrame(records)


def calculate_calibration_metrics(
    target: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    *,
    bin_count: int = 10,
) -> CalibrationMetrics:
    """Calculate expected and maximum calibration errors."""
    table = binary_reliability_table(
        target,
        probabilities,
        bin_count=bin_count,
    )

    populated = table.filter(pl.col("sample_count") > 0)

    total_count = int(populated["sample_count"].sum())

    if total_count == 0:
        raise ValueError("No populated calibration bins were found.")

    weighted_error_value = populated.select(
        (pl.col("absolute_gap") * pl.col("sample_count") / total_count)
        .sum()
        .alias("value")
    )["value"].item()

    maximum_error_value = populated.select(pl.col("absolute_gap").max().alias("value"))[
        "value"
    ].item()

    if not isinstance(
        weighted_error_value,
        (int, float),
    ):
        raise TypeError("Expected calibration error must be numeric.")

    if not isinstance(
        maximum_error_value,
        (int, float),
    ):
        raise TypeError("Maximum calibration error must be numeric.")

    mean_probability = float(probabilities.mean())
    observed_frequency = float(target.mean())

    return CalibrationMetrics(
        expected_calibration_error=float(weighted_error_value),
        maximum_calibration_error=float(maximum_error_value),
        mean_predicted_probability=mean_probability,
        observed_frequency=observed_frequency,
        calibration_bias=(mean_probability - observed_frequency),
        populated_bins=populated.height,
    )


def extract_prediction_arrays(
    predictions: pl.DataFrame,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
]:
    """Extract targets and ordered probability arrays."""
    required = {
        "target",
        *PROBABILITY_COLUMNS,
    }

    missing = sorted(required - set(predictions.columns))

    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    labelled = predictions.filter(pl.col("target").is_not_null())

    if labelled.is_empty():
        raise ValueError("Prediction dataset contains no labelled rows.")

    target = np.asarray(
        labelled["target"].to_numpy(),
        dtype=np.int64,
    )

    probabilities = np.asarray(
        labelled.select(PROBABILITY_COLUMNS).to_numpy(),
        dtype=np.float64,
    )

    if probabilities.shape[1] != 3:
        raise ValueError("Exactly three probability columns are required.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities contain non-finite values.")

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-8,
    ):
        raise ValueError("Prediction probability rows must sum to one.")

    return target, probabilities


def build_class_calibration_diagnostics(
    predictions: pl.DataFrame,
    *,
    model_name: str,
    bin_count: int = 10,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Calculate one-vs-rest calibration for each class."""
    target, probabilities = extract_prediction_arrays(predictions)

    summary_records: list[dict[str, str | int | float]] = []

    bin_frames: list[pl.DataFrame] = []

    for class_position, (
        class_label,
        class_name,
    ) in enumerate(
        zip(
            CLASS_LABELS,
            CLASS_NAMES,
            strict=True,
        )
    ):
        binary_target = (target == class_label).astype(np.int64)

        class_probabilities = probabilities[
            :,
            class_position,
        ]

        metrics = calculate_calibration_metrics(
            binary_target,
            class_probabilities,
            bin_count=bin_count,
        )

        summary_records.append(
            {
                "model": model_name,
                "class_label": class_label,
                "class_name": class_name,
                **metrics.to_dict(),
            }
        )

        table = binary_reliability_table(
            binary_target,
            class_probabilities,
            bin_count=bin_count,
        ).with_columns(
            pl.lit(model_name).alias("model"),
            pl.lit(class_label).alias("class_label"),
            pl.lit(class_name).alias("class_name"),
        )

        bin_frames.append(table)

    return (
        pl.DataFrame(summary_records),
        pl.concat(bin_frames),
    )


def build_confidence_diagnostics(
    predictions: pl.DataFrame,
    *,
    model_name: str,
    bin_count: int = 10,
) -> tuple[ConfidenceMetrics, pl.DataFrame]:
    """Evaluate calibration of each prediction's confidence."""
    target, probabilities = extract_prediction_arrays(predictions)

    predicted_class = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)

    confidence = np.max(
        probabilities,
        axis=1,
    ).astype(np.float64)

    correctness = (predicted_class == target).astype(np.int64)

    base_metrics = calculate_calibration_metrics(
        correctness,
        confidence,
        bin_count=bin_count,
    )

    metrics = ConfidenceMetrics(
        expected_calibration_error=(base_metrics.expected_calibration_error),
        maximum_calibration_error=(base_metrics.maximum_calibration_error),
        mean_confidence=float(confidence.mean()),
        observed_accuracy=float(correctness.mean()),
        overconfidence_gap=float(confidence.mean() - correctness.mean()),
        populated_bins=(base_metrics.populated_bins),
    )

    table = binary_reliability_table(
        correctness,
        confidence,
        bin_count=bin_count,
    ).with_columns(
        pl.lit(model_name).alias("model"),
        pl.lit("confidence").alias("diagnostic_type"),
    )

    return metrics, table
