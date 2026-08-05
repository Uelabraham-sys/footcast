"""Tests for FootCast probability ensembles."""

import numpy as np
import polars as pl
import pytest

from footcast.modelling.ensemble import (
    align_prediction_frames,
    blend_probabilities,
    create_ensemble_prediction_frame,
    extract_aligned_arrays,
)


def create_first_predictions() -> pl.DataFrame:
    """Create first-model prediction data."""
    return pl.DataFrame(
        {
            "match_key": ["m1", "m2", "m3"],
            "target": [0, 1, 2],
            "probability_away_win": [
                0.70,
                0.20,
                0.10,
            ],
            "probability_draw": [
                0.20,
                0.60,
                0.20,
            ],
            "probability_home_win": [
                0.10,
                0.20,
                0.70,
            ],
        }
    )


def create_second_predictions() -> pl.DataFrame:
    """Create second-model prediction data."""
    return pl.DataFrame(
        {
            "match_key": ["m3", "m1", "m2"],
            "target": [2, 0, 1],
            "probability_away_win": [
                0.15,
                0.60,
                0.20,
            ],
            "probability_draw": [
                0.15,
                0.25,
                0.55,
            ],
            "probability_home_win": [
                0.70,
                0.15,
                0.25,
            ],
        }
    )


def test_blended_probabilities_sum_to_one() -> None:
    """Convex blends should remain probability distributions."""
    first = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
        ],
        dtype=np.float64,
    )

    second = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
        ],
        dtype=np.float64,
    )

    blended = blend_probabilities(
        first,
        second,
        first_weight=0.4,
    )

    assert blended.shape == (2, 3)
    assert blended.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_zero_weight_returns_second_model() -> None:
    """A zero first-model weight should return model two."""
    first = np.array(
        [[0.7, 0.2, 0.1]],
        dtype=np.float64,
    )

    second = np.array(
        [[0.2, 0.3, 0.5]],
        dtype=np.float64,
    )

    result = blend_probabilities(
        first,
        second,
        first_weight=0.0,
    )

    np.testing.assert_allclose(
        result,
        second,
    )


def test_alignment_uses_match_key() -> None:
    """Prediction rows should align regardless of input order."""
    aligned = align_prediction_frames(
        create_first_predictions(),
        create_second_predictions(),
        first_name="first",
        second_name="second",
    )

    target, first, second = extract_aligned_arrays(aligned)

    assert target.tolist() == [0, 1, 2]
    assert first.shape == (3, 3)
    assert second.shape == (3, 3)

    assert second[0].tolist() == pytest.approx([0.60, 0.25, 0.15])


def test_missing_match_key_fails() -> None:
    """Both prediction files must cover identical matches."""
    second = create_second_predictions().filter(pl.col("match_key") != "m3")

    with pytest.raises(
        ValueError,
        match="do not align",
    ):
        align_prediction_frames(
            create_first_predictions(),
            second,
            first_name="first",
            second_name="second",
        )


def test_prediction_frame_contains_weights() -> None:
    """Ensemble output should include component weights."""
    aligned = align_prediction_frames(
        create_first_predictions(),
        create_second_predictions(),
        first_name="first",
        second_name="second",
    )

    _, first, second = extract_aligned_arrays(aligned)

    probabilities = blend_probabilities(
        first,
        second,
        first_weight=0.25,
    )

    result = create_ensemble_prediction_frame(
        aligned,
        probabilities,
        first_weight=0.25,
        first_model="first",
        second_model="second",
    )

    assert result.height == 3
    assert result["first_model_weight"].unique().to_list() == [0.25]

    assert result["second_model_weight"].unique().to_list() == [0.75]
