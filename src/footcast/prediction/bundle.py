"""Versioned production model bundle for FootCast."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from footcast.modelling.ensemble import (
    blend_probabilities,
)
from footcast.modelling.gradient_boosting import (
    ordered_predict_proba as hgb_predict_proba,
)
from footcast.modelling.logistic import (
    ordered_predict_proba as logistic_predict_proba,
)

BUNDLE_SCHEMA_VERSION: Final[str] = "1.0.0"

CLASS_LABELS: Final[tuple[int, int, int]] = (
    0,
    1,
    2,
)

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
class ProductionBundle:
    """All components required for FootCast inference."""

    schema_version: str
    created_at: str
    logistic_model: Pipeline
    calibrated_hgb_model: CalibratedClassifierCV
    feature_names: tuple[str, ...]
    logistic_weight: float
    hgb_weight: float
    class_labels: tuple[int, int, int]
    class_names: tuple[str, str, str]
    metadata: dict[str, Any]


def validate_ensemble_weights(
    logistic_weight: float,
    hgb_weight: float,
) -> None:
    """Validate production ensemble weights."""
    values = np.array(
        [logistic_weight, hgb_weight],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError("Production ensemble weights must be finite.")

    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Production ensemble weights must be between zero and one.")

    if not np.isclose(
        logistic_weight + hgb_weight,
        1.0,
        atol=1e-10,
    ):
        raise ValueError("Production ensemble weights must sum to one.")


def create_production_bundle(
    *,
    logistic_model: Pipeline,
    calibrated_hgb_model: CalibratedClassifierCV,
    feature_names: tuple[str, ...],
    logistic_weight: float,
    hgb_weight: float,
    metadata: dict[str, Any],
) -> ProductionBundle:
    """Create a validated production bundle."""
    if not feature_names:
        raise ValueError("Production bundle requires feature names.")

    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Production feature names must be unique.")

    validate_ensemble_weights(
        logistic_weight,
        hgb_weight,
    )

    return ProductionBundle(
        schema_version=BUNDLE_SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        logistic_model=logistic_model,
        calibrated_hgb_model=calibrated_hgb_model,
        feature_names=feature_names,
        logistic_weight=logistic_weight,
        hgb_weight=hgb_weight,
        class_labels=CLASS_LABELS,
        class_names=CLASS_NAMES,
        metadata=metadata,
    )


def validate_feature_contract(
    dataframe: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> None:
    """Validate inference data against the bundle contract."""
    missing = sorted(set(feature_names) - set(dataframe.columns))

    if missing:
        raise ValueError(f"Inference data is missing features: {missing}")

    duplicated = [
        feature for feature in feature_names if dataframe.columns.count(feature) > 1
    ]

    if duplicated:
        raise ValueError(f"Inference data contains duplicate features: {duplicated}")

    if dataframe.is_empty():
        raise ValueError("Inference data cannot be empty.")

    non_numeric = [
        feature
        for feature in feature_names
        if not dataframe.schema[feature].is_numeric()
    ]

    if non_numeric:
        raise TypeError(f"Inference features must be numeric: {non_numeric}")


def extract_feature_matrix(
    dataframe: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> NDArray[np.float64]:
    """Extract a finite model-ready feature matrix."""
    validate_feature_contract(
        dataframe,
        feature_names,
    )

    matrix = np.asarray(
        dataframe.select(feature_names).to_numpy(),
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError("Inference feature matrix must be two-dimensional.")

    if matrix.shape[1] != len(feature_names):
        raise ValueError("Inference feature width does not match the model contract.")

    if not np.isfinite(matrix).all():
        raise ValueError("Inference features contain null or non-finite values.")

    return matrix


def predict_bundle_probabilities(
    bundle: ProductionBundle,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Generate ensemble probabilities from a production bundle."""
    logistic_probabilities = logistic_predict_proba(
        bundle.logistic_model,
        features,
    )

    hgb_probabilities = hgb_predict_proba(
        bundle.calibrated_hgb_model,
        features,
    )

    return blend_probabilities(
        logistic_probabilities,
        hgb_probabilities,
        first_weight=bundle.logistic_weight,
    )


def feature_contract_hash(
    feature_names: tuple[str, ...],
) -> str:
    """Create a stable hash of the ordered feature contract."""
    payload = json.dumps(
        list(feature_names),
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def save_production_bundle(
    bundle: ProductionBundle,
    path: Path,
) -> None:
    """Persist a production bundle."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(bundle, path)


def load_production_bundle(
    path: Path,
) -> ProductionBundle:
    """Load and validate a production bundle."""
    if not path.exists():
        raise FileNotFoundError(f"Production bundle was not found: {path}")

    value = joblib.load(path)

    if not isinstance(value, ProductionBundle):
        raise TypeError("Loaded artifact is not a FootCast ProductionBundle.")

    if value.schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported production bundle schema version: {value.schema_version}"
        )

    validate_ensemble_weights(
        value.logistic_weight,
        value.hgb_weight,
    )

    return value
