"""Validate FootCast production artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

MODEL_BUNDLE: Final[Path] = Path(
    "artifacts/models/production/footcast_bundle.joblib"
)

MODEL_MANIFEST: Final[Path] = Path(
    "artifacts/models/production/manifest.json"
)

TRAINING_REPORT: Final[Path] = Path(
    "artifacts/reports/production_training.json"
)

REQUIRED_PATHS: Final[tuple[Path, ...]] = (
    MODEL_BUNDLE,
    MODEL_MANIFEST,
    TRAINING_REPORT,
)

REQUIRED_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "created_at",
        "bundle_path",
        "feature_count",
        "feature_names",
        "class_labels",
        "class_names",
        "ensemble_weights",
        "training_cutoff",
    }
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a JSON object in {path}."
        )

    return payload


def validate_required_paths() -> None:
    """Ensure required production artefacts exist."""
    missing = [
        path
        for path in REQUIRED_PATHS
        if not path.is_file()
    ]

    if not missing:
        return

    rendered = "\n".join(
        f"- {path}"
        for path in missing
    )

    raise FileNotFoundError(
        "Required production artefacts are missing:\n"
        f"{rendered}"
    )


def validate_manifest_fields(
    payload: dict[str, Any],
) -> None:
    """Ensure the manifest contains required fields."""
    missing_fields = (
        REQUIRED_MANIFEST_FIELDS - payload.keys()
    )

    if not missing_fields:
        return

    rendered = ", ".join(
        sorted(missing_fields)
    )

    raise ValueError(
        "Production manifest is missing fields: "
        f"{rendered}"
    )


def validate_feature_contract(
    payload: dict[str, Any],
) -> None:
    """Validate manifest feature metadata."""
    feature_names = payload["feature_names"]
    feature_count = payload["feature_count"]

    if not isinstance(feature_names, list):
        raise TypeError(
            "Manifest feature_names must be a list."
        )

    if not all(
        isinstance(feature, str)
        and feature.strip()
        for feature in feature_names
    ):
        raise ValueError(
            "Manifest feature_names must contain "
            "non-empty strings."
        )

    if len(feature_names) != len(
        set(feature_names)
    ):
        raise ValueError(
            "Manifest feature_names contain duplicates."
        )

    if not isinstance(feature_count, int):
        raise TypeError(
            "Manifest feature_count must be an integer."
        )

    if feature_count <= 0:
        raise ValueError(
            "Manifest feature_count must be positive."
        )

    if feature_count != len(feature_names):
        raise ValueError(
            "Manifest feature_count does not match "
            "feature_names."
        )


def validate_classes(
    payload: dict[str, Any],
) -> None:
    """Validate model class metadata."""
    class_labels = payload["class_labels"]
    class_names = payload["class_names"]

    if not isinstance(class_labels, list):
        raise TypeError(
            "Manifest class_labels must be a list."
        )

    if not isinstance(class_names, list):
        raise TypeError(
            "Manifest class_names must be a list."
        )

    if len(class_labels) != len(class_names):
        raise ValueError(
            "Manifest class_labels and class_names "
            "must have equal lengths."
        )

    if len(class_labels) != 3:
        raise ValueError(
            "FootCast production models must expose "
            "three outcome classes."
        )

    if len(class_labels) != len(
        set(class_labels)
    ):
        raise ValueError(
            "Manifest class_labels contain duplicates."
        )

    if len(class_names) != len(
        set(class_names)
    ):
        raise ValueError(
            "Manifest class_names contain duplicates."
        )


def validate_ensemble_weights(
    payload: dict[str, Any],
) -> None:
    """Validate ensemble weights."""
    weights = payload["ensemble_weights"]

    if not isinstance(weights, dict):
        raise TypeError(
            "Manifest ensemble_weights must be an object."
        )

    if not weights:
        raise ValueError(
            "Manifest ensemble_weights cannot be empty."
        )

    numeric_weights: list[float] = []

    for name, value in weights.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Ensemble weight names must be "
                "non-empty strings."
            )

        if not isinstance(value, int | float):
            raise TypeError(
                f"Ensemble weight {name!r} must be numeric."
            )

        numeric_value = float(value)

        if not 0.0 <= numeric_value <= 1.0:
            raise ValueError(
                f"Ensemble weight {name!r} must be "
                "between zero and one."
            )

        numeric_weights.append(
            numeric_value
        )

    if abs(sum(numeric_weights) - 1.0) > 1e-9:
        raise ValueError(
            "Manifest ensemble weights must sum to one."
        )


def validate_bundle_path(
    payload: dict[str, Any],
) -> None:
    """Ensure the manifest points to the expected bundle."""
    bundle_path = payload["bundle_path"]

    if not isinstance(bundle_path, str):
        raise TypeError(
            "Manifest bundle_path must be a string."
        )

    declared = Path(bundle_path)

    if declared != MODEL_BUNDLE:
        raise ValueError(
            "Manifest bundle_path does not match the "
            "expected production bundle path."
        )


def validate_training_report() -> None:
    """Validate the production training report."""
    payload = read_json(
        TRAINING_REPORT
    )

    if not payload:
        raise ValueError(
            "Production training report is empty."
        )


def validate_manifest() -> None:
    """Validate the production model manifest."""
    payload = read_json(
        MODEL_MANIFEST
    )

    validate_manifest_fields(payload)
    validate_feature_contract(payload)
    validate_classes(payload)
    validate_ensemble_weights(payload)
    validate_bundle_path(payload)


def main() -> None:
    """Run all production artefact checks."""
    validate_required_paths()
    validate_manifest()
    validate_training_report()

    print("Production artefacts are valid.")
    print(f"Bundle: {MODEL_BUNDLE}")
    print(f"Manifest: {MODEL_MANIFEST}")
    print(f"Training report: {TRAINING_REPORT}")


if __name__ == "__main__":
    main()