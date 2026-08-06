"""CI smoke tests for production artefacts."""

from pathlib import Path

import polars as pl

from footcast.prediction.bundle import (
    load_production_bundle,
    save_production_bundle,
)
from footcast.prediction.predict import (
    generate_future_predictions,
)
from tests.helpers import (
    create_bundle,
    create_future_dataset,
)


def test_ci_bundle_and_prediction_smoke(
    tmp_path: Path,
) -> None:
    """A saved bundle should score synthetic fixtures."""
    dataset_path = tmp_path / "model_dataset.parquet"
    bundle_path = tmp_path / "footcast_bundle.joblib"
    parquet_output = tmp_path / "future_predictions.parquet"
    csv_output = tmp_path / "future_predictions.csv"
    audit_output = tmp_path / "prediction_audit.json"

    create_future_dataset().write_parquet(dataset_path)

    save_production_bundle(
        create_bundle(),
        bundle_path,
    )

    loaded = load_production_bundle(bundle_path)

    assert loaded.feature_names

    predictions = generate_future_predictions(
        dataset_path=dataset_path,
        bundle_path=bundle_path,
        parquet_output=parquet_output,
        csv_output=csv_output,
        audit_output=audit_output,
    )

    assert predictions.height > 0
    assert parquet_output.exists()
    assert csv_output.exists()
    assert audit_output.exists()

    saved = pl.read_parquet(parquet_output)

    assert saved.height == predictions.height

from footcast.pipelines.orchestrator import (
    PipelineStage,
    execute_pipeline,
    write_pipeline_result,
)


def test_ci_pipeline_manifest_smoke(
    tmp_path: Path,
) -> None:
    """A successful pipeline should write its manifest."""
    stage = PipelineStage(
        name="smoke",
        command=(
            "python",
            "-c",
            "print('pipeline smoke passed')",
        ),
        description="CI pipeline smoke stage.",
    )

    result = execute_pipeline(
        pipeline_name="ci-smoke",
        stages=(stage,),
        working_directory=tmp_path,
    )

    manifest_path = write_pipeline_result(
        result,
        run_directory=tmp_path / "runs",
        latest_path=(
            tmp_path / "runs" / "latest.json"
        ),
    )

    assert result.status == "success"
    assert manifest_path.exists()
    assert (
        tmp_path / "runs" / "latest.json"
    ).exists()