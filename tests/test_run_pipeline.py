"""Tests for named FootCast pipeline execution."""

from pathlib import Path

import pytest

from footcast.pipelines.orchestrator import (
    PipelineStage,
)
from footcast.pipelines.run_pipeline import (
    PIPELINES,
    run_selected_pipeline,
)


def test_named_pipeline_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named pipeline should execute and persist its run."""
    stage = PipelineStage(
        name="success",
        command=(
            "python",
            "-c",
            "print('success')",
        ),
        description="Successful test stage.",
    )

    monkeypatch.setitem(
        PIPELINES,
        "test",
        (stage,),
    )

    result = run_selected_pipeline(
        pipeline_name="test",
        project_root=tmp_path,
        run_directory=tmp_path / "runs",
        latest_path=(tmp_path / "runs" / "latest.json"),
    )

    assert result.status == "success"

    manifests = list((tmp_path / "runs").glob("*/pipeline_manifest.json"))

    assert len(manifests) == 1


def test_unknown_pipeline_fails(
    tmp_path: Path,
) -> None:
    """Unknown pipeline names should fail explicitly."""
    with pytest.raises(
        ValueError,
        match="Unknown pipeline",
    ):
        run_selected_pipeline(
            pipeline_name="unknown",
            project_root=tmp_path,
            run_directory=tmp_path / "runs",
            latest_path=(tmp_path / "runs" / "latest.json"),
        )
