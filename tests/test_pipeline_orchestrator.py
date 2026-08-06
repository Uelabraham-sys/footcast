"""Tests for FootCast pipeline orchestration."""

from pathlib import Path

import pytest

from footcast.pipelines.orchestrator import (
    PipelineExecutionError,
    PipelineStage,
    StageStatus,
    execute_pipeline,
    write_pipeline_result,
)


def python_stage(
    *,
    name: str,
    source: str,
    optional: bool = False,
) -> PipelineStage:
    """Create a small Python subprocess stage."""
    return PipelineStage(
        name=name,
        command=(
            "python",
            "-c",
            source,
        ),
        description=f"Test stage {name}.",
        optional=optional,
    )


def test_successful_pipeline_executes_all_stages(
    tmp_path: Path,
) -> None:
    """Every stage should run when all commands succeed."""
    result = execute_pipeline(
        pipeline_name="test",
        stages=(
            python_stage(
                name="first",
                source="print('first')",
            ),
            python_stage(
                name="second",
                source="print('second')",
            ),
        ),
        working_directory=tmp_path,
    )

    assert result.status == (StageStatus.SUCCESS.value)

    assert [stage.status for stage in result.stages] == [
        StageStatus.SUCCESS.value,
        StageStatus.SUCCESS.value,
    ]


def test_required_failure_skips_later_stages(
    tmp_path: Path,
) -> None:
    """A required failure should prevent later execution."""
    with pytest.raises(PipelineExecutionError) as error_info:
        execute_pipeline(
            pipeline_name="test",
            stages=(
                python_stage(
                    name="failure",
                    source=("raise SystemExit(2)"),
                ),
                python_stage(
                    name="never",
                    source="print('never')",
                ),
            ),
            working_directory=tmp_path,
        )

    result = error_info.value.pipeline_result

    assert result.status == (StageStatus.FAILED.value)

    assert result.stages[0].status == (StageStatus.FAILED.value)

    assert result.stages[1].status == (StageStatus.SKIPPED.value)


def test_optional_failure_does_not_stop_pipeline(
    tmp_path: Path,
) -> None:
    """An optional stage may fail without stopping execution."""
    result = execute_pipeline(
        pipeline_name="test",
        stages=(
            python_stage(
                name="optional_failure",
                source="raise SystemExit(3)",
                optional=True,
            ),
            python_stage(
                name="success",
                source="print('success')",
            ),
        ),
        working_directory=tmp_path,
    )

    assert result.status == (StageStatus.SUCCESS.value)

    assert result.stages[0].status == (StageStatus.FAILED.value)

    assert result.stages[1].status == (StageStatus.SUCCESS.value)


def test_pipeline_manifest_is_written(
    tmp_path: Path,
) -> None:
    """Pipeline results should persist as JSON."""
    result = execute_pipeline(
        pipeline_name="test",
        stages=(
            python_stage(
                name="success",
                source="print('success')",
            ),
        ),
        working_directory=tmp_path,
    )

    manifest = write_pipeline_result(
        result,
        run_directory=tmp_path / "runs",
        latest_path=(tmp_path / "runs" / "latest.json"),
    )

    assert manifest.exists()
    assert (tmp_path / "runs" / "latest.json").exists()
