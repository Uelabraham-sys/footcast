"""Deterministic pipeline orchestration for FootCast."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

RUN_SCHEMA_VERSION: Final[str] = "1.0.0"


class StageStatus(StrEnum):
    """Supported pipeline-stage outcomes."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class PipelineStage:
    """One executable FootCast pipeline stage."""

    name: str
    command: tuple[str, ...]
    description: str
    optional: bool = False


@dataclass(frozen=True)
class StageResult:
    """Recorded outcome of one pipeline stage."""

    name: str
    description: str
    command: tuple[str, ...]
    optional: bool
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    return_code: int | None
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PipelineResult:
    """Complete pipeline-execution result."""

    schema_version: str
    run_id: str
    pipeline_name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    stages: tuple[StageResult, ...]


class PipelineExecutionError(RuntimeError):
    """Raised when a required pipeline stage fails."""

    def __init__(
        self,
        *,
        stage_result: StageResult,
        pipeline_result: PipelineResult,
    ) -> None:
        """Initialise the pipeline execution error."""
        self.stage_result = stage_result
        self.pipeline_result = pipeline_result

        super().__init__(f"Required pipeline stage failed: {stage_result.name}")


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def create_run_id(
    *,
    timestamp: datetime | None = None,
) -> str:
    """Create a sortable pipeline-run identifier."""
    value = timestamp or utc_now()

    return value.strftime("%Y%m%dT%H%M%S%fZ")


def validate_stage(
    stage: PipelineStage,
) -> None:
    """Validate a pipeline-stage definition."""
    if not stage.name.strip():
        raise ValueError("Pipeline stage name cannot be empty.")

    if not stage.command:
        raise ValueError(f"Pipeline stage {stage.name!r} has no command.")

    if any(not argument.strip() for argument in stage.command):
        raise ValueError(
            f"Pipeline stage {stage.name!r} contains an empty command argument."
        )


def execute_stage(
    stage: PipelineStage,
    *,
    working_directory: Path,
    environment: dict[str, str] | None = None,
) -> StageResult:
    """Execute one pipeline stage and capture its output."""
    validate_stage(stage)

    started = utc_now()
    monotonic_start = time.monotonic()

    completed = subprocess.run(
        stage.command,
        cwd=working_directory,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    finished = utc_now()
    duration = time.monotonic() - monotonic_start

    status = StageStatus.SUCCESS if completed.returncode == 0 else StageStatus.FAILED

    return StageResult(
        name=stage.name,
        description=stage.description,
        command=stage.command,
        optional=stage.optional,
        status=status.value,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_seconds=round(
            duration,
            6,
        ),
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def skipped_stage_result(
    stage: PipelineStage,
    *,
    reason: str,
) -> StageResult:
    """Create a skipped-stage result."""
    timestamp = utc_now().isoformat()

    return StageResult(
        name=stage.name,
        description=stage.description,
        command=stage.command,
        optional=stage.optional,
        status=StageStatus.SKIPPED.value,
        started_at=timestamp,
        finished_at=timestamp,
        duration_seconds=0.0,
        return_code=None,
        stdout="",
        stderr=reason,
    )


def execute_pipeline(
    *,
    pipeline_name: str,
    stages: Sequence[PipelineStage],
    working_directory: Path,
    environment: dict[str, str] | None = None,
    continue_after_optional_failure: bool = True,
) -> PipelineResult:
    """Execute ordered stages and return a run result."""
    if not pipeline_name.strip():
        raise ValueError("Pipeline name cannot be empty.")

    if not stages:
        raise ValueError("Pipeline requires at least one stage.")

    if not working_directory.exists():
        raise FileNotFoundError(
            f"Pipeline working directory was not found: {working_directory}"
        )

    started = utc_now()
    monotonic_start = time.monotonic()
    run_id = create_run_id(timestamp=started)

    results: list[StageResult] = []
    required_failure: StageResult | None = None

    for stage in stages:
        if required_failure is not None:
            results.append(
                skipped_stage_result(
                    stage,
                    reason=("Skipped because an earlier required stage failed."),
                )
            )
            continue

        result = execute_stage(
            stage,
            working_directory=working_directory,
            environment=environment,
        )

        results.append(result)

        if result.status != StageStatus.FAILED.value:
            continue

        if stage.optional:
            if continue_after_optional_failure:
                continue

        required_failure = result

    finished = utc_now()
    duration = time.monotonic() - monotonic_start

    pipeline_status = (
        StageStatus.FAILED if required_failure is not None else StageStatus.SUCCESS
    )

    pipeline_result = PipelineResult(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=run_id,
        pipeline_name=pipeline_name,
        status=pipeline_status.value,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_seconds=round(
            duration,
            6,
        ),
        stages=tuple(results),
    )

    if required_failure is not None:
        raise PipelineExecutionError(
            stage_result=required_failure,
            pipeline_result=pipeline_result,
        )
    return pipeline_result


def write_pipeline_result(
    result: PipelineResult,
    *,
    run_directory: Path,
    latest_path: Path,
) -> Path:
    """Persist a pipeline run and latest-run pointer."""
    destination = run_directory / result.run_id / "pipeline_manifest.json"

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(result)

    destination.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    latest_payload = {
        "schema_version": result.schema_version,
        "run_id": result.run_id,
        "pipeline_name": result.pipeline_name,
        "status": result.status,
        "manifest_path": str(destination),
        "started_at": result.started_at,
        "finished_at": result.finished_at,
    }

    latest_path.write_text(
        json.dumps(
            latest_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    return destination
