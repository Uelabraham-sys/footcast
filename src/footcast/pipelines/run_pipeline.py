"""Run complete or partial FootCast pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final

import typer

from footcast.pipelines.orchestrator import (
    PipelineExecutionError,
    PipelineResult,
    PipelineStage,
    execute_pipeline,
    write_pipeline_result,
)

PROJECT_ROOT: Final[Path] = Path.cwd()

DEFAULT_RUN_DIRECTORY: Final[Path] = Path("artifacts/runs")

DEFAULT_LATEST_PATH: Final[Path] = Path("artifacts/runs/latest.json")

PYTHON_COMMAND: Final[tuple[str, ...]] = (
    "uv",
    "run",
    "python",
    "-m",
)

app = typer.Typer(help="Execute FootCast data and modelling pipelines.")


def module_stage(
    *,
    name: str,
    module: str,
    description: str,
    optional: bool = False,
) -> PipelineStage:
    """Create a Python-module pipeline stage."""
    return PipelineStage(
        name=name,
        command=(
            *PYTHON_COMMAND,
            module,
        ),
        description=description,
        optional=optional,
    )


DATA_STAGES: Final[tuple[PipelineStage, ...]] = (
    module_stage(
        name="historical_ingestion",
        module="footcast.ingestion.historical",
        description=("Ingest historical football match data."),
    ),
    module_stage(
        name="current_ingestion",
        module="footcast.ingestion.current",
        description=("Ingest current football data and fixtures."),
        optional=True,
    ),
    module_stage(
        name="clean_matches",
        module="footcast.processing.clean_matches",
        description=("Build canonical Silver match records."),
    ),
    module_stage(
        name="rolling_features",
        module="footcast.features.build_features",
        description=("Build leakage-safe rolling form features."),
    ),
    module_stage(
        name="elo_features",
        module="footcast.features.build_elo_features",
        description=("Build chronological Elo features."),
    ),
    module_stage(
        name="model_dataset",
        module="footcast.features.model_dataset",
        description=("Build the Gold modelling dataset."),
    ),
)

MODEL_STAGES: Final[tuple[PipelineStage, ...]] = (
    module_stage(
        name="evaluate_baselines",
        module="footcast.modelling.run_baselines",
        description=("Evaluate deterministic baseline models."),
    ),
    module_stage(
        name="train_logistic",
        module="footcast.modelling.train_logistic",
        description=("Select and train logistic regression."),
    ),
    module_stage(
        name="train_gradient_boosting",
        module=("footcast.modelling.train_gradient_boosting"),
        description=("Select, calibrate and evaluate gradient boosting."),
    ),
    module_stage(
        name="train_ensemble",
        module="footcast.modelling.train_ensemble",
        description=("Select the probability-ensemble weight."),
    ),
    module_stage(
        name="compare_day_4",
        module="footcast.modelling.day_4_comparison",
        description=("Build the final model comparison."),
    ),
)

PRODUCTION_STAGES: Final[tuple[PipelineStage, ...]] = (
    module_stage(
        name="fit_production",
        module="footcast.prediction.fit_production",
        description=("Fit and persist the production model bundle."),
    ),
    module_stage(
        name="predict_future",
        module="footcast.prediction.predict",
        description=("Generate future fixture predictions when available."),
        optional=True,
    ),
)

PIPELINES: Final[dict[str, tuple[PipelineStage, ...]]] = {
    "data": DATA_STAGES,
    "models": MODEL_STAGES,
    "production": PRODUCTION_STAGES,
    "full": (
        *DATA_STAGES,
        *MODEL_STAGES,
        *PRODUCTION_STAGES,
    ),
}


def run_selected_pipeline(
    *,
    pipeline_name: str,
    project_root: Path,
    run_directory: Path,
    latest_path: Path,
) -> PipelineResult:
    """Execute a named pipeline and persist its manifest."""
    stages = PIPELINES.get(pipeline_name)

    if stages is None:
        supported = ", ".join(sorted(PIPELINES))

        raise ValueError(
            f"Unknown pipeline {pipeline_name!r}. Supported pipelines: {supported}."
        )

    try:
        result = execute_pipeline(
            pipeline_name=pipeline_name,
            stages=stages,
            working_directory=project_root,
        )
    except PipelineExecutionError as error:
        write_pipeline_result(
            error.pipeline_result,
            run_directory=run_directory,
            latest_path=latest_path,
        )
        raise

    write_pipeline_result(
        result,
        run_directory=run_directory,
        latest_path=latest_path,
    )

    return result


@app.command()
def run(
    pipeline: Annotated[
        str,
        typer.Option(
            help=("Pipeline to execute: data, models, production or full."),
        ),
    ] = "full",
    project_root: Annotated[
        Path,
        typer.Option(
            help="FootCast repository root.",
        ),
    ] = PROJECT_ROOT,
    run_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for pipeline manifests.",
        ),
    ] = DEFAULT_RUN_DIRECTORY,
) -> None:
    """Execute one FootCast pipeline."""
    latest_path = run_directory / "latest.json"

    try:
        result = run_selected_pipeline(
            pipeline_name=pipeline,
            project_root=project_root,
            run_directory=run_directory,
            latest_path=latest_path,
        )
    except PipelineExecutionError as error:
        typer.echo(
            "PIPELINE FAILED",
            err=True,
        )
        typer.echo(
            f"Stage: {error.stage_result.name}",
            err=True,
        )
        typer.echo(
            error.stage_result.stderr,
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo("PIPELINE COMPLETE")
    typer.echo("=" * 78)
    typer.echo(f"Pipeline: {result.pipeline_name}")
    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Duration: {result.duration_seconds:.3f}s")


if __name__ == "__main__":
    app()
