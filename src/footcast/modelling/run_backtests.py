"""Run expanding-window FootCast model backtests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import polars as pl
import typer

from footcast.modelling.backtesting import (
    BacktestFold,
    build_backtest_folds,
)
from footcast.modelling.baselines import (
    frequency_probabilities,
)
from footcast.modelling.dataset import (
    MODEL_FEATURE_COLUMNS,
    load_model_dataset,
)
from footcast.modelling.evaluation import (
    create_prediction_frame,
    write_prediction_frame,
)
from footcast.modelling.gradient_boosting import (
    HGBParameters,
    fit_hgb_classifier,
)
from footcast.modelling.gradient_boosting import (
    ordered_predict_proba as hgb_predict_proba,
)
from footcast.modelling.logistic import (
    fit_logistic_pipeline,
)
from footcast.modelling.logistic import (
    ordered_predict_proba as logistic_predict_proba,
)
from footcast.modelling.metrics import (
    ClassificationMetrics,
    evaluate_probabilities,
)

DEFAULT_DATASET_PATH: Final[Path] = Path("data/gold/model_dataset.parquet")
DEFAULT_REPORT_DIRECTORY: Final[Path] = Path("artifacts/reports")
DEFAULT_PREDICTION_DIRECTORY: Final[Path] = Path("artifacts/predictions/backtests")

DEFAULT_LOGISTIC_C: Final[float] = 0.1

DEFAULT_HGB_PARAMETERS: Final[HGBParameters] = HGBParameters(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=5.0,
)

app = typer.Typer(help="Run expanding-window model backtests.")


def metric_record(
    *,
    fold: BacktestFold,
    model_name: str,
    metrics: ClassificationMetrics,
) -> dict[str, str | int | float]:
    """Create one flat fold-metric record."""
    return {
        "fold_number": fold.fold_number,
        "model": model_name,
        "evaluation_season": (fold.evaluation_season),
        "training_season_count": len(fold.training_seasons),
        "training_rows": int(fold.train_target.shape[0]),
        "evaluation_rows": int(fold.evaluation_target.shape[0]),
        **metrics.to_dict(),
    }


def evaluate_fold_probabilities(
    *,
    fold: BacktestFold,
    model_name: str,
    probabilities: object,
    prediction_directory: Path,
) -> ClassificationMetrics:
    """Evaluate and persist one fold's predictions."""
    import numpy as np
    from numpy.typing import NDArray

    probability_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    typed_probabilities: NDArray[np.float64] = probability_array

    metrics = evaluate_probabilities(
        fold.evaluation_target,
        typed_probabilities,
    )

    predictions = create_prediction_frame(
        fold.evaluation_metadata,
        typed_probabilities,
    ).with_columns(
        pl.lit(fold.fold_number).alias("fold_number"),
        pl.lit(model_name).alias("model"),
    )

    write_prediction_frame(
        predictions,
        prediction_directory
        / (
            f"fold_{fold.fold_number:02d}_"
            f"{fold.evaluation_season.replace('/', '-')}_"
            f"{model_name}.parquet"
        ),
    )

    return metrics


def evaluate_backtest_fold(
    fold: BacktestFold,
    *,
    prediction_directory: Path,
    logistic_c: float,
    hgb_parameters: HGBParameters,
) -> list[dict[str, str | int | float]]:
    """Fit and evaluate every model for one fold."""
    records: list[dict[str, str | int | float]] = []

    frequency = frequency_probabilities(
        fold.train_target,
        row_count=fold.evaluation_target.shape[0],
    )

    frequency_metrics = evaluate_fold_probabilities(
        fold=fold,
        model_name="frequency",
        probabilities=frequency,
        prediction_directory=prediction_directory,
    )

    records.append(
        metric_record(
            fold=fold,
            model_name="frequency",
            metrics=frequency_metrics,
        )
    )

    logistic = fit_logistic_pipeline(
        features=fold.train_features,
        target=fold.train_target,
        regularisation_strength=logistic_c,
        class_weight=None,
    )

    logistic_probabilities = logistic_predict_proba(
        logistic,
        fold.evaluation_features,
    )

    logistic_metrics = evaluate_fold_probabilities(
        fold=fold,
        model_name="logistic_regression",
        probabilities=logistic_probabilities,
        prediction_directory=prediction_directory,
    )

    records.append(
        metric_record(
            fold=fold,
            model_name="logistic_regression",
            metrics=logistic_metrics,
        )
    )

    hgb = fit_hgb_classifier(
        features=fold.train_features,
        target=fold.train_target,
        parameters=hgb_parameters,
    )

    hgb_probabilities = hgb_predict_proba(
        hgb,
        fold.evaluation_features,
    )

    hgb_metrics = evaluate_fold_probabilities(
        fold=fold,
        model_name="hist_gradient_boosting",
        probabilities=hgb_probabilities,
        prediction_directory=prediction_directory,
    )

    records.append(
        metric_record(
            fold=fold,
            model_name="hist_gradient_boosting",
            metrics=hgb_metrics,
        )
    )

    return records


def summarise_backtests(
    fold_metrics: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate model performance across folds."""
    return (
        fold_metrics.group_by("model")
        .agg(
            pl.len().alias("folds"),
            pl.col("accuracy").mean().alias("mean_accuracy"),
            pl.col("accuracy").std().alias("std_accuracy"),
            pl.col("balanced_accuracy").mean().alias("mean_balanced_accuracy"),
            pl.col("macro_f1").mean().alias("mean_macro_f1"),
            pl.col("log_loss").mean().alias("mean_log_loss"),
            pl.col("log_loss").std().alias("std_log_loss"),
            pl.col("brier_score").mean().alias("mean_brier_score"),
            pl.col("ranked_probability_score")
            .mean()
            .alias("mean_ranked_probability_score"),
        )
        .sort(
            [
                "mean_log_loss",
                "mean_brier_score",
            ]
        )
    )


def write_backtest_reports(
    *,
    fold_metrics: pl.DataFrame,
    summary: pl.DataFrame,
    report_directory: Path,
) -> None:
    """Write fold-level and aggregate backtest reports."""
    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics.write_parquet(
        report_directory / "backtest_fold_metrics.parquet",
        compression="zstd",
        statistics=True,
    )

    summary.write_parquet(
        report_directory / "backtest_summary.parquet",
        compression="zstd",
        statistics=True,
    )

    best = summary.row(0, named=True)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": ("expanding-window complete-season backtesting"),
        "primary_metric": "mean_log_loss",
        "best_model": best.get("model"),
        "fold_count": fold_metrics["fold_number"].n_unique(),
        "fold_metrics": fold_metrics.to_dicts(),
        "summary": summary.to_dicts(),
    }

    (report_directory / "backtest_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def run_backtests(
    *,
    dataset_path: Path,
    report_directory: Path,
    prediction_directory: Path,
    minimum_training_seasons: int = 2,
    maximum_evaluation_seasons: int | None = 4,
    feature_columns: tuple[
        str,
        ...,
    ] = MODEL_FEATURE_COLUMNS,
    logistic_c: float = DEFAULT_LOGISTIC_C,
    hgb_parameters: HGBParameters = (DEFAULT_HGB_PARAMETERS),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run and report expanding-window model backtests."""
    dataframe = load_model_dataset(dataset_path)

    folds = build_backtest_folds(
        dataframe,
        minimum_training_seasons=(minimum_training_seasons),
        maximum_evaluation_seasons=(maximum_evaluation_seasons),
        feature_columns=feature_columns,
    )

    records: list[dict[str, str | int | float]] = []

    for fold in folds:
        records.extend(
            evaluate_backtest_fold(
                fold,
                prediction_directory=(prediction_directory),
                logistic_c=logistic_c,
                hgb_parameters=hgb_parameters,
            )
        )

    fold_metrics = pl.DataFrame(records).sort(["fold_number", "log_loss"])

    summary = summarise_backtests(fold_metrics)

    write_backtest_reports(
        fold_metrics=fold_metrics,
        summary=summary,
        report_directory=report_directory,
    )

    return fold_metrics, summary


@app.command()
def run(
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to the Gold model dataset.",
        ),
    ] = DEFAULT_DATASET_PATH,
    report_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for backtest reports.",
        ),
    ] = DEFAULT_REPORT_DIRECTORY,
    prediction_directory: Annotated[
        Path,
        typer.Option(
            help="Directory for fold predictions.",
        ),
    ] = DEFAULT_PREDICTION_DIRECTORY,
    minimum_training_seasons: Annotated[
        int,
        typer.Option(
            min=1,
            help=("Minimum seasons required before the first evaluation fold."),
        ),
    ] = 2,
    maximum_evaluation_seasons: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum number of recent evaluation seasons.",
        ),
    ] = 4,
) -> None:
    """Run expanding-window model backtests."""
    _, summary = run_backtests(
        dataset_path=dataset_path,
        report_directory=report_directory,
        prediction_directory=prediction_directory,
        minimum_training_seasons=(minimum_training_seasons),
        maximum_evaluation_seasons=(maximum_evaluation_seasons),
    )

    typer.echo("BACKTEST SUMMARY")
    typer.echo("=" * 78)
    typer.echo(str(summary))


if __name__ == "__main__":
    app()
