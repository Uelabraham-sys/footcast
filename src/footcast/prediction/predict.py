"""Generate FootCast predictions for future fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import numpy as np
import polars as pl
import typer

from footcast.prediction.bundle import (
    PROBABILITY_COLUMNS,
    extract_feature_matrix,
    load_production_bundle,
    predict_bundle_probabilities,
)

DEFAULT_DATASET_PATH: Final[Path] = Path("data/gold/model_dataset.parquet")

DEFAULT_BUNDLE_PATH: Final[Path] = Path(
    "artifacts/models/production/footcast_bundle.joblib"
)

DEFAULT_PARQUET_OUTPUT: Final[Path] = Path(
    "artifacts/predictions/future_predictions.parquet"
)

DEFAULT_CSV_OUTPUT: Final[Path] = Path("artifacts/predictions/future_predictions.csv")

DEFAULT_AUDIT_OUTPUT: Final[Path] = Path(
    "artifacts/reports/production_prediction_audit.json"
)

app = typer.Typer(help="Generate FootCast predictions from a production bundle.")


def select_future_rows(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Select unlabelled fixtures in chronological order."""
    required = {
        "match_key",
        "kickoff_utc",
        "target",
    }

    missing = sorted(required - set(dataframe.columns))

    if missing:
        raise ValueError(f"Prediction dataset is missing columns: {missing}")

    future = dataframe.filter(pl.col("target").is_null()).sort(
        [
            "kickoff_utc",
            "match_key",
        ]
    )

    if future.is_empty():
        raise ValueError("No unlabelled future fixtures were found.")

    duplicate_count = (
        future.group_by("match_key").len().filter(pl.col("len") > 1).height
    )

    if duplicate_count > 0:
        raise ValueError("Future fixtures contain duplicate match keys.")

    return future


def build_prediction_output(
    future: pl.DataFrame,
    probabilities: np.ndarray,
) -> pl.DataFrame:
    """Combine fixture metadata and predicted probabilities."""
    if probabilities.shape != (
        future.height,
        3,
    ):
        raise ValueError("Prediction probability shape does not match fixtures.")

    metadata_columns = [
        column
        for column in (
            "match_key",
            "season",
            "kickoff_utc",
            "match_date",
            "home_team_id",
            "home_team",
            "away_team_id",
            "away_team",
        )
        if column in future.columns
    ]

    output = (
        future.select(metadata_columns)
        .with_columns(
            pl.Series(
                PROBABILITY_COLUMNS[0],
                probabilities[:, 0],
            ),
            pl.Series(
                PROBABILITY_COLUMNS[1],
                probabilities[:, 1],
            ),
            pl.Series(
                PROBABILITY_COLUMNS[2],
                probabilities[:, 2],
            ),
        )
        .with_columns(
            pl.concat_list(list(PROBABILITY_COLUMNS))
            .list.arg_max()
            .cast(pl.Int64)
            .alias("predicted_class"),
            pl.max_horizontal(list(PROBABILITY_COLUMNS)).alias("prediction_confidence"),
        )
        .with_columns(
            pl.when(pl.col("predicted_class") == 0)
            .then(pl.lit("away_win"))
            .when(pl.col("predicted_class") == 1)
            .then(pl.lit("draw"))
            .otherwise(pl.lit("home_win"))
            .alias("predicted_outcome"),
        )
    )

    return output


def validate_prediction_output(
    predictions: pl.DataFrame,
) -> None:
    """Validate generated future predictions."""
    if predictions.is_empty():
        raise ValueError("Generated predictions cannot be empty.")

    probability_sum = pl.sum_horizontal(list(PROBABILITY_COLUMNS))

    invalid_probability_rows = predictions.filter(
        (probability_sum < 1.0 - 1e-8) | (probability_sum > 1.0 + 1e-8)
    ).height

    if invalid_probability_rows > 0:
        raise ValueError("Generated probability rows do not sum to one.")

    null_count = predictions.select(
        pl.sum_horizontal(pl.all().null_count()).alias("null_count")
    )["null_count"].item()

    if not isinstance(null_count, int):
        raise TypeError("Prediction null count must be an integer.")

    if null_count > 0:
        raise ValueError("Generated predictions contain null values.")


def write_prediction_outputs(
    *,
    predictions: pl.DataFrame,
    parquet_output: Path,
    csv_output: Path,
    audit_output: Path,
    bundle_path: Path,
) -> None:
    """Write future predictions and an audit report."""
    parquet_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.write_parquet(
        parquet_output,
        compression="zstd",
        statistics=True,
    )

    predictions.write_csv(csv_output)

    minimum_kickoff = predictions["kickoff_utc"].min()

    maximum_kickoff = predictions["kickoff_utc"].max()

    confidence_summary = predictions.select(
        pl.col("prediction_confidence").mean().alias("mean_confidence"),
        pl.col("prediction_confidence").min().alias("minimum_confidence"),
        pl.col("prediction_confidence").max().alias("maximum_confidence"),
    ).row(
        0,
        named=True,
    )

    outcome_counts = (
        predictions.group_by("predicted_outcome")
        .len()
        .sort("predicted_outcome")
        .to_dicts()
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "bundle_path": str(bundle_path),
        "prediction_rows": predictions.height,
        "minimum_kickoff": str(minimum_kickoff),
        "maximum_kickoff": str(maximum_kickoff),
        "confidence": confidence_summary,
        "predicted_outcome_counts": (outcome_counts),
        "outputs": {
            "parquet": str(parquet_output),
            "csv": str(csv_output),
        },
    }

    audit_output.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def generate_future_predictions(
    *,
    dataset_path: Path,
    bundle_path: Path,
    parquet_output: Path,
    csv_output: Path,
    audit_output: Path,
) -> pl.DataFrame:
    """Load data and bundle, then generate future predictions."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prediction dataset was not found: {dataset_path}")

    dataframe = pl.read_parquet(dataset_path)

    bundle = load_production_bundle(bundle_path)

    future = select_future_rows(dataframe)

    features = extract_feature_matrix(
        future,
        bundle.feature_names,
    )

    probabilities = predict_bundle_probabilities(
        bundle,
        features,
    )

    predictions = build_prediction_output(
        future,
        probabilities,
    )

    validate_prediction_output(predictions)

    write_prediction_outputs(
        predictions=predictions,
        parquet_output=parquet_output,
        csv_output=csv_output,
        audit_output=audit_output,
        bundle_path=bundle_path,
    )

    return predictions


@app.command()
def run(
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to model-ready fixtures.",
        ),
    ] = DEFAULT_DATASET_PATH,
    bundle_path: Annotated[
        Path,
        typer.Option(
            help="Path to the production model bundle.",
        ),
    ] = DEFAULT_BUNDLE_PATH,
    parquet_output: Annotated[
        Path,
        typer.Option(
            help="Parquet prediction output.",
        ),
    ] = DEFAULT_PARQUET_OUTPUT,
    csv_output: Annotated[
        Path,
        typer.Option(
            help="CSV prediction output.",
        ),
    ] = DEFAULT_CSV_OUTPUT,
) -> None:
    """Generate predictions for unlabelled fixtures."""
    try:
        predictions = generate_future_predictions(
            dataset_path=dataset_path,
            bundle_path=bundle_path,
            parquet_output=parquet_output,
            csv_output=csv_output,
            audit_output=DEFAULT_AUDIT_OUTPUT,
        )
    except ValueError as error:
        if str(error) != ("No unlabelled future fixtures were found."):
            raise

        typer.echo(
            "No unlabelled future fixtures were found. "
            "Prediction generation was skipped."
        )
        return

    typer.echo("FUTURE PREDICTIONS")
    typer.echo("=" * 78)
    typer.echo(str(predictions))


if __name__ == "__main__":
    app()
