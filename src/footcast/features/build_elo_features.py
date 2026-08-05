"""Build FootCast Gold Elo feature datasets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer

from footcast.config import Settings, get_settings
from footcast.features.elo import (
    EloParameters,
    build_elo_history,
)

app = typer.Typer(help="Build chronological football Elo features.")


MATCH_ELO_COLUMNS: tuple[str, ...] = (
    "match_key",
    "home_elo_pre",
    "away_elo_pre",
    "elo_difference",
    "home_expected_score",
    "away_expected_score",
)


def load_silver_matches(
    silver_directory: Path,
) -> pl.DataFrame:
    """Load canonical Silver matches."""
    path = silver_directory / "matches.parquet"

    if not path.exists():
        raise FileNotFoundError(
            "Silver matches were not found. Run `make build-silver` first."
        )

    return pl.read_parquet(path)


def build_match_elo_features(
    matches: pl.DataFrame,
    elo_history: pl.DataFrame,
) -> pl.DataFrame:
    """Join pre-match Elo values onto canonical match rows."""
    elo_features = elo_history.select(*MATCH_ELO_COLUMNS)

    result = matches.join(
        elo_features,
        on="match_key",
        how="left",
        validate="1:1",
    )

    missing_elo = result.filter(
        pl.col("home_elo_pre").is_null() | pl.col("away_elo_pre").is_null()
    )

    if missing_elo.height:
        raise ValueError(f"Found {missing_elo.height} matches without Elo features.")

    return result.sort(["kickoff_utc", "match_key"])


def validate_elo_history(
    dataframe: pl.DataFrame,
) -> None:
    """Validate Elo history invariants."""
    if dataframe.is_empty():
        raise ValueError("Elo history is empty.")

    duplicate_matches = (
        dataframe.group_by("match_key").len().filter(pl.col("len") > 1).height
    )

    if duplicate_matches:
        raise ValueError(f"Found {duplicate_matches} duplicate Elo records.")

    invalid_probabilities = dataframe.filter(
        (pl.col("home_expected_score") < 0)
        | (pl.col("home_expected_score") > 1)
        | (pl.col("away_expected_score") < 0)
        | (pl.col("away_expected_score") > 1)
    )

    if invalid_probabilities.height:
        raise ValueError("Elo expected scores must be between zero and one.")

    invalid_sums = dataframe.filter(
        (pl.col("home_expected_score") + pl.col("away_expected_score") - 1.0).abs()
        > 1e-10
    )

    if invalid_sums.height:
        raise ValueError("Home and away Elo expectations must sum to one.")

    finished = dataframe.filter(pl.col("status") == "FINISHED")

    invalid_conservation = finished.filter(
        (
            (pl.col("home_elo_post") + pl.col("away_elo_post"))
            - (pl.col("home_elo_pre") + pl.col("away_elo_pre"))
        ).abs()
        > 1e-8
    )

    if invalid_conservation.height:
        raise ValueError("Elo updates must conserve total rating.")


def write_elo_datasets(
    elo_history: pl.DataFrame,
    match_features: pl.DataFrame,
    gold_directory: Path,
    parameters: EloParameters,
) -> None:
    """Write Elo history, match features and metadata."""
    gold_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = gold_directory / "elo_history.parquet"
    features_path = gold_directory / "match_elo_features.parquet"
    report_path = gold_directory / "elo_feature_report.json"

    elo_history.write_parquet(
        history_path,
        compression="zstd",
        statistics=True,
    )

    match_features.write_parquet(
        features_path,
        compression="zstd",
        statistics=True,
    )

    finished = elo_history.filter(pl.col("status") == "FINISHED")

    all_pre_match_ratings = pl.concat(
        [
            elo_history.select(pl.col("home_elo_pre").alias("rating")),
            elo_history.select(pl.col("away_elo_pre").alias("rating")),
        ]
    )

    minimum_rating = all_pre_match_ratings.select(
        pl.col("rating").min().alias("minimum")
    )["minimum"].item()

    maximum_rating = all_pre_match_ratings.select(
        pl.col("rating").max().alias("maximum")
    )["maximum"].item()

    if not isinstance(minimum_rating, (int, float)):
        raise TypeError("Minimum Elo rating must be numeric.")

    if not isinstance(maximum_rating, (int, float)):
        raise TypeError("Maximum Elo rating must be numeric.")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "match_count": elo_history.height,
        "finished_match_count": finished.height,
        "team_count": pl.concat(
            [
                elo_history.select(pl.col("home_team_id").alias("team_id")),
                elo_history.select(pl.col("away_team_id").alias("team_id")),
            ]
        )["team_id"].n_unique(),
        "initial_rating": parameters.initial_rating,
        "k_factor": parameters.k_factor,
        "home_advantage": parameters.home_advantage,
        "minimum_pre_match_rating": float(minimum_rating),
        "maximum_pre_match_rating": float(maximum_rating),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def build_and_write_elo_features(
    settings: Settings,
    parameters: EloParameters | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build, validate and write Elo feature datasets."""
    elo_parameters = parameters or EloParameters()
    matches = load_silver_matches(settings.silver_directory)

    elo_history = build_elo_history(
        matches,
        parameters=elo_parameters,
    )
    validate_elo_history(elo_history)

    match_features = build_match_elo_features(
        matches=matches,
        elo_history=elo_history,
    )

    write_elo_datasets(
        elo_history=elo_history,
        match_features=match_features,
        gold_directory=settings.gold_directory,
        parameters=elo_parameters,
    )

    return elo_history, match_features


@app.command()
def run(
    initial_rating: float = typer.Option(
        1500.0,
        help="Initial Elo rating assigned to unseen teams.",
    ),
    k_factor: float = typer.Option(
        20.0,
        help="Rating sensitivity after each completed match.",
    ),
    home_advantage: float = typer.Option(
        60.0,
        help="Home advantage in Elo rating points.",
    ),
) -> None:
    """Build FootCast Gold Elo features."""
    settings = get_settings()

    parameters = EloParameters(
        initial_rating=initial_rating,
        k_factor=k_factor,
        home_advantage=home_advantage,
    )

    elo_history, match_features = build_and_write_elo_features(
        settings=settings,
        parameters=parameters,
    )

    typer.echo(
        f"Built {elo_history.height} Elo records and "
        f"{match_features.height} match feature rows."
    )


if __name__ == "__main__":
    app()
