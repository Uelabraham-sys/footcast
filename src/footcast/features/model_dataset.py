"""Build the final FootCast modelling dataset."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
import typer

from footcast.config import Settings, get_settings
from footcast.features.model_validation import (
    validate_model_dataset,
)

app = typer.Typer(help="Build the final football modelling dataset.")

DEFAULT_VALIDATION_SEASON: Final[str] = "2024/25"
DEFAULT_TEST_SEASON: Final[str] = "2025/26"


def load_gold_dataset(
    gold_directory: Path,
    filename: str,
) -> pl.DataFrame:
    """Load a required Gold Parquet dataset."""
    path = gold_directory / filename

    if not path.exists():
        raise FileNotFoundError(f"Required Gold dataset was not found: {path}")

    return pl.read_parquet(path)


def join_form_and_elo_features(
    form_features: pl.DataFrame,
    elo_features: pl.DataFrame,
) -> pl.DataFrame:
    """Combine rolling-form and Elo match features."""
    elo_columns = elo_features.select(
        "match_key",
        "home_elo_pre",
        "away_elo_pre",
        "elo_difference",
        "home_expected_score",
        "away_expected_score",
    )

    joined = form_features.join(
        elo_columns,
        on="match_key",
        how="inner",
        validate="1:1",
    )

    if joined.height != form_features.height:
        missing_count = form_features.height - joined.height

        raise ValueError(f"{missing_count} form rows had no matching Elo row.")

    return joined


def add_target_column(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Encode away win, draw and home win as 0, 1 and 2."""
    return dataframe.with_columns(
        pl.when(pl.col("full_time_result") == "A")
        .then(pl.lit(0))
        .when(pl.col("full_time_result") == "D")
        .then(pl.lit(1))
        .when(pl.col("full_time_result") == "H")
        .then(pl.lit(2))
        .otherwise(pl.lit(None))
        .cast(pl.Int64)
        .alias("target")
    )


def add_cold_start_features(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Identify teams with no prior match history."""
    return dataframe.with_columns(
        (pl.col("home_matches_played_before") == 0).alias("home_is_cold_start"),
        (pl.col("away_matches_played_before") == 0).alias("away_is_cold_start"),
        (pl.col("home_matches_played_before") < 5).alias("home_has_limited_history"),
        (pl.col("away_matches_played_before") < 5).alias("away_has_limited_history"),
    )


def assign_chronological_splits(
    dataframe: pl.DataFrame,
    validation_season: str,
    test_season: str,
) -> pl.DataFrame:
    """Assign split labels according to complete seasons."""
    available_seasons = set(dataframe["season"].drop_nulls().to_list())

    if validation_season not in available_seasons:
        raise ValueError(
            f"Validation season {validation_season!r} is not present in the dataset."
        )

    if test_season not in available_seasons:
        raise ValueError(f"Test season {test_season!r} is not present in the dataset.")

    season_order = (
        dataframe.select(
            "season",
            pl.col("kickoff_utc").min().over("season").alias("season_start"),
        )
        .unique()
        .sort("season_start")
    )

    season_positions = {
        row["season"]: index
        for index, row in enumerate(season_order.iter_rows(named=True))
    }

    validation_position = season_positions[validation_season]
    test_position = season_positions[test_season]

    if validation_position >= test_position:
        raise ValueError("Validation season must occur before test season.")

    split_mapping: dict[str, str] = {}

    for season, position in season_positions.items():
        if position < validation_position:
            split_mapping[season] = "train"
        elif position == validation_position:
            split_mapping[season] = "validation"
        elif position == test_position:
            split_mapping[season] = "test"
        else:
            split_mapping[season] = "future"

    return dataframe.with_columns(
        pl.col("season").replace(split_mapping).alias("split")
    )


def select_final_columns(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Select and order final model columns."""
    preferred_columns = (
        "match_key",
        "season",
        "competition",
        "kickoff_utc",
        "match_date",
        "status",
        "split",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "home_goals",
        "away_goals",
        "full_time_result",
        "target",
        "home_matches_played_before",
        "away_matches_played_before",
        "home_is_cold_start",
        "away_is_cold_start",
        "home_has_limited_history",
        "away_has_limited_history",
        "home_points_last_5",
        "away_points_last_5",
        "form_points_difference",
        "home_goals_for_last_5",
        "away_goals_for_last_5",
        "recent_attack_difference",
        "home_goals_against_last_5",
        "away_goals_against_last_5",
        "recent_defence_difference",
        "home_goal_difference_last_5",
        "away_goal_difference_last_5",
        "recent_goal_difference_difference",
        "home_wins_last_5",
        "away_wins_last_5",
        "home_draws_last_5",
        "away_draws_last_5",
        "home_losses_last_5",
        "away_losses_last_5",
        "home_average_points_last_5",
        "away_average_points_last_5",
        "average_form_difference",
        "home_days_since_previous_match",
        "away_days_since_previous_match",
        "rest_days_difference",
        "home_elo_pre",
        "away_elo_pre",
        "elo_difference",
        "home_expected_score",
        "away_expected_score",
    )

    existing_columns = [
        column for column in preferred_columns if column in dataframe.columns
    ]

    return dataframe.select(existing_columns)


def build_model_dataset(
    settings: Settings,
    validation_season: str,
    test_season: str,
) -> pl.DataFrame:
    """Build the complete modelling dataset."""
    form_features = load_gold_dataset(
        settings.gold_directory,
        "match_form_features.parquet",
    )
    elo_features = load_gold_dataset(
        settings.gold_directory,
        "match_elo_features.parquet",
    )

    return (
        join_form_and_elo_features(
            form_features=form_features,
            elo_features=elo_features,
        )
        .pipe(add_target_column)
        .pipe(add_cold_start_features)
        .pipe(
            assign_chronological_splits,
            validation_season=validation_season,
            test_season=test_season,
        )
        .pipe(select_final_columns)
        .sort(["kickoff_utc", "match_key"])
    )


def write_model_dataset(
    dataframe: pl.DataFrame,
    gold_directory: Path,
    validation_season: str,
    test_season: str,
) -> None:
    """Write the model dataset and metadata reports."""
    gold_directory.mkdir(parents=True, exist_ok=True)

    dataset_path = gold_directory / "model_dataset.parquet"
    report_path = gold_directory / "model_dataset_report.json"
    split_path = gold_directory / "split_metadata.json"

    dataframe.write_parquet(
        dataset_path,
        compression="zstd",
        statistics=True,
    )

    split_summary = (
        dataframe.group_by("split")
        .agg(
            pl.len().alias("rows"),
            pl.col("kickoff_utc").min().alias("first_match"),
            pl.col("kickoff_utc").max().alias("last_match"),
            pl.col("season").n_unique().alias("seasons"),
        )
        .sort("first_match")
    )

    target_distribution = (
        dataframe.filter(pl.col("target").is_not_null())
        .group_by(["split", "target"])
        .len()
        .sort(["split", "target"])
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "row_count": dataframe.height,
        "column_count": dataframe.width,
        "season_count": dataframe["season"].n_unique(),
        "team_count": pl.concat(
            [
                dataframe.select(pl.col("home_team_id").alias("team_id")),
                dataframe.select(pl.col("away_team_id").alias("team_id")),
            ]
        )["team_id"].n_unique(),
        "completed_match_count": dataframe.filter(
            pl.col("target").is_not_null()
        ).height,
        "cold_start_home_count": dataframe.filter(pl.col("home_is_cold_start")).height,
        "cold_start_away_count": dataframe.filter(pl.col("away_is_cold_start")).height,
        "split_counts": split_summary.to_dicts(),
        "target_distribution": target_distribution.to_dicts(),
    }

    split_metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy": "complete-season chronological split",
        "validation_season": validation_season,
        "test_season": test_season,
        "split_summary": split_summary.to_dicts(),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    split_path.write_text(
        json.dumps(
            split_metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def build_and_write_model_dataset(
    settings: Settings,
    validation_season: str,
    test_season: str,
) -> pl.DataFrame:
    """Build, validate and write the modelling dataset."""
    dataframe = build_model_dataset(
        settings=settings,
        validation_season=validation_season,
        test_season=test_season,
    )

    validate_model_dataset(dataframe)

    write_model_dataset(
        dataframe=dataframe,
        gold_directory=settings.gold_directory,
        validation_season=validation_season,
        test_season=test_season,
    )

    return dataframe


@app.command()
def run(
    validation_season: str = typer.Option(
        DEFAULT_VALIDATION_SEASON,
        help="Season used for model selection.",
    ),
    test_season: str = typer.Option(
        DEFAULT_TEST_SEASON,
        help="Final held-out test season.",
    ),
) -> None:
    """Build the final FootCast modelling dataset."""
    dataframe = build_and_write_model_dataset(
        settings=get_settings(),
        validation_season=validation_season,
        test_season=test_season,
    )

    typer.echo(
        f"Built model dataset with {dataframe.height} rows "
        f"and {dataframe.width} columns."
    )


if __name__ == "__main__":
    app()
