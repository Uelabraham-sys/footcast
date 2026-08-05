"""Build FootCast Gold rolling-form feature datasets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer

from footcast.config import Settings, get_settings
from footcast.features.form import build_team_form_features
from footcast.features.validation import (
    validate_match_form_features,
    validate_team_match_history,
)

app = typer.Typer(help="Build leakage-safe football form features.")


TEAM_FEATURE_COLUMNS: tuple[str, ...] = (
    "matches_played_before",
    "points_last_5",
    "goals_for_last_5",
    "goals_against_last_5",
    "goal_difference_last_5",
    "wins_last_5",
    "draws_last_5",
    "losses_last_5",
    "average_points_last_5",
    "days_since_previous_match",
)


def load_silver_matches(silver_directory: Path) -> pl.DataFrame:
    """Load the canonical Silver match dataset."""
    path = silver_directory / "matches.parquet"

    if not path.exists():
        raise FileNotFoundError(
            "Silver matches were not found. Run `make build-silver` first."
        )

    return pl.read_parquet(path)


def select_team_features(
    team_features: pl.DataFrame,
    venue: str,
    prefix: str,
) -> pl.DataFrame:
    """Select and prefix one venue side's team features."""
    return team_features.filter(pl.col("venue") == venue).select(
        "match_key",
        pl.col("team_id").alias(f"{prefix}_team_feature_id"),
        *[
            pl.col(column).alias(f"{prefix}_{column}")
            for column in TEAM_FEATURE_COLUMNS
        ],
    )


def build_match_form_features(
    matches: pl.DataFrame,
    team_features: pl.DataFrame,
) -> pl.DataFrame:
    """Join long-form team features onto match rows."""
    base = matches.select(
        "match_key",
        "season",
        "competition",
        "kickoff_utc",
        "match_date",
        "status",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "home_goals",
        "away_goals",
        "full_time_result",
        "home_points",
        "away_points",
    )

    home_features = select_team_features(
        team_features,
        venue="HOME",
        prefix="home",
    )
    away_features = select_team_features(
        team_features,
        venue="AWAY",
        prefix="away",
    )

    result = (
        base.join(
            home_features,
            on="match_key",
            how="left",
            validate="1:1",
        )
        .join(
            away_features,
            on="match_key",
            how="left",
            validate="1:1",
        )
        .drop(
            "home_team_feature_id",
            "away_team_feature_id",
        )
    )

    zero_fill_columns = [
        f"{prefix}_{column}"
        for prefix in ("home", "away")
        for column in TEAM_FEATURE_COLUMNS
        if column != "days_since_previous_match"
    ]

    result = result.with_columns(
        *[pl.col(column).fill_null(0) for column in zero_fill_columns]
    )

    return result.with_columns(
        (pl.col("home_points_last_5") - pl.col("away_points_last_5")).alias(
            "form_points_difference"
        ),
        (pl.col("home_goals_for_last_5") - pl.col("away_goals_for_last_5")).alias(
            "recent_attack_difference"
        ),
        (
            pl.col("home_goals_against_last_5") - pl.col("away_goals_against_last_5")
        ).alias("recent_defence_difference"),
        (
            pl.col("home_goal_difference_last_5")
            - pl.col("away_goal_difference_last_5")
        ).alias("recent_goal_difference_difference"),
        (
            pl.col("home_average_points_last_5") - pl.col("away_average_points_last_5")
        ).alias("average_form_difference"),
        (
            pl.col("home_days_since_previous_match")
            - pl.col("away_days_since_previous_match")
        ).alias("rest_days_difference"),
    ).sort(["kickoff_utc", "match_key"])


def write_form_feature_datasets(
    team_features: pl.DataFrame,
    match_features: pl.DataFrame,
    gold_directory: Path,
) -> None:
    """Write Gold form feature datasets and metadata."""
    gold_directory.mkdir(parents=True, exist_ok=True)

    team_path = gold_directory / "team_match_history.parquet"
    match_path = gold_directory / "match_form_features.parquet"
    report_path = gold_directory / "form_feature_report.json"

    team_features.write_parquet(
        team_path,
        compression="zstd",
        statistics=True,
    )
    match_features.write_parquet(
        match_path,
        compression="zstd",
        statistics=True,
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "team_history_rows": team_features.height,
        "match_feature_rows": match_features.height,
        "season_count": match_features["season"].n_unique(),
        "team_count": pl.concat(
            [
                match_features.select(pl.col("home_team_id").alias("team_id")),
                match_features.select(pl.col("away_team_id").alias("team_id")),
            ]
        )["team_id"].n_unique(),
        "first_match": str(match_features["match_date"].min()),
        "last_match": str(match_features["match_date"].max()),
        "rolling_window": 5,
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def build_and_write_form_features(
    settings: Settings,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build, validate and write form feature datasets."""
    matches = load_silver_matches(settings.silver_directory)

    team_features = build_team_form_features(matches)
    validate_team_match_history(team_features)

    match_features = build_match_form_features(
        matches,
        team_features,
    )
    validate_match_form_features(match_features)

    write_form_feature_datasets(
        team_features=team_features,
        match_features=match_features,
        gold_directory=settings.gold_directory,
    )

    return team_features, match_features


@app.command()
def run() -> None:
    """Build FootCast Gold rolling-form features."""
    settings = get_settings()

    team_features, match_features = build_and_write_form_features(settings)

    typer.echo(
        f"Built {team_features.height} team-match rows and "
        f"{match_features.height} match feature rows."
    )


if __name__ == "__main__":
    app()
