"""Tests for logistic-regression training orchestration."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import polars as pl

from footcast.modelling.train_logistic import (
    parse_c_values,
    train_and_evaluate_logistic,
)

FEATURE_COLUMNS = (
    "feature_one",
    "feature_two",
)


def create_model_dataset() -> pl.DataFrame:
    """Create a three-split synthetic modelling dataset."""
    start = datetime(
        2022,
        8,
        1,
        tzinfo=UTC,
    )

    rows: list[dict[str, object]] = []
    match_index = 0

    split_definitions = (
        ("train", "2022/23", 15),
        ("validation", "2023/24", 9),
        ("test", "2024/25", 9),
    )

    for split, season, repetitions in split_definitions:
        for repeat in range(repetitions):
            target = repeat % 3

            feature_one = {
                0: -2.0,
                1: 0.0,
                2: 2.0,
            }[target]

            feature_two = {
                0: -1.0,
                1: 0.0,
                2: 1.0,
            }[target]

            kickoff = start + timedelta(days=match_index * 7)

            rows.append(
                {
                    "match_key": f"m{match_index}",
                    "season": season,
                    "kickoff_utc": kickoff,
                    "match_date": kickoff.date(),
                    "home_team_id": "home",
                    "home_team": "Home",
                    "away_team_id": "away",
                    "away_team": "Away",
                    "full_time_result": {
                        0: "A",
                        1: "D",
                        2: "H",
                    }[target],
                    "target": target,
                    "split": split,
                    "feature_one": (feature_one + repeat * 0.001),
                    "feature_two": (feature_two + repeat * 0.001),
                }
            )

            match_index += 1

    return pl.DataFrame(rows)


def test_parse_c_values() -> None:
    """Comma-separated C values should parse correctly."""
    assert parse_c_values("0.1,1,10") == (0.1, 1.0, 10.0)


def test_training_writes_all_artifacts(
    tmp_path: Path,
) -> None:
    """Training should persist model, reports and predictions."""
    dataset_path = tmp_path / "model_dataset.parquet"
    model_path = tmp_path / "models" / "logistic.joblib"
    report_directory = tmp_path / "reports"
    prediction_directory = tmp_path / "predictions"

    create_model_dataset().write_parquet(dataset_path)

    selection = train_and_evaluate_logistic(
        dataset_path=dataset_path,
        model_path=model_path,
        report_directory=report_directory,
        prediction_directory=prediction_directory,
        c_values=(0.1, 1.0),
        feature_columns=FEATURE_COLUMNS,
    )

    assert selection.height == 4
    assert model_path.exists()

    assert (report_directory / "logistic_selection.parquet").exists()
    assert (report_directory / "logistic_coefficients.parquet").exists()
    assert (report_directory / "logistic_regression.json").exists()

    assert (prediction_directory / "logistic_validation.parquet").exists()
    assert (prediction_directory / "logistic_test.parquet").exists()


def test_saved_model_has_required_metadata(
    tmp_path: Path,
) -> None:
    """Saved model bundle should include feature metadata."""
    dataset_path = tmp_path / "model_dataset.parquet"
    model_path = tmp_path / "logistic.joblib"

    create_model_dataset().write_parquet(dataset_path)

    train_and_evaluate_logistic(
        dataset_path=dataset_path,
        model_path=model_path,
        report_directory=tmp_path / "reports",
        prediction_directory=tmp_path / "predictions",
        c_values=(1.0,),
        feature_columns=FEATURE_COLUMNS,
    )

    bundle = joblib.load(model_path)

    assert bundle["feature_names"] == FEATURE_COLUMNS
    assert bundle["class_labels"] == (0, 1, 2)
    assert "model" in bundle
