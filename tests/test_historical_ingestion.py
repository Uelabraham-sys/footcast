"""Tests for historical Premier League ingestion."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from footcast.ingestion.historical import (
    calculate_sha256,
    historical_csv_url,
    normalise_historical_matches,
    save_bronze_parquet,
    season_code,
    season_label,
)
from footcast.ingestion.validation import (
    HistoricalDataValidationError,
    validate_historical_matches,
)


@pytest.fixture
def sample_source_data() -> pl.DataFrame:
    """Return a small source-format historical match dataset."""
    return pl.DataFrame(
        {
            "Div": ["E0", "E0", "E0"],
            "Date": ["09/08/2024", "10/08/2024", "11/08/2024"],
            "Time": ["20:00", "15:00", "16:30"],
            "HomeTeam": ["Arsenal", "Chelsea", "Liverpool"],
            "AwayTeam": ["Everton", "Fulham", "Manchester City"],
            "FTHG": [2, 1, 0],
            "FTAG": [0, 1, 3],
            "FTR": ["H", "D", "A"],
            "HTHG": [1, 0, 0],
            "HTAG": [0, 0, 1],
            "HTR": ["H", "D", "A"],
            "HS": [15, 11, 8],
            "AS": [6, 9, 17],
            "HST": [7, 4, 2],
            "AST": [2, 4, 9],
        }
    )


def test_season_code() -> None:
    """Season codes should contain the final two digits of each year."""
    assert season_code(2019) == "1920"
    assert season_code(2024) == "2425"
    assert season_code(2025) == "2526"


def test_season_label() -> None:
    """Season labels should use the conventional football format."""
    assert season_label(2019) == "2019/20"
    assert season_label(2024) == "2024/25"


def test_historical_csv_url() -> None:
    """The source URL should point to the Premier League CSV."""
    assert historical_csv_url(2024) == (
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    )


def test_sha256_is_deterministic() -> None:
    """Identical source content should produce identical checksums."""
    content = b"football-data"

    assert calculate_sha256(content) == calculate_sha256(content)
    assert len(calculate_sha256(content)) == 64


def test_normalisation(
    sample_source_data: pl.DataFrame,
) -> None:
    """Source columns should be converted to the FootCast schema."""
    fixed_time = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)

    result = normalise_historical_matches(
        source=sample_source_data,
        start_year=2024,
        ingested_at=fixed_time,
    )

    assert result.height == 3
    assert result["season"].unique().to_list() == ["2024/25"]
    assert result["competition"].unique().to_list() == ["Premier League"]
    assert result["source"].unique().to_list() == ["football-data.co.uk"]
    assert result["match_date"].dtype == pl.Date
    assert result["home_goals"].dtype == pl.Int64
    assert result["match_id"].n_unique() == 3


def test_normalised_data_passes_validation(
    sample_source_data: pl.DataFrame,
) -> None:
    """A valid source dataset should pass all validation rules."""
    result = normalise_historical_matches(
        source=sample_source_data,
        start_year=2024,
    )

    validate_historical_matches(result)


def test_incorrect_result_fails_validation(
    sample_source_data: pl.DataFrame,
) -> None:
    """A result label inconsistent with the score should fail."""
    result = normalise_historical_matches(
        source=sample_source_data,
        start_year=2024,
    ).with_columns(
        pl.when(pl.col("home_team") == "Arsenal")
        .then(pl.lit("A"))
        .otherwise(pl.col("full_time_result"))
        .alias("full_time_result")
    )

    with pytest.raises(
        HistoricalDataValidationError,
        match="does not agree with the score",
    ):
        validate_historical_matches(result)


def test_duplicate_match_id_fails_validation(
    sample_source_data: pl.DataFrame,
) -> None:
    """Duplicated match identifiers should fail validation."""
    result = normalise_historical_matches(
        source=sample_source_data,
        start_year=2024,
    )

    duplicated = pl.concat([result, result.head(1)])

    with pytest.raises(
        HistoricalDataValidationError,
        match="duplicated match identifiers",
    ):
        validate_historical_matches(duplicated)


def test_parquet_is_written(
    sample_source_data: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """A season should be written to partitioned Bronze storage."""
    result = normalise_historical_matches(
        source=sample_source_data,
        start_year=2024,
    )

    output_path = save_bronze_parquet(
        dataframe=result,
        start_year=2024,
        bronze_directory=tmp_path,
    )

    assert output_path.exists()
    assert output_path.name == "matches.parquet"
    assert "season=2425" in str(output_path)

    restored = pl.read_parquet(output_path)

    assert restored.height == 3
    assert restored.columns == result.columns
