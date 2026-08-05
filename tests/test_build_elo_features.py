"""Tests for match-level Elo feature construction."""

from datetime import UTC, datetime

import polars as pl

from footcast.features.build_elo_features import (
    build_match_elo_features,
    validate_elo_history,
)
from footcast.features.elo import build_elo_history


def create_matches() -> pl.DataFrame:
    """Create representative Silver match records."""
    kickoff = datetime(
        2024,
        8,
        1,
        15,
        0,
        tzinfo=UTC,
    )

    return pl.DataFrame(
        {
            "match_key": ["m1"],
            "season": ["2024/25"],
            "competition": ["Premier League"],
            "kickoff_utc": [kickoff],
            "match_date": [kickoff.date()],
            "status": ["FINISHED"],
            "home_team_id": ["arsenal"],
            "home_team": ["Arsenal"],
            "away_team_id": ["chelsea"],
            "away_team": ["Chelsea"],
            "home_goals": [2],
            "away_goals": [1],
            "full_time_result": ["H"],
            "home_points": [3],
            "away_points": [0],
        }
    )


def test_match_elo_features_have_one_row_per_match() -> None:
    """Every Silver match should receive one Elo feature row."""
    matches = create_matches()
    history = build_elo_history(matches)

    result = build_match_elo_features(
        matches=matches,
        elo_history=history,
    )

    assert result.height == 1
    assert result["match_key"].n_unique() == 1
    assert result["home_elo_pre"].item() == 1500.0
    assert result["away_elo_pre"].item() == 1500.0


def test_elo_history_passes_validation() -> None:
    """Valid Elo history should satisfy invariants."""
    history = build_elo_history(create_matches())

    validate_elo_history(history)
