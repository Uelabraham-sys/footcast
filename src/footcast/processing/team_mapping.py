"""Canonical football-team identity mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from footcast.config import PROJECT_ROOT


class UnknownTeamError(ValueError):
    """Raised when a source team cannot be mapped canonically."""


class TeamAliasConfigurationError(ValueError):
    """Raised when the alias configuration is invalid."""


@dataclass(frozen=True)
class TeamIdentity:
    """Canonical identity for one football team."""

    canonical_name: str
    team_slug: str


def normalise_alias(value: str) -> str:
    """Normalise an alias for case-insensitive matching."""
    return " ".join(value.strip().lower().split())


def create_team_slug(canonical_name: str) -> str:
    """Create a deterministic team identifier."""
    return (
        canonical_name.casefold()
        .replace("&", "and")
        .replace("'", "")
        .replace(".", "")
        .replace(" ", "-")
    )


def load_team_aliases(
    path: Path | None = None,
) -> dict[str, str]:
    """Load aliases as normalised alias-to-canonical mappings."""
    alias_path = path or PROJECT_ROOT / "config" / "team_aliases.yaml"

    if not alias_path.exists():
        raise TeamAliasConfigurationError(
            f"Team alias file does not exist: {alias_path}"
        )

    payload: Any = yaml.safe_load(alias_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TeamAliasConfigurationError("Team alias configuration must be a mapping.")

    alias_mapping: dict[str, str] = {}

    for canonical_name, aliases in payload.items():
        if not isinstance(canonical_name, str):
            raise TeamAliasConfigurationError("Canonical team names must be strings.")

        if not isinstance(aliases, list):
            raise TeamAliasConfigurationError(
                f"Aliases for {canonical_name} must be a list."
            )

        all_aliases = [canonical_name, *aliases]

        for alias in all_aliases:
            if not isinstance(alias, str):
                raise TeamAliasConfigurationError(
                    f"Invalid alias configured for {canonical_name}."
                )

            normalised = normalise_alias(alias)
            existing = alias_mapping.get(normalised)

            if existing is not None and existing != canonical_name:
                raise TeamAliasConfigurationError(
                    f"Alias {alias!r} maps to both {existing!r} and {canonical_name!r}."
                )

            alias_mapping[normalised] = canonical_name

    return alias_mapping


def canonicalise_team_name(
    team_name: str,
    alias_mapping: dict[str, str],
) -> str:
    """Convert a source team name to its canonical identity."""
    normalised = normalise_alias(team_name)
    canonical_name = alias_mapping.get(normalised)

    if canonical_name is None:
        raise UnknownTeamError(f"No canonical identity exists for team {team_name!r}.")

    return canonical_name


def find_unknown_teams(
    dataframe: pl.DataFrame,
    alias_mapping: dict[str, str],
    columns: tuple[str, ...] = ("home_team", "away_team"),
) -> list[str]:
    """Return source team names not represented in the alias mapping."""
    source_names: set[str] = set()

    for column in columns:
        if column not in dataframe.columns:
            continue

        source_names.update(
            value
            for value in dataframe[column].drop_nulls().to_list()
            if isinstance(value, str)
        )

    return sorted(
        team_name
        for team_name in source_names
        if normalise_alias(team_name) not in alias_mapping
    )


def canonicalise_match_teams(
    dataframe: pl.DataFrame,
    alias_mapping: dict[str, str],
) -> pl.DataFrame:
    """Canonicalise home and away teams in a match dataframe."""
    unknown_teams = find_unknown_teams(dataframe, alias_mapping)

    if unknown_teams:
        raise UnknownTeamError(
            "Unknown team aliases found: "
            + ", ".join(repr(team) for team in unknown_teams)
        )

    alias_frame = pl.DataFrame(
        {
            "normalised_alias": list(alias_mapping.keys()),
            "canonical_team": list(alias_mapping.values()),
        }
    )

    result = dataframe.with_columns(
        pl.col("home_team")
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .alias("_home_alias"),
        pl.col("away_team")
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .alias("_away_alias"),
    )

    result = result.join(
        alias_frame.rename(
            {
                "normalised_alias": "_home_alias",
                "canonical_team": "_canonical_home",
            }
        ),
        on="_home_alias",
        how="left",
        validate="m:1",
    )

    result = result.join(
        alias_frame.rename(
            {
                "normalised_alias": "_away_alias",
                "canonical_team": "_canonical_away",
            }
        ),
        on="_away_alias",
        how="left",
        validate="m:1",
    )

    return (
        result.drop("home_team", "away_team")
        .rename(
            {
                "_canonical_home": "home_team",
                "_canonical_away": "away_team",
            }
        )
        .with_columns(
            pl.col("home_team")
            .map_elements(
                create_team_slug,
                return_dtype=pl.String,
            )
            .alias("home_team_id"),
            pl.col("away_team")
            .map_elements(
                create_team_slug,
                return_dtype=pl.String,
            )
            .alias("away_team_id"),
        )
        .drop("_home_alias", "_away_alias")
    )


def build_team_dimension(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Build a unique team dimension from canonical matches."""
    home_teams = dataframe.select(
        pl.col("home_team_id").alias("team_id"),
        pl.col("home_team").alias("team_name"),
    )
    away_teams = dataframe.select(
        pl.col("away_team_id").alias("team_id"),
        pl.col("away_team").alias("team_name"),
    )

    return pl.concat([home_teams, away_teams]).unique().sort("team_name")
