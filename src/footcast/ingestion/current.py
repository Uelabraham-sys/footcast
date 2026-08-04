"""Current Premier League fixture, result and standings ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import httpx
import polars as pl
import typer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from footcast.config import Settings, get_settings
from footcast.ingestion.current_validation import (
    validate_current_matches,
    validate_current_standings,
)
from footcast.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)

SOURCE_NAME: Final[str] = "football-data.org"
FINISHED_STATUS: Final[str] = "FINISHED"

app = typer.Typer(help="Ingest current Premier League fixtures, results and standings.")


class FootballDataAPIError(RuntimeError):
    """Raised when the football-data.org API request fails."""


@dataclass(frozen=True)
class CurrentIngestionManifest:
    """Metadata for one current-data ingestion run."""

    competition_code: str
    generated_at: str
    matches_raw_path: str
    standings_raw_path: str
    matches_parquet_path: str
    standings_parquet_path: str
    match_count: int
    standing_count: int
    matches_sha256: str
    standings_sha256: str


class FootballDataClient:
    """Small HTTP client for football-data.org v4."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialise the authenticated API client."""
        if not api_key.strip():
            raise ValueError(
                "FOOTBALL_DATA_API_KEY is missing. Add it to the local .env file."
            )

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "X-Auth-Token": api_key,
                "Accept": "application/json",
                "User-Agent": "FootCast/0.1",
            },
            timeout=timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )

    def __enter__(self) -> FootballDataClient:
        """Enter the managed-client context."""
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Close the HTTP client when leaving its context."""
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    @retry(
        retry=retry_if_exception_type(
            (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _get(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform one authenticated GET request."""
        response = self._client.get(endpoint, params=params)

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            raise FootballDataAPIError(
                "football-data.org rate limit reached. Wait before retrying."
            )

        if response.status_code in {
            httpx.codes.UNAUTHORIZED,
            httpx.codes.FORBIDDEN,
        }:
            raise FootballDataAPIError("football-data.org rejected the API token.")

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise FootballDataAPIError(
                f"API request failed with status {response.status_code}: {endpoint}"
            ) from error

        payload = response.json()

        if not isinstance(payload, dict):
            raise FootballDataAPIError(f"Expected a JSON object from {endpoint}.")

        return payload

    def get_competition_matches(
        self,
        competition_code: str,
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Fetch competition fixtures and results."""
        params: dict[str, str] = {}

        if date_from is not None:
            params["dateFrom"] = date_from.isoformat()
        if date_to is not None:
            params["dateTo"] = date_to.isoformat()
        if status is not None:
            params["status"] = status

        return self._get(
            f"/competitions/{competition_code}/matches",
            params=params or None,
        )

    def get_competition_standings(
        self,
        competition_code: str,
    ) -> dict[str, Any]:
        """Fetch the current competition standings."""
        return self._get(f"/competitions/{competition_code}/standings")


def timestamp_slug(timestamp: datetime) -> str:
    """Convert a timestamp to a filesystem-safe UTC string."""
    return timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialise JSON deterministically for storage and checksums."""
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def calculate_payload_sha256(payload: dict[str, Any]) -> str:
    """Calculate the SHA-256 checksum of a JSON payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def save_raw_payload(
    payload: dict[str, Any],
    dataset_name: str,
    bronze_directory: Path,
    ingested_at: datetime,
) -> Path:
    """Preserve one untouched API response in Bronze storage."""
    output_directory = bronze_directory / "api" / dataset_name
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory / f"{dataset_name}_{timestamp_slug(ingested_at)}.json"
    )
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path


def season_from_payload(
    season_payload: dict[str, Any] | None,
) -> str | None:
    """Convert an API season start date to a football season label."""
    if not season_payload:
        return None

    start_date = season_payload.get("startDate")
    if not isinstance(start_date, str) or len(start_date) < 4:
        return None

    start_year = int(start_date[:4])
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def extract_score(
    score_payload: dict[str, Any] | None,
    side: str,
) -> int | None:
    """Extract a full-time score for one side."""
    if not score_payload:
        return None

    full_time = score_payload.get("fullTime")
    if not isinstance(full_time, dict):
        return None

    value = full_time.get(side)
    return value if isinstance(value, int) else None


def derive_result(
    home_goals: int | None,
    away_goals: int | None,
) -> str | None:
    """Derive H, D or A from a completed score."""
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def normalise_current_matches(
    payload: dict[str, Any],
    ingested_at: datetime | None = None,
) -> pl.DataFrame:
    """Normalise the competition match response."""
    timestamp = ingested_at or datetime.now(UTC)
    competition = payload.get("competition", {})
    matches = payload.get("matches", [])

    if not isinstance(matches, list):
        raise ValueError("The matches response has no valid matches list.")

    records: list[dict[str, Any]] = []

    for match in matches:
        if not isinstance(match, dict):
            continue

        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}
        score = match.get("score") or {}
        season = match.get("season") or {}

        home_goals = extract_score(score, "home")
        away_goals = extract_score(score, "away")
        api_match_id = match.get("id")

        records.append(
            {
                "match_id": (
                    f"football-data.org:{api_match_id}"
                    if api_match_id is not None
                    else None
                ),
                "api_match_id": api_match_id,
                "competition_id": competition.get("id"),
                "competition_code": competition.get("code"),
                "competition": competition.get("name"),
                "season": season_from_payload(season),
                "season_start": season.get("startDate"),
                "season_end": season.get("endDate"),
                "kickoff_utc": match.get("utcDate"),
                "status": match.get("status"),
                "matchday": match.get("matchday"),
                "stage": match.get("stage"),
                "group": match.get("group"),
                "home_team_id": home_team.get("id"),
                "home_team": home_team.get("name"),
                "home_team_short_name": home_team.get("shortName"),
                "home_team_tla": home_team.get("tla"),
                "away_team_id": away_team.get("id"),
                "away_team": away_team.get("name"),
                "away_team_short_name": away_team.get("shortName"),
                "away_team_tla": away_team.get("tla"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "full_time_result": derive_result(
                    home_goals,
                    away_goals,
                ),
                "winner": score.get("winner"),
                "duration": score.get("duration"),
                "last_updated": match.get("lastUpdated"),
                "source": SOURCE_NAME,
                "ingested_at": timestamp,
            }
        )

    if not records:
        return pl.DataFrame()

    dataframe = pl.DataFrame(records)

    return dataframe.with_columns(
        pl.col("api_match_id").cast(pl.Int64, strict=False),
        pl.col("competition_id").cast(pl.Int64, strict=False),
        pl.col("home_team_id").cast(pl.Int64, strict=False),
        pl.col("away_team_id").cast(pl.Int64, strict=False),
        pl.col("matchday").cast(pl.Int64, strict=False),
        pl.col("home_goals").cast(pl.Int64, strict=False),
        pl.col("away_goals").cast(pl.Int64, strict=False),
        pl.col("kickoff_utc")
        .cast(pl.String)
        .str.to_datetime(
            time_zone="UTC",
            strict=False,
        ),
        pl.col("last_updated")
        .cast(pl.String)
        .str.to_datetime(
            time_zone="UTC",
            strict=False,
        ),
    ).sort(["kickoff_utc", "api_match_id"])


def normalise_current_standings(
    payload: dict[str, Any],
    ingested_at: datetime | None = None,
) -> pl.DataFrame:
    """Normalise all standings tables from the API response."""
    timestamp = ingested_at or datetime.now(UTC)
    competition = payload.get("competition", {})
    season_payload = payload.get("season", {})
    standings = payload.get("standings", [])

    if not isinstance(standings, list):
        raise ValueError("The standings response has no valid standings list.")

    records: list[dict[str, Any]] = []

    for standing in standings:
        if not isinstance(standing, dict):
            continue

        standing_type = standing.get("type")
        standing_stage = standing.get("stage")
        standing_group = standing.get("group")
        table = standing.get("table", [])

        if not isinstance(table, list):
            continue

        for row in table:
            if not isinstance(row, dict):
                continue

            team = row.get("team") or {}

            records.append(
                {
                    "competition_id": competition.get("id"),
                    "competition_code": competition.get("code"),
                    "competition": competition.get("name"),
                    "season": season_from_payload(season_payload),
                    "standing_type": standing_type,
                    "stage": standing_stage,
                    "group": standing_group,
                    "position": row.get("position"),
                    "team_id": team.get("id"),
                    "team": team.get("name"),
                    "team_short_name": team.get("shortName"),
                    "team_tla": team.get("tla"),
                    "played_games": row.get("playedGames"),
                    "form": row.get("form"),
                    "won": row.get("won"),
                    "draw": row.get("draw"),
                    "lost": row.get("lost"),
                    "points": row.get("points"),
                    "goals_for": row.get("goalsFor"),
                    "goals_against": row.get("goalsAgainst"),
                    "goal_difference": row.get("goalDifference"),
                    "source": SOURCE_NAME,
                    "ingested_at": timestamp,
                }
            )

    if not records:
        return pl.DataFrame()

    integer_columns = (
        "competition_id",
        "position",
        "team_id",
        "played_games",
        "won",
        "draw",
        "lost",
        "points",
        "goals_for",
        "goals_against",
        "goal_difference",
    )

    return (
        pl.DataFrame(records)
        .with_columns(
            *[pl.col(column).cast(pl.Int64, strict=False) for column in integer_columns]
        )
        .sort(["standing_type", "position"])
    )


def save_current_parquet(
    dataframe: pl.DataFrame,
    dataset_name: str,
    bronze_directory: Path,
) -> Path:
    """Write the latest normalised API dataset as Parquet."""
    output_directory = bronze_directory / "current"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / f"{dataset_name}.parquet"
    dataframe.write_parquet(
        output_path,
        compression="zstd",
        statistics=True,
    )
    return output_path


def write_current_manifest(
    manifest: CurrentIngestionManifest,
    bronze_directory: Path,
) -> Path:
    """Write metadata for the latest current-data ingestion."""
    manifest_directory = bronze_directory / "manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)

    output_path = manifest_directory / "current_ingestion.json"
    output_path.write_text(
        json.dumps(asdict(manifest), indent=2),
        encoding="utf-8",
    )
    return output_path


def ingest_current_data(
    settings: Settings,
) -> CurrentIngestionManifest:
    """Fetch, validate and store current matches and standings."""
    ingested_at = datetime.now(UTC)

    with FootballDataClient(
        api_key=settings.football_data_api_key,
        base_url=settings.football_data_base_url,
    ) as client:
        LOGGER.info(
            "Fetching %s matches.",
            settings.competition_code,
        )
        matches_payload = client.get_competition_matches(settings.competition_code)

        LOGGER.info(
            "Fetching %s standings.",
            settings.competition_code,
        )
        standings_payload = client.get_competition_standings(settings.competition_code)

    matches_raw_path = save_raw_payload(
        payload=matches_payload,
        dataset_name="matches",
        bronze_directory=settings.bronze_directory,
        ingested_at=ingested_at,
    )
    standings_raw_path = save_raw_payload(
        payload=standings_payload,
        dataset_name="standings",
        bronze_directory=settings.bronze_directory,
        ingested_at=ingested_at,
    )

    matches = normalise_current_matches(
        matches_payload,
        ingested_at=ingested_at,
    )
    standings = normalise_current_standings(
        standings_payload,
        ingested_at=ingested_at,
    )

    validate_current_matches(matches)
    validate_current_standings(standings)

    matches_parquet_path = save_current_parquet(
        dataframe=matches,
        dataset_name="matches",
        bronze_directory=settings.bronze_directory,
    )
    standings_parquet_path = save_current_parquet(
        dataframe=standings,
        dataset_name="standings",
        bronze_directory=settings.bronze_directory,
    )

    manifest = CurrentIngestionManifest(
        competition_code=settings.competition_code,
        generated_at=ingested_at.isoformat(),
        matches_raw_path=str(matches_raw_path),
        standings_raw_path=str(standings_raw_path),
        matches_parquet_path=str(matches_parquet_path),
        standings_parquet_path=str(standings_parquet_path),
        match_count=matches.height,
        standing_count=standings.height,
        matches_sha256=calculate_payload_sha256(matches_payload),
        standings_sha256=calculate_payload_sha256(standings_payload),
    )

    write_current_manifest(
        manifest=manifest,
        bronze_directory=settings.bronze_directory,
    )

    return manifest


@app.command()
def run() -> None:
    """Ingest current Premier League data."""
    settings = get_settings()
    configure_logging(settings.footcast_log_level)

    manifest = ingest_current_data(settings)

    typer.echo(
        f"Ingested {manifest.match_count} matches and "
        f"{manifest.standing_count} standings rows."
    )


if __name__ == "__main__":
    app()
