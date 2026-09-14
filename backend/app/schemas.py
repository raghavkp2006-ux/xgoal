"""Pydantic request/response schemas for all entities."""
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Competitions ──────────────────────────────────────────────────────────

class CompetitionCreate(BaseModel):
    code: str = Field(..., min_length=1, max_length=50)
    name: str
    country: str
    tier: int = Field(..., ge=1, le=5)
    api_football_id: Optional[int] = None


class CompetitionResponse(BaseModel):
    id: int
    code: str
    name: str
    country: str
    tier: int
    api_football_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


# ── Seasons ───────────────────────────────────────────────────────────────

class SeasonCreate(BaseModel):
    competition_id: int
    label: str
    start_year: int
    start_date: date
    end_date: date
    is_current: bool = False


class SeasonResponse(BaseModel):
    id: int
    competition_id: int
    label: str
    start_year: int
    start_date: date
    end_date: date
    is_current: bool

    model_config = ConfigDict(from_attributes=True)


# ── Teams ─────────────────────────────────────────────────────────────────

class TeamCreate(BaseModel):
    canonical_name: str
    short_name: Optional[str] = None
    api_football_id: Optional[int] = None
    founded: Optional[int] = Field(None, ge=1800, le=2100)
    venue_name: Optional[str] = None
    logo_url: Optional[str] = None


class TeamResponse(BaseModel):
    id: int
    canonical_name: str
    short_name: Optional[str] = None
    api_football_id: Optional[int] = None
    founded: Optional[int] = None
    venue_name: Optional[str] = None
    logo_url: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Team Aliases ──────────────────────────────────────────────────────────

class TeamAliasCreate(BaseModel):
    source: str
    raw_name: str
    team_id: int


class TeamAliasResponse(BaseModel):
    id: int
    source: str
    raw_name: str
    team_id: int

    model_config = ConfigDict(from_attributes=True)


# ── Matches ───────────────────────────────────────────────────────────────

class MatchCreate(BaseModel):
    competition_id: int
    season_id: int
    matchday: Optional[int] = None
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    status: str = "NS"
    minute: Optional[int] = None
    home_goals: Optional[int] = Field(None, ge=0, le=20)
    away_goals: Optional[int] = Field(None, ge=0, le=20)
    home_goals_ht: Optional[int] = None
    away_goals_ht: Optional[int] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_tgt: Optional[int] = None
    away_shots_on_tgt: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellows: Optional[int] = None
    away_yellows: Optional[int] = None
    home_reds: Optional[int] = None
    away_reds: Optional[int] = None
    referee: Optional[str] = None
    closing_p_home: Optional[Decimal] = None
    closing_p_draw: Optional[Decimal] = None
    closing_p_away: Optional[Decimal] = None
    source: str
    external_id: Optional[str] = None


class MatchResponse(BaseModel):
    id: int
    competition_id: int
    season_id: int
    matchday: Optional[int] = None
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    status: str
    minute: Optional[int] = None
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    home_goals_ht: Optional[int] = None
    away_goals_ht: Optional[int] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_tgt: Optional[int] = None
    away_shots_on_tgt: Optional[int] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellows: Optional[int] = None
    away_yellows: Optional[int] = None
    home_reds: Optional[int] = None
    away_reds: Optional[int] = None
    referee: Optional[str] = None
    closing_p_home: Optional[Decimal] = None
    closing_p_draw: Optional[Decimal] = None
    closing_p_away: Optional[Decimal] = None
    source: str
    external_id: Optional[str] = None
    ingested_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Players ───────────────────────────────────────────────────────────────

class PlayerCreate(BaseModel):
    api_football_id: Optional[int] = None
    name: str
    nationality: Optional[str] = None
    birth_date: Optional[date] = None
    position: Optional[str] = None


class PlayerResponse(BaseModel):
    id: int
    api_football_id: Optional[int] = None
    name: str
    nationality: Optional[str] = None
    birth_date: Optional[date] = None
    position: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ── Player Season Stats ───────────────────────────────────────────────────

class PlayerSeasonStatCreate(BaseModel):
    player_id: int
    team_id: int
    season_id: int
    snapshot_date: date
    appearances: Optional[int] = None
    lineups: Optional[int] = None
    minutes: Optional[int] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    passes: Optional[int] = None
    pass_accuracy: Optional[Decimal] = None
    yellows: Optional[int] = None
    reds: Optional[int] = None
    rating: Optional[Decimal] = None


class PlayerSeasonStatResponse(BaseModel):
    id: int
    player_id: int
    team_id: int
    season_id: int
    snapshot_date: date
    appearances: Optional[int] = None
    lineups: Optional[int] = None
    minutes: Optional[int] = None
    goals: Optional[int] = None
    assists: Optional[int] = None
    shots: Optional[int] = None
    shots_on_target: Optional[int] = None
    passes: Optional[int] = None
    pass_accuracy: Optional[Decimal] = None
    yellows: Optional[int] = None
    reds: Optional[int] = None
    rating: Optional[Decimal] = None

    model_config = ConfigDict(from_attributes=True)


# ── Standings Snapshots ──────────────────────────────────────────────────

class StandingsSnapshotCreate(BaseModel):
    season_id: int
    computed_at: datetime
    as_of_matchday: Optional[int] = None
    position: int
    team_id: int
    played: Optional[int] = None
    won: Optional[int] = None
    drawn: Optional[int] = None
    lost: Optional[int] = None
    goals_for: Optional[int] = None
    goals_against: Optional[int] = None
    points: Optional[int] = None
    form: Optional[str] = None


class StandingsSnapshotResponse(BaseModel):
    id: int
    season_id: int
    computed_at: datetime
    as_of_matchday: Optional[int] = None
    position: int
    team_id: int
    played: Optional[int] = None
    won: Optional[int] = None
    drawn: Optional[int] = None
    lost: Optional[int] = None
    goals_for: Optional[int] = None
    goals_against: Optional[int] = None
    points: Optional[int] = None
    form: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ── API Request Log ──────────────────────────────────────────────────────

class ApiRequestLogResponse(BaseModel):
    id: int
    requested_at: datetime
    endpoint: str
    params: Optional[dict] = None
    status_code: Optional[int] = None
    quota_remaining: Optional[int] = None
    duration_ms: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)