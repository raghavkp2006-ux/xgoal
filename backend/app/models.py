"""SQLAlchemy ORM models matching the Part 4 schema from the build plan."""

import enum
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class MatchStatus(str, enum.Enum):
    NS = "NS"
    _1H = "1H"
    HT = "HT"
    _2H = "2H"
    ET = "ET"
    BT = "BT"
    P = "P"
    FT = "FT"
    AET = "AET"
    PEN = "PEN"
    PST = "PST"
    CANC = "CANC"
    ABD = "ABD"
    SUSP = "SUSP"
    INT = "INT"
    TBD = "TBD"
    AWD = "AWD"
    WO = "WO"


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    country: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    api_football_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True)


class Season(Base):
    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("competitions.id"), nullable=False
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    start_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)

    competition: Mapped["Competition"] = relationship()

    __table_args__ = (UniqueConstraint("competition_id", "start_year"),)


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    short_name: Mapped[Optional[str]] = mapped_column(Text)
    api_football_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True)
    founded: Mapped[Optional[int]] = mapped_column(SmallInteger)
    venue_name: Mapped[Optional[str]] = mapped_column(Text)
    logo_url: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class TeamAlias(Base):
    __tablename__ = "team_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    raw_name: Mapped[str] = mapped_column(Text, nullable=False)
    team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False
    )

    team: Mapped["Team"] = relationship()

    __table_args__ = (UniqueConstraint("source", "raw_name"),)


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("competitions.id"), nullable=False
    )
    season_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("seasons.id"), nullable=False
    )
    matchday: Mapped[Optional[int]] = mapped_column(SmallInteger)
    kickoff_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    home_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False
    )
    away_team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False
    )
    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus, name="match_status"), default=MatchStatus.NS
    )
    minute: Mapped[Optional[int]] = mapped_column(SmallInteger)

    home_goals: Mapped[Optional[int]] = mapped_column(
        SmallInteger,
        CheckConstraint("home_goals BETWEEN 0 AND 20"),
    )
    away_goals: Mapped[Optional[int]] = mapped_column(
        SmallInteger,
        CheckConstraint("away_goals BETWEEN 0 AND 20"),
    )
    home_goals_ht: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_goals_ht: Mapped[Optional[int]] = mapped_column(SmallInteger)

    home_shots: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_shots: Mapped[Optional[int]] = mapped_column(SmallInteger)
    home_shots_on_tgt: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_shots_on_tgt: Mapped[Optional[int]] = mapped_column(SmallInteger)
    home_corners: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_corners: Mapped[Optional[int]] = mapped_column(SmallInteger)
    home_fouls: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_fouls: Mapped[Optional[int]] = mapped_column(SmallInteger)
    home_yellows: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_yellows: Mapped[Optional[int]] = mapped_column(SmallInteger)
    home_reds: Mapped[Optional[int]] = mapped_column(SmallInteger)
    away_reds: Mapped[Optional[int]] = mapped_column(SmallInteger)
    referee: Mapped[Optional[str]] = mapped_column(Text)

    closing_p_home: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    closing_p_draw: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    closing_p_away: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))

    source: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[Optional[str]] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    competition: Mapped["Competition"] = relationship()
    season: Mapped["Season"] = relationship()
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])

    __table_args__ = (
        CheckConstraint("home_team_id <> away_team_id", name="no_self_play"),
        UniqueConstraint("season_id", "home_team_id", "away_team_id"),
        UniqueConstraint("source", "external_id"),
        Index("idx_matches_kickoff", "kickoff_utc"),
        Index("idx_matches_season_status", "season_id", "status"),
        Index("idx_matches_home_time", "home_team_id", "kickoff_utc"),
        Index("idx_matches_away_time", "away_team_id", "kickoff_utc"),
        Index("idx_matches_live", "status"),
    )


class MatchEvent(Base):
    __tablename__ = "match_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id", ondelete="CASCADE"), nullable=False
    )
    minute: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    extra_minute: Mapped[Optional[int]] = mapped_column(SmallInteger)
    team_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("teams.id"))
    player_id: Mapped[Optional[int]] = mapped_column(Integer)
    assist_id: Mapped[Optional[int]] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[Optional[str]] = mapped_column(Text)
    external_id: Mapped[Optional[str]] = mapped_column(Text)

    match: Mapped["Match"] = relationship()

    __table_args__ = (
        UniqueConstraint("match_id", "minute", "event_type", "player_id", "detail"),
    )


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_football_id: Mapped[Optional[int]] = mapped_column(Integer, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    nationality: Mapped[Optional[str]] = mapped_column(Text)
    birth_date: Mapped[Optional[date]] = mapped_column(Date)
    position: Mapped[Optional[str]] = mapped_column(Text)


class PlayerSeasonStat(Base):
    __tablename__ = "player_season_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id"), nullable=False
    )
    team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False
    )
    season_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("seasons.id"), nullable=False
    )
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    appearances: Mapped[Optional[int]] = mapped_column(SmallInteger)
    lineups: Mapped[Optional[int]] = mapped_column(SmallInteger)
    minutes: Mapped[Optional[int]] = mapped_column(Integer)
    goals: Mapped[Optional[int]] = mapped_column(SmallInteger)
    assists: Mapped[Optional[int]] = mapped_column(SmallInteger)
    shots: Mapped[Optional[int]] = mapped_column(SmallInteger)
    shots_on_target: Mapped[Optional[int]] = mapped_column(SmallInteger)
    passes: Mapped[Optional[int]] = mapped_column(Integer)
    pass_accuracy: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    yellows: Mapped[Optional[int]] = mapped_column(SmallInteger)
    reds: Mapped[Optional[int]] = mapped_column(SmallInteger)
    rating: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))

    player: Mapped["Player"] = relationship()
    team: Mapped["Team"] = relationship()
    season: Mapped["Season"] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "player_id", "team_id", "season_id", "snapshot_date",
            name="uq_player_season_snapshot"
        ),
        Index("idx_pss_lookup", "season_id", "team_id", "snapshot_date.desc()"),
        Index("idx_pss_goals", "season_id", "goals.desc()"),
    )


class StandingsSnapshot(Base):
    __tablename__ = "standings_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("seasons.id"), nullable=False
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    as_of_matchday: Mapped[Optional[int]] = mapped_column(SmallInteger)
    position: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    team_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("teams.id"), nullable=False
    )
    played: Mapped[Optional[int]] = mapped_column(SmallInteger)
    won: Mapped[Optional[int]] = mapped_column(SmallInteger)
    drawn: Mapped[Optional[int]] = mapped_column(SmallInteger)
    lost: Mapped[Optional[int]] = mapped_column(SmallInteger)
    goals_for: Mapped[Optional[int]] = mapped_column(SmallInteger)
    goals_against: Mapped[Optional[int]] = mapped_column(SmallInteger)
    points: Mapped[Optional[int]] = mapped_column(SmallInteger)
    form: Mapped[Optional[str]] = mapped_column(Text)

    season: Mapped["Season"] = relationship()
    team: Mapped["Team"] = relationship()

    __table_args__ = (
        UniqueConstraint("season_id", "computed_at", "team_id"),
        Index("idx_standings_latest", "season_id", "computed_at.desc()"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    git_sha: Mapped[str] = mapped_column(Text, nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    train_start: Mapped[date] = mapped_column(Date, nullable=False)
    train_end: Mapped[date] = mapped_column(Date, nullable=False)
    n_train_matches: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[dict] = mapped_column(JSONB, nullable=False)
    eval_metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("name", "version"),)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("matches.id"), nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=False
    )
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    as_of: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    p_home: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    p_draw: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    p_away: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    expected_home_goals: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    expected_away_goals: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2))
    score_matrix: Mapped[Optional[dict]] = mapped_column(JSONB)
    feature_hash: Mapped[str] = mapped_column(Text, nullable=False)

    match: Mapped["Match"] = relationship()
    model_version: Mapped["ModelVersion"] = relationship()

    __table_args__ = (
        CheckConstraint(
            "abs(p_home + p_draw + p_away - 1) < 0.001",
            name="probs_sum_to_one",
        ),
        UniqueConstraint("match_id", "model_version_id", "as_of"),
        Index("idx_predictions_match", "match_id"),
    )


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("seasons.id"), nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=False
    )
    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    n_simulations: Mapped[int] = mapped_column(Integer, nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of_matchday: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    forced_results: Mapped[Optional[dict]] = mapped_column(JSONB)
    results: Mapped[dict] = mapped_column(JSONB, nullable=False)

    season: Mapped["Season"] = relationship()
    model_version: Mapped["ModelVersion"] = relationship()

    __table_args__ = (
        Index("idx_simruns_latest", "season_id", "run_at.desc()"),
    )


class ApiRequestLog(Base):
    __tablename__ = "api_request_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    params: Mapped[Optional[dict]] = mapped_column(JSONB)
    status_code: Mapped[Optional[int]] = mapped_column(SmallInteger)
    quota_remaining: Mapped[Optional[int]] = mapped_column(Integer)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    __table_args__ = (Index("idx_api_log_day", "requested_at"),)


class DataFreshness(Base):
    __tablename__ = "data_freshness"

    source: Mapped[str] = mapped_column(Text, primary_key=True)
    last_success_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    last_attempt_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    rows_affected: Mapped[Optional[int]] = mapped_column(Integer)