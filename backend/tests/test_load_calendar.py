from __future__ import annotations

from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.models import Competition, Match, MatchStatus, Season, Team
from jobs.load_calendar import UnknownTeamError, load_calendar, resolve_team_id


def _write_calendar(path: Path, kickoff: str = "2026-09-12T12:00:00Z") -> Path:
    path.write_text(
        "matchday,kickoff_utc,home,away,time_tbc\n"
        f"4,{kickoff},Home,Away,true\n",
        encoding="utf-8",
    )
    return path


def _session() -> tuple[Session, object]:
    engine = create_engine("sqlite://")
    for model in (Competition, Season, Team, Match):
        model.__table__.create(engine)
    db = Session(engine)
    competition = Competition(id=1, code="SP1", name="La Liga", country="Spain", tier=1)
    db.add(competition)
    db.add(
        Season(
            id=1734,
            competition_id=1,
            label="2026/27",
            start_year=2026,
            start_date=date(2026, 8, 1),
            end_date=date(2027, 5, 31),
            is_current=True,
        )
    )
    db.add_all(
        [
            Team(id=1, canonical_name="Home", created_at=datetime.now(timezone.utc)),
            Team(id=2, canonical_name="Away", created_at=datetime.now(timezone.utc)),
        ]
    )
    db.commit()
    return db, engine


def test_unmapped_team_raises_unknown_team_error() -> None:
    with pytest.raises(UnknownTeamError, match="No laliga_official alias"):
        resolve_team_id("Unmapped Club", {}, {"Home": 1})


def test_existing_ft_row_is_not_modified(tmp_path: Path) -> None:
    db, engine = _session()
    try:
        original_kickoff = datetime(2026, 9, 12, 16, 0, tzinfo=timezone.utc)
        original_updated = datetime(2026, 9, 12, 19, 0, tzinfo=timezone.utc)
        existing = Match(
            id=1,
            competition_id=1,
            season_id=1734,
            matchday=1,
            kickoff_utc=original_kickoff,
            home_team_id=1,
            away_team_id=2,
            status=MatchStatus.FT,
            home_goals=2,
            away_goals=1,
            source="football_data_co_uk",
            ingested_at=original_updated,
            updated_at=original_updated,
        )
        db.add(existing)
        db.commit()

        report = load_calendar(
            db,
            _write_calendar(tmp_path / "calendar.csv"),
            aliases={"Home": "Home", "Away": "Away"},
            output=StringIO(),
        )
        untouched = db.get(Match, 1)
        assert report.inserted == 0
        assert report.skipped_existing == 1
        assert untouched is not None
        assert untouched.status == MatchStatus.FT
        assert untouched.home_goals == 2
        assert untouched.away_goals == 1
        assert untouched.matchday == 1
        assert untouched.kickoff_utc.replace(tzinfo=timezone.utc) == original_kickoff
        assert untouched.updated_at.replace(tzinfo=timezone.utc) == original_updated
        assert untouched.source == "football_data_co_uk"
    finally:
        db.close()
        engine.dispose()


def test_successful_load_logs_insert_not_dry_run(tmp_path: Path) -> None:
    db, engine = _session()
    try:
        output = StringIO()
        report = load_calendar(
            db,
            _write_calendar(tmp_path / "calendar.csv"),
            aliases={"Home": "Home", "Away": "Away"},
            output=output,
        )
        assert report.inserted == 1
        assert "INSERTED id=" in output.getvalue()
        assert "WOULD INSERT" not in output.getvalue()
    finally:
        db.close()
        engine.dispose()


def test_repeated_load_is_idempotent(tmp_path: Path) -> None:
    db, engine = _session()
    try:
        calendar = _write_calendar(tmp_path / "calendar.csv")
        aliases = {"Home": "Home", "Away": "Away"}
        first = load_calendar(db, calendar, aliases=aliases, output=StringIO())
        first_rows = db.scalars(select(Match).where(Match.season_id == 1734)).all()
        first_snapshot = [
            (
                match.id,
                match.matchday,
                match.kickoff_utc,
                match.home_team_id,
                match.away_team_id,
                match.status,
                match.home_goals,
                match.away_goals,
                match.source,
                match.updated_at,
            )
            for match in first_rows
        ]

        second = load_calendar(db, calendar, aliases=aliases, output=StringIO())
        second_rows = db.scalars(select(Match).where(Match.season_id == 1734)).all()
        second_snapshot = [
            (
                match.id,
                match.matchday,
                match.kickoff_utc,
                match.home_team_id,
                match.away_team_id,
                match.status,
                match.home_goals,
                match.away_goals,
                match.source,
                match.updated_at,
            )
            for match in second_rows
        ]

        assert first.inserted == 1
        assert second.inserted == 0
        assert second.skipped_existing == 1
        assert db.scalar(
            select(func.count()).select_from(Match).where(Match.season_id == 1734)
        ) == 1
        assert second_snapshot == first_snapshot
        assert second_rows[0].status == MatchStatus.NS
        assert second_rows[0].home_goals is None
        assert second_rows[0].away_goals is None
    finally:
        db.close()
        engine.dispose()
