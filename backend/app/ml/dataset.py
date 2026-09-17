"""Database -> MatchInput adapters shared by the Phase 2 jobs.

The models never touch SQLAlchemy; every read of the database lives here so the
jobs stay thin and the models stay unit-testable without a database.

The two Phase 2.2 readers at the bottom are the exception to "no leakage": they
are deliberately worded so the leakage test can shadow ``matches`` with a
temporary view filtered to ``kickoff_utc < as_of`` and prove the feature builder
never sees the future (see ``tests/test_ml_point_in_time.py``).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ml.data import MatchIdentity, MatchInput, PointInTimeMatch
from app.models import Competition, Match, Season, Team


def load_team_names(db: Session) -> dict[int, str]:
    """team_id -> canonical_name, i.e. the model's stable team key."""
    return {team.id: team.canonical_name for team in db.query(Team).all()}


def to_match_input(match: Match, names: dict[int, str]) -> MatchInput | None:
    """ORM row -> MatchInput; returns None when the fixture has no final score."""
    if match.home_goals is None or match.away_goals is None:
        return None
    return MatchInput(
        home=names[match.home_team_id],
        away=names[match.away_team_id],
        kickoff=match.kickoff_utc,
        home_goals=match.home_goals,
        away_goals=match.away_goals,
        match_id=match.id,
        home_shots_on_tgt=match.home_shots_on_tgt,
        away_shots_on_tgt=match.away_shots_on_tgt,
        closing_p_home=float(match.closing_p_home) if match.closing_p_home is not None else None,
        closing_p_draw=float(match.closing_p_draw) if match.closing_p_draw is not None else None,
        closing_p_away=float(match.closing_p_away) if match.closing_p_away is not None else None,
    )


def load_completed_matches(
    db: Session,
    competition_id: int,
    names: dict[int, str],
    before: datetime | None = None,
) -> list[MatchInput]:
    """Finished fixtures of a competition, chronological, optionally cut at ``before``."""
    query = db.query(Match).filter(
        Match.competition_id == competition_id,
        Match.home_goals.isnot(None),
        Match.away_goals.isnot(None),
    )
    if before is not None:
        query = query.filter(Match.kickoff_utc < before)
    rows = query.order_by(Match.kickoff_utc).all()
    built = (to_match_input(row, names) for row in rows)
    return [match for match in built if match is not None]


def load_labeled_matches(
    db: Session, competition_id: int, names: dict[int, str]
) -> tuple[list[str], list[MatchInput]]:
    """(season labels, matches) — aligned, chronological, finished fixtures only."""
    rows = (
        db.query(Match, Season.label)
        .join(Season, Match.season_id == Season.id)
        .filter(
            Match.competition_id == competition_id,
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
        )
        .order_by(Match.kickoff_utc)
        .all()
    )
    labels: list[str] = []
    matches: list[MatchInput] = []
    for row, label in rows:
        built = to_match_input(row, names)
        if built is None:
            continue
        labels.append(label)
        matches.append(built)
    return labels, matches


def load_upcoming_matches(
    db: Session, competition_id: int, horizon: datetime
) -> list[Match]:
    """Unplayed fixtures kicking off at or before ``horizon``, earliest first."""
    return (
        db.query(Match)
        .filter(
            Match.competition_id == competition_id,
            Match.home_goals.is_(None),
            Match.kickoff_utc <= horizon,
        )
        .order_by(Match.kickoff_utc)
        .all()
    )


def load_recent_completed(db: Session, competition_id: int, count: int) -> list[Match]:
    """The ``count`` most recent finished fixtures, returned oldest first."""
    rows = (
        db.query(Match)
        .filter(
            Match.competition_id == competition_id,
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
        )
        .order_by(Match.kickoff_utc.desc())
        .limit(count)
        .all()
    )
    return list(reversed(rows))


def get_competition(db: Session, code: str) -> Competition | None:
    """Competition by code (``SP1`` for La Liga)."""
    return db.query(Competition).filter(Competition.code == code).one_or_none()


def get_season(db: Session, competition_id: int, start_year: int) -> Season | None:
    """Season of a competition by its starting year."""
    return (
        db.query(Season)
        .filter(Season.competition_id == competition_id, Season.start_year == start_year)
        .one_or_none()
    )


# --------------------------------------------------------------------------- #
# Phase 2.2 — point-in-time readers
# --------------------------------------------------------------------------- #

MATCH_STREAM_SQL = text(
    """
    SELECT m.id, m.competition_id, c.tier AS competition_tier, m.season_id,
           s.start_year AS season_start_year, m.kickoff_utc,
           m.home_team_id, m.away_team_id, m.home_goals, m.away_goals,
           m.home_shots_on_tgt, m.away_shots_on_tgt
    FROM matches AS m
    JOIN competitions AS c ON c.id = m.competition_id
    JOIN seasons AS s ON s.id = m.season_id
    WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
    ORDER BY m.kickoff_utc, m.id
    """
)
"""Every completed fixture, chronological.

``matches`` is intentionally left unqualified: Postgres searches the temporary
schema first, so the leakage test can shadow this table with a view filtered to
``kickoff_utc < as_of``. The query carries **no** kick-off filter of its own — the
feature builder applies the ``as_of`` cut, which is exactly what the leakage test
measures.
"""

MATCH_IDENTITY_SQL = text(
    """
    SELECT m.id, m.competition_id, c.tier AS competition_tier, m.season_id,
           s.start_year AS season_start_year, m.matchday, m.kickoff_utc,
           m.home_team_id, m.away_team_id
    FROM public.matches AS m
    JOIN public.competitions AS c ON c.id = m.competition_id
    JOIN public.seasons AS s ON s.id = m.season_id
    WHERE m.id = :match_id
    """
)
"""The fixture being forecast, with no score columns.

Explicitly ``public.``-qualified on purpose: which teams play, where and when is
known at forecast time, so the fixture's identity must stay readable even when
the leakage test has hidden the future behind a restricted view.
"""

MATCH_STREAM_CACHE_KEY = "xgoal.match_stream"


def load_match_stream(db: Session, *, refresh: bool = False) -> list[PointInTimeMatch]:
    """The whole completed-match corpus as :class:`PointInTimeMatch` rows.

    Cached on the session (``db.info``): a job that prices forty fixtures should
    pay for one scan, not forty. Pass ``refresh=True`` after an ingestion run, or
    use a fresh session.
    """
    if not refresh:
        cached = db.info.get(MATCH_STREAM_CACHE_KEY)
        if isinstance(cached, list):
            return cached
    rows = db.execute(MATCH_STREAM_SQL).fetchall()
    stream = [
        PointInTimeMatch(
            match_id=int(row.id),
            competition_id=int(row.competition_id),
            competition_tier=int(row.competition_tier),
            season_id=int(row.season_id),
            season_start_year=int(row.season_start_year),
            kickoff_utc=row.kickoff_utc,
            home_team_id=int(row.home_team_id),
            away_team_id=int(row.away_team_id),
            home_goals=int(row.home_goals),
            away_goals=int(row.away_goals),
            home_shots_on_tgt=(
                None if row.home_shots_on_tgt is None else int(row.home_shots_on_tgt)
            ),
            away_shots_on_tgt=(
                None if row.away_shots_on_tgt is None else int(row.away_shots_on_tgt)
            ),
        )
        for row in rows
    ]
    db.info[MATCH_STREAM_CACHE_KEY] = stream
    return stream


def load_match_identity(db: Session, match_id: int) -> MatchIdentity:
    """The fixture being forecast: teams, venue-side, kick-off, season, division."""
    row = db.execute(MATCH_IDENTITY_SQL, {"match_id": match_id}).one_or_none()
    if row is None:
        raise LookupError(f"match {match_id} not found")
    return MatchIdentity(
        match_id=int(row.id),
        competition_id=int(row.competition_id),
        competition_tier=int(row.competition_tier),
        season_id=int(row.season_id),
        season_start_year=int(row.season_start_year),
        kickoff_utc=row.kickoff_utc,
        home_team_id=int(row.home_team_id),
        away_team_id=int(row.away_team_id),
        matchday=None if row.matchday is None else int(row.matchday),
    )

