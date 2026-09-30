"""Refresh 2026/27 La Liga results from football-data.co.uk.

This is the Phase 3 fallback pipeline: it updates the already-loaded fixture
calendar in place, never inserts matches, and rebuilds a standings snapshot
after every successful run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
import yaml
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import (
    DataFreshness,
    Match,
    MatchStatus,
    Season,
    StandingsSnapshot,
    Team,
    TeamAlias,
)
from app.standings import rank_standings

SEASON_ID = 1734
RESULTS_URL = "https://www.football-data.co.uk/mmz4281/2627/SP1.csv"
ALIAS_SOURCE = "football_data_co_uk"
FRESHNESS_SOURCE = "results"
ALIAS_FILE = (
    Path(__file__).resolve().parents[2]
    / "db"
    / "aliases"
    / "football_data_co_uk.yaml"
)

CLOSING_ODDS_COLUMNS = (
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("BWCH", "BWCD", "BWCA"),
    ("IWCH", "IWCD", "IWCA"),
    ("WHCH", "WHCD", "WHCA"),
    ("VCCH", "VCCD", "VCCA"),
)


class UnknownTeamError(ValueError):
    """Raised when a football-data.co.uk name has no approved alias."""


class ScoreChangeError(RuntimeError):
    """Raised when a refresh would rewrite a completed match score."""


@dataclass(frozen=True)
class ResultRow:
    """One completed football-data.co.uk match."""

    row_number: int
    home_name: str
    away_name: str
    home_goals: int
    away_goals: int
    home_goals_ht: int | None
    away_goals_ht: int | None
    home_shots: int | None
    away_shots: int | None
    home_shots_on_tgt: int | None
    away_shots_on_tgt: int | None
    home_corners: int | None
    away_corners: int | None
    home_fouls: int | None
    away_fouls: int | None
    home_yellows: int | None
    away_yellows: int | None
    home_reds: int | None
    away_reds: int | None
    referee: str | None
    closing_p_home: Decimal | None
    closing_p_draw: Decimal | None
    closing_p_away: Decimal | None


@dataclass(frozen=True)
class RefreshReport:
    played_rows: int
    changed_rows: int
    snapshot_rows: int
    before_md5: str
    after_md5: str


def _optional_int(value: str | None, *, column: str, row_number: int) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(
            f"Invalid {column} on football-data.co.uk row {row_number}: {value!r}"
        ) from exc


def _positive_decimal(value: str | None) -> Decimal | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return None
    return parsed if parsed > 0 else None


def _closing_probabilities(
    values: Mapping[str, str | None],
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    for home_key, draw_key, away_key in CLOSING_ODDS_COLUMNS:
        odds = (
            _positive_decimal(values.get(home_key)),
            _positive_decimal(values.get(draw_key)),
            _positive_decimal(values.get(away_key)),
        )
        if any(odd is None for odd in odds):
            continue
        home_odds, draw_odds, away_odds = odds
        assert home_odds is not None
        assert draw_odds is not None
        assert away_odds is not None
        inverse = (Decimal(1) / home_odds, Decimal(1) / draw_odds, Decimal(1) / away_odds)
        overround = sum(inverse)
        quantum = Decimal("0.0001")
        home_probability, draw_probability, away_probability = (
            (probability / overround).quantize(quantum) for probability in inverse
        )
        return home_probability, draw_probability, away_probability
    return None, None, None


def parse_played_rows(
    rows: Iterable[Mapping[str, str | None]],
) -> list[ResultRow]:
    """Parse completed rows; future fixtures are intentionally ignored."""
    parsed: list[ResultRow] = []
    for row_number, values in enumerate(rows, start=2):
        home_text = (values.get("FTHG") or "").strip()
        away_text = (values.get("FTAG") or "").strip()
        if not home_text and not away_text:
            continue
        if not home_text or not away_text:
            raise ValueError(f"Partial final score on football-data.co.uk row {row_number}")
        home_name = (values.get("HomeTeam") or "").strip()
        away_name = (values.get("AwayTeam") or "").strip()
        if not home_name or not away_name:
            raise ValueError(f"Missing team on football-data.co.uk row {row_number}")
        probabilities = _closing_probabilities(values)
        home_goals = _optional_int(values.get("FTHG"), column="FTHG", row_number=row_number)
        away_goals = _optional_int(values.get("FTAG"), column="FTAG", row_number=row_number)
        assert home_goals is not None
        assert away_goals is not None
        referee = (values.get("Referee") or "").strip() or None
        parsed.append(
            ResultRow(
                row_number=row_number,
                home_name=home_name,
                away_name=away_name,
                home_goals=home_goals,
                away_goals=away_goals,
                home_goals_ht=_optional_int(
                    values.get("HTHG"), column="HTHG", row_number=row_number
                ),
                away_goals_ht=_optional_int(
                    values.get("HTAG"), column="HTAG", row_number=row_number
                ),
                home_shots=_optional_int(values.get("HS"), column="HS", row_number=row_number),
                away_shots=_optional_int(values.get("AS"), column="AS", row_number=row_number),
                home_shots_on_tgt=_optional_int(
                    values.get("HST"), column="HST", row_number=row_number
                ),
                away_shots_on_tgt=_optional_int(
                    values.get("AST"), column="AST", row_number=row_number
                ),
                home_corners=_optional_int(values.get("HC"), column="HC", row_number=row_number),
                away_corners=_optional_int(values.get("AC"), column="AC", row_number=row_number),
                home_fouls=_optional_int(values.get("HF"), column="HF", row_number=row_number),
                away_fouls=_optional_int(values.get("AF"), column="AF", row_number=row_number),
                home_yellows=_optional_int(values.get("HY"), column="HY", row_number=row_number),
                away_yellows=_optional_int(values.get("AY"), column="AY", row_number=row_number),
                home_reds=_optional_int(values.get("HR"), column="HR", row_number=row_number),
                away_reds=_optional_int(values.get("AR"), column="AR", row_number=row_number),
                referee=referee,
                closing_p_home=probabilities[0],
                closing_p_draw=probabilities[1],
                closing_p_away=probabilities[2],
            )
        )
    return parsed


def download_rows(url: str = RESULTS_URL) -> list[dict[str, str]]:
    """Download the current-season CSV using the campus-network TLS setting."""
    response = httpx.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
        verify=False,
        follow_redirects=True,
    )
    response.raise_for_status()
    content = response.content.decode("utf-8-sig", errors="replace")
    return [dict(row) for row in csv.DictReader(io.StringIO(content))]


def load_alias_ids(db: Session) -> dict[str, int]:
    """Load approved aliases from the table and checked-in alias registry."""
    alias_ids = {
        raw_name: team_id
        for raw_name, team_id in db.execute(
            select(TeamAlias.raw_name, TeamAlias.team_id).where(
                or_(
                    TeamAlias.source == ALIAS_SOURCE,
                    TeamAlias.source == "football-data.co.uk",
                )
            )
        ).all()
    }
    with ALIAS_FILE.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    registry = document.get(ALIAS_SOURCE, {}) if isinstance(document, dict) else {}
    if not isinstance(registry, dict):
        raise ValueError(f"Alias YAML has no valid {ALIAS_SOURCE!r} mapping")
    team_ids = {
        canonical_name: team_id
        for team_id, canonical_name in db.execute(select(Team.id, Team.canonical_name)).all()
    }
    for raw_name, canonical_name in registry.items():
        if not isinstance(raw_name, str) or not isinstance(canonical_name, str):
            raise ValueError("Alias YAML entries must map strings to strings")
        team_id = team_ids.get(canonical_name)
        if team_id is not None:
            alias_ids.setdefault(raw_name, team_id)
    return alias_ids


def _result_values(row: ResultRow, *, include_score: bool) -> dict[str, Any]:
    values: dict[str, Any] = {
        "home_shots": row.home_shots,
        "away_shots": row.away_shots,
        "home_shots_on_tgt": row.home_shots_on_tgt,
        "away_shots_on_tgt": row.away_shots_on_tgt,
        "home_corners": row.home_corners,
        "away_corners": row.away_corners,
        "home_fouls": row.home_fouls,
        "away_fouls": row.away_fouls,
        "home_yellows": row.home_yellows,
        "away_yellows": row.away_yellows,
        "home_reds": row.home_reds,
        "away_reds": row.away_reds,
        "referee": row.referee,
        "closing_p_home": row.closing_p_home,
        "closing_p_draw": row.closing_p_draw,
        "closing_p_away": row.closing_p_away,
    }
    if include_score:
        values.update(
            status=MatchStatus.FT,
            home_goals=row.home_goals,
            away_goals=row.away_goals,
            home_goals_ht=row.home_goals_ht,
            away_goals_ht=row.away_goals_ht,
        )
    return values


def apply_result_rows(
    matches: Sequence[Match],
    result_rows: Sequence[ResultRow],
    alias_ids: Mapping[str, int],
    *,
    updated_at: datetime,
    allow_score_changes: bool = False,
) -> int:
    """Match played rows to existing fixtures and mutate only changed rows."""
    by_pair = {(match.home_team_id, match.away_team_id): match for match in matches}
    seen_pairs: set[tuple[int, int]] = set()
    resolved: list[tuple[Match, ResultRow]] = []
    for row in result_rows:
        try:
            home_id = alias_ids[row.home_name]
        except KeyError as exc:
            raise UnknownTeamError(
                f"No approved football-data.co.uk alias for {row.home_name!r}"
            ) from exc
        try:
            away_id = alias_ids[row.away_name]
        except KeyError as exc:
            raise UnknownTeamError(
                f"No approved football-data.co.uk alias for {row.away_name!r}"
            ) from exc
        pair = (home_id, away_id)
        if pair in seen_pairs:
            raise ValueError(
                f"Duplicate played fixture in CSV: {row.home_name} vs {row.away_name}"
            )
        seen_pairs.add(pair)
        match = by_pair.get(pair)
        if match is None:
            raise LookupError(
                f"No existing season {SEASON_ID} fixture for "
                f"{row.home_name} vs {row.away_name}"
            )
        resolved.append((match, row))

    score_changes: list[
        tuple[
            int,
            tuple[int | None, int | None, int | None, int | None],
            tuple[int | None, int | None, int | None, int | None],
        ]
    ] = []
    for match, row in resolved:
        if match.status != MatchStatus.FT:
            continue
        old_score = (
            match.home_goals,
            match.away_goals,
            match.home_goals_ht,
            match.away_goals_ht,
        )
        new_score = (
            row.home_goals,
            row.away_goals,
            row.home_goals_ht,
            row.away_goals_ht,
        )
        if old_score != new_score:
            score_changes.append((match.id, old_score, new_score))

    print(f"existing FT score changes: {len(score_changes)}")
    for match_id, old_score, new_score in score_changes:
        print(f"match id={match_id} old={old_score} new={new_score}")
    if score_changes and not allow_score_changes:
        raise ScoreChangeError(
            "Refusing to rewrite existing FT scores; rerun with "
            "--allow-score-changes to permit them"
        )

    changed = 0
    for match, row in resolved:
        include_score = match.status == MatchStatus.NS or allow_score_changes
        values = _result_values(row, include_score=include_score)
        if all(getattr(match, field) == value for field, value in values.items()):
            continue
        for field, value in values.items():
            setattr(match, field, value)
        match.updated_at = updated_at
        changed += 1
    return changed


MATCH_DIGEST_FIELDS = (
    "id",
    "season_id",
    "matchday",
    "kickoff_utc",
    "home_team_id",
    "away_team_id",
    "status",
    "home_goals",
    "away_goals",
    "home_goals_ht",
    "away_goals_ht",
    "home_shots",
    "away_shots",
    "home_shots_on_tgt",
    "away_shots_on_tgt",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellows",
    "away_yellows",
    "home_reds",
    "away_reds",
    "referee",
    "closing_p_home",
    "closing_p_draw",
    "closing_p_away",
    "updated_at",
)


def season_matches_md5(matches: Sequence[Match]) -> str:
    """Stable content fingerprint for the current season's match rows."""
    digest = hashlib.md5()
    for match in sorted(matches, key=lambda item: item.id):
        cells = []
        for field in MATCH_DIGEST_FIELDS:
            value = getattr(match, field)
            if isinstance(value, MatchStatus):
                value = value.value
            cells.append("" if value is None else str(value))
        digest.update("|".join(cells).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def status_counts(matches: Sequence[Match]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for match in matches:
        status = match.status.value if isinstance(match.status, MatchStatus) else str(match.status)
        counts[status] += 1
    return dict(sorted(counts.items()))


def rebuild_standings_snapshot(
    db: Session, matches: Sequence[Match], *, computed_at: datetime
) -> int:
    """Replace the season snapshot using the tested La Liga standings ranker."""
    ranked = rank_standings(matches, SEASON_ID)
    completed = sorted(
        (
            match
            for match in matches
            if match.home_goals is not None and match.away_goals is not None
        ),
        key=lambda match: (match.kickoff_utc, match.id),
    )
    forms: defaultdict[int, list[str]] = defaultdict(list)
    for match in completed:
        assert match.home_goals is not None
        assert match.away_goals is not None
        if match.home_goals > match.away_goals:
            home_form, away_form = "W", "L"
        elif match.home_goals < match.away_goals:
            home_form, away_form = "L", "W"
        else:
            home_form = away_form = "D"
        forms[match.home_team_id].append(home_form)
        forms[match.away_team_id].append(away_form)
    latest_matchday = max(
        (match.matchday for match in completed if match.matchday is not None),
        default=None,
    )
    db.query(StandingsSnapshot).filter(
        StandingsSnapshot.season_id == SEASON_ID
    ).delete(synchronize_session=False)
    for row in ranked:
        db.add(
            StandingsSnapshot(
                season_id=SEASON_ID,
                computed_at=computed_at,
                as_of_matchday=latest_matchday,
                position=row.position,
                team_id=int(row.team_id),
                played=row.played,
                won=row.won,
                drawn=row.drawn,
                lost=row.lost,
                goals_for=row.goals_for,
                goals_against=row.goals_against,
                points=row.points,
                form="".join(forms[row.team_id][-5:]),
            )
        )
    return len(ranked)


def _record_freshness(
    db: Session,
    *,
    attempted_at: datetime,
    success: bool,
    rows: int,
    error: str | None = None,
) -> None:
    freshness = db.get(DataFreshness, FRESHNESS_SOURCE)
    if freshness is None:
        freshness = DataFreshness(source=FRESHNESS_SOURCE)
        db.add(freshness)
    freshness.last_attempt_at = attempted_at
    freshness.rows_affected = rows
    if success:
        freshness.last_success_at = attempted_at
        freshness.last_error = None
    else:
        freshness.last_error = error


def refresh_results(
    db: Session,
    raw_rows: Iterable[Mapping[str, str | None]],
    *,
    allow_score_changes: bool = False,
) -> RefreshReport:
    """Apply one downloaded CSV and commit results, standings, and freshness."""
    attempted_at = datetime.now(timezone.utc)
    season = db.get(Season, SEASON_ID)
    if season is None:
        raise ValueError(f"Season {SEASON_ID} does not exist")
    matches = list(
        db.scalars(
            select(Match).where(Match.season_id == SEASON_ID).order_by(Match.id)
        )
    )
    if not matches:
        raise ValueError(f"Season {SEASON_ID} has no existing fixtures")
    before_md5 = season_matches_md5(matches)
    before_counts = status_counts(matches)
    played_rows = parse_played_rows(raw_rows)
    changed_rows = apply_result_rows(
        matches,
        played_rows,
        load_alias_ids(db),
        updated_at=attempted_at,
        allow_score_changes=allow_score_changes,
    )
    snapshot_rows = rebuild_standings_snapshot(db, matches, computed_at=attempted_at)
    _record_freshness(
        db,
        attempted_at=attempted_at,
        success=True,
        rows=changed_rows,
    )
    db.commit()
    after_md5 = season_matches_md5(matches)
    after_counts = status_counts(matches)
    print(f"season {SEASON_ID} status before: {before_counts}")
    print(f"season {SEASON_ID} matches before md5: {before_md5}")
    print(f"played CSV rows: {len(played_rows)}")
    print(f"changed match rows: {changed_rows}")
    print(f"standings snapshot rows: {snapshot_rows}")
    print(f"season {SEASON_ID} status after: {after_counts}")
    print(f"season {SEASON_ID} matches after md5: {after_md5}")
    return RefreshReport(
        played_rows=len(played_rows),
        changed_rows=changed_rows,
        snapshot_rows=snapshot_rows,
        before_md5=before_md5,
        after_md5=after_md5,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-score-changes",
        action="store_true",
        help="allow football-data.co.uk to rewrite scores already stored as FT",
    )
    args = parser.parse_args(argv)
    db = SessionLocal()
    try:
        rows = download_rows()
        refresh_results(db, rows, allow_score_changes=args.allow_score_changes)
    except Exception as exc:
        db.rollback()
        attempted_at = datetime.now(timezone.utc)
        _record_freshness(
            db,
            attempted_at=attempted_at,
            success=False,
            rows=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        db.commit()
        raise
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
