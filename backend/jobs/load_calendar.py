"""Load the manually curated La Liga 2026/27 fixture calendar."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import yaml
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql.dml import Insert
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Competition, Match, MatchStatus, Season, Team

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "laliga_2026_27_calendar.csv"
ALIAS_FILE = PROJECT_ROOT / "db" / "aliases" / "football_data_co_uk.yaml"
SEASON_ID = 1734
COMPETITION_CODE = "SP1"
SOURCE = "laliga_official"


class UnknownTeamError(ValueError):
    """Raised when an official calendar name lacks an exact alias or team."""


@dataclass(frozen=True)
class CalendarRow:
    row_number: int
    matchday: int
    kickoff_utc: datetime
    home: str
    away: str
    time_tbc: bool


@dataclass(frozen=True)
class LoadReport:
    inserted: int
    skipped_existing: int
    skipped_conflict: int
    mismatches: int
    time_tbc: int


def load_aliases(path: Path = ALIAS_FILE) -> dict[str, str]:
    """Read exact raw-name to canonical-name mappings for this source."""
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Alias YAML must contain a mapping: {path}")
    aliases = document.get(SOURCE)
    if not isinstance(aliases, dict) or not all(
        isinstance(raw, str) and isinstance(canonical, str)
        for raw, canonical in aliases.items()
    ):
        raise ValueError(f"Alias YAML has no valid {SOURCE!r} mapping: {path}")
    return aliases


def resolve_team_id(
    raw_name: str,
    aliases: dict[str, str],
    team_ids: dict[str, int],
) -> int:
    """Resolve by exact alias and canonical name; never guess or fuzzy-match."""
    canonical = aliases.get(raw_name)
    if canonical is None:
        raise UnknownTeamError(f"No {SOURCE} alias for team {raw_name!r}")
    team_id = team_ids.get(canonical)
    if team_id is None:
        raise UnknownTeamError(
            f"{SOURCE} alias for {raw_name!r} points to missing team {canonical!r}"
        )
    return team_id


def _parse_kickoff(value: str, row_number: int) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid kickoff_utc on CSV row {row_number}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"kickoff_utc must include a timezone on CSV row {row_number}")
    return parsed.astimezone(timezone.utc)


def read_calendar(path: Path = DEFAULT_CSV) -> list[CalendarRow]:
    """Parse and validate the official calendar CSV."""
    required = {"matchday", "kickoff_utc", "home", "away", "time_tbc"}
    rows: list[CalendarRow] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must include columns: {', '.join(sorted(required))}")
        for row_number, values in enumerate(reader, start=2):
            try:
                matchday = int((values["matchday"] or "").strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid matchday on CSV row {row_number}") from exc
            home = (values["home"] or "").strip()
            away = (values["away"] or "").strip()
            if not home or not away:
                raise ValueError(f"Missing team name on CSV row {row_number}")
            tbc_text = (values["time_tbc"] or "").strip().lower()
            if tbc_text not in {"true", "false"}:
                raise ValueError(f"Invalid time_tbc on CSV row {row_number}: {tbc_text!r}")
            rows.append(
                CalendarRow(
                    row_number=row_number,
                    matchday=matchday,
                    kickoff_utc=_parse_kickoff(values["kickoff_utc"] or "", row_number),
                    home=home,
                    away=away,
                    time_tbc=tbc_text == "true",
                )
            )
    return rows


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _display(row: CalendarRow) -> str:
    return (
        f"MD{row.matchday:02d} {row.kickoff_utc.isoformat()} "
        f"{row.home} vs {row.away}"
    )


def _insert_statement(dialect_name: str, values: dict[str, object]) -> Insert:
    insert: Insert
    if dialect_name == "postgresql":
        insert = pg_insert(Match)
    elif dialect_name == "sqlite":
        insert = sqlite_insert(Match)
    else:
        raise RuntimeError(f"Unsupported database dialect for safe inserts: {dialect_name}")
    return (
        insert.values(**values)
        .on_conflict_do_nothing(
            index_elements=[Match.season_id, Match.home_team_id, Match.away_team_id]
        )
        .returning(Match.id)
    )


def load_calendar(
    db: Session,
    csv_path: Path = DEFAULT_CSV,
    aliases: dict[str, str] | None = None,
    dry_run: bool = False,
    output: TextIO = sys.stdout,
) -> LoadReport:
    """Plan and optionally insert missing fixtures without updating any row."""
    source_aliases = load_aliases() if aliases is None else aliases
    rows = read_calendar(csv_path)

    season = db.get(Season, SEASON_ID)
    if season is None:
        raise ValueError(f"Season {SEASON_ID} does not exist")
    competition = db.get(Competition, season.competition_id)
    if competition is None or competition.code != COMPETITION_CODE:
        raise ValueError(
            f"Season {SEASON_ID} must belong to {COMPETITION_CODE}; "
            f"found {None if competition is None else competition.code}"
        )

    team_ids = {
        canonical_name: team_id
        for team_id, canonical_name in db.execute(
            select(Team.id, Team.canonical_name)
        ).all()
    }
    resolved: list[tuple[CalendarRow, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for row in rows:
        home_id = resolve_team_id(row.home, source_aliases, team_ids)
        away_id = resolve_team_id(row.away, source_aliases, team_ids)
        pair = (home_id, away_id)
        if home_id == away_id:
            raise ValueError(f"Self-fixture in CSV row {row.row_number}: {_display(row)}")
        if pair in seen_pairs:
            raise ValueError(f"Duplicate ordered team pair in CSV: {_display(row)}")
        seen_pairs.add(pair)
        resolved.append((row, home_id, away_id))

    existing = {
        (match.home_team_id, match.away_team_id): match
        for match in db.scalars(select(Match).where(Match.season_id == SEASON_ID))
    }
    inserts: list[tuple[CalendarRow, int, int]] = []
    skipped_existing = 0
    mismatches = 0
    time_tbc = sum(row.time_tbc for row in rows)

    for row, home_id, away_id in resolved:
        match = existing.get((home_id, away_id))
        if match is None:
            inserts.append((row, home_id, away_id))
            if dry_run:
                print(f"DRY-RUN WOULD INSERT {_display(row)}", file=output)
            continue

        skipped_existing += 1
        date_matches = _utc(match.kickoff_utc).date() == row.kickoff_utc.date()
        if not date_matches:
            mismatches += 1
        status = match.status.value if isinstance(match.status, MatchStatus) else match.status
        detail = "date matches" if date_matches else (
            f"DATE MISMATCH csv={row.kickoff_utc.date()} "
            f"db={_utc(match.kickoff_utc).date()}"
        )
        print(f"SKIP existing status={status} {detail} {_display(row)}", file=output)

    inserted = 0
    skipped_conflict = 0
    if not dry_run:
        now = datetime.now(timezone.utc)
        try:
            for row, home_id, away_id in inserts:
                values: dict[str, object] = {
                    "competition_id": competition.id,
                    "season_id": SEASON_ID,
                    "matchday": row.matchday,
                    "kickoff_utc": row.kickoff_utc,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "status": MatchStatus.NS,
                    "home_goals": None,
                    "away_goals": None,
                    "source": SOURCE,
                    "external_id": None,
                    "ingested_at": now,
                    "updated_at": now,
                }
                match_id = db.execute(
                    _insert_statement(db.get_bind().dialect.name, values)
                ).scalar_one_or_none()
                if match_id is None:
                    skipped_conflict += 1
                    print(f"SKIP concurrent insert {_display(row)}", file=output)
                else:
                    inserted += 1
                    print(f"INSERTED id={match_id} {_display(row)}", file=output)
            db.commit()
        except Exception:
            db.rollback()
            raise

    print(f"time_tbc rows: {time_tbc}", file=output)
    insert_label = "would insert" if dry_run else "inserted"
    insert_count = len(inserts) if dry_run else inserted
    print(f"{insert_label}: {insert_count}", file=output)
    print(f"skipped existing: {skipped_existing}", file=output)
    print(f"skipped concurrent conflicts: {skipped_conflict}", file=output)
    print(f"kickoff date mismatches: {mismatches}", file=output)
    return LoadReport(
        inserted=0 if dry_run else inserted,
        skipped_existing=skipped_existing,
        skipped_conflict=skipped_conflict,
        mismatches=mismatches,
        time_tbc=time_tbc,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Load the La Liga 2026/27 calendar.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="calendar CSV path")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    with SessionLocal() as db:
        load_calendar(db, csv_path=args.csv, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
