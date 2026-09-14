"""Data ingestion service for pulling league data from API-Football."""
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.api_client import ApiFootballClient
from app.database import SessionLocal
from app.models import (
    Competition,
    DataFreshness,
    Match,
    Player,
    PlayerSeasonStat,
    Season,
    StandingsSnapshot,
    Team,
    TeamAlias,
)


def _upsert_freshness(db: Session, source: str, success: bool, error: Optional[str] = None, rows: int = 0):
    """Update or insert a data_freshness row."""
    row = db.query(DataFreshness).filter(DataFreshness.source == source).first()
    now = datetime.now(timezone.utc)
    if row:
        row.last_attempt_at = now
        if success:
            row.last_success_at = now
            row.last_error = None
            row.rows_affected = rows
        else:
            row.last_error = error
    else:
        row = DataFreshness(
            source=source,
            last_attempt_at=now,
            last_success_at=now if success else None,
            last_error=None if success else error,
            rows_affected=rows if success else None,
        )
        db.add(row)
    db.commit()


def seed_competition(db: Session, client: ApiFootballClient, api_id: int) -> Competition:
    """Fetch a league from API and create/update the competition + seasons."""
    data = client.get_leagues(api_id)
    league_info = data["response"][0]
    l = league_info["league"]

    # Upsert competition
    comp = db.query(Competition).filter(Competition.api_football_id == api_id).first()
    country = league_info.get("country", {})
    if isinstance(country, dict):
        country_name = country.get("name", "Unknown")
    else:
        country_name = str(country)

    if comp:
        comp.code = l["name"][:50].upper().replace(" ", "_")
        comp.name = l["name"]
        comp.country = country_name
    else:
        comp = Competition(
            code=l["name"][:50].upper().replace(" ", "_"),
            name=l["name"],
            country=country_name,
            tier=1 if l.get("type") == "League" else 2,
            api_football_id=api_id,
        )
        db.add(comp)
    db.flush()

    # Upsert seasons
    for s in league_info.get("seasons", []):
        year = s.get("year") or s.get("season")
        if not year:
            continue
        existing = (
            db.query(Season)
            .filter(Season.competition_id == comp.id, Season.start_year == year)
            .first()
        )
        if existing:
            existing.is_current = s.get("current", False)
            if s.get("start"):
                try:
                    existing.start_date = date.fromisoformat(s["start"])
                except (ValueError, TypeError):
                    pass
            if s.get("end"):
                try:
                    existing.end_date = date.fromisoformat(s["end"])
                except (ValueError, TypeError):
                    pass
        else:
            label = f"{year}/{year + 1 - 2000}"
            try:
                sd = date.fromisoformat(s.get("start", f"{year}-01-01"))
            except (ValueError, TypeError):
                sd = date(year, 1, 1)
            try:
                ed = date.fromisoformat(s.get("end", f"{year}-12-31"))
            except (ValueError, TypeError):
                ed = date(year, 12, 31)

            season = Season(
                competition_id=comp.id,
                label=label,
                start_year=year,
                start_date=sd,
                end_date=ed,
                is_current=s.get("current", False),
            )
            db.add(season)

    db.commit()
    _upsert_freshness(db, f"competition_{api_id}", success=True, rows=1)
    return comp


def ingest_teams(
    db: Session,
    client: ApiFootballClient,
    competition_id: int,
    season_year: int,
    api_league_id: int,
) -> list[Team]:
    """Fetch teams for a league/season and upsert into DB with aliases."""
    data = client.get_teams(api_league_id, season_year)
    teams_created = []

    for item in data.get("response", []):
        t = item["team"]
        v = item.get("venue", {}) or {}

        team = db.query(Team).filter(Team.api_football_id == t["id"]).first()
        if team:
            team.venue_name = v.get("name") or team.venue_name
            team.logo_url = t.get("logo") or team.logo_url
        else:
            team = Team(
                canonical_name=t["name"],
                short_name=t.get("code", ""),
                api_football_id=t["id"],
                founded=t.get("founded"),
                venue_name=v.get("name"),
                logo_url=t.get("logo"),
                created_at=datetime.now(timezone.utc),
            )
            db.add(team)
            db.flush()
            teams_created.append(team)

        # Add alias from API source
        existing_alias = (
            db.query(TeamAlias)
            .filter(TeamAlias.source == "api-football", TeamAlias.raw_name == t["name"])
            .first()
        )
        if not existing_alias:
            alias = TeamAlias(
                source="api-football",
                raw_name=t["name"],
                team_id=team.id,
            )
            db.add(alias)

    db.commit()
    _upsert_freshness(
        db,
        f"teams_{api_league_id}_{season_year}",
        success=True,
        rows=len(teams_created),
    )
    return teams_created


def ingest_fixtures(
    db: Session,
    client: ApiFootballClient,
    competition_id: int,
    season_id: int,
    api_league_id: int,
    season_year: int,
) -> list[Match]:
    """Fetch fixtures for a league/season and upsert into DB."""
    data = client.get_fixtures(api_league_id, season_year)
    matches = []

    for item in data.get("response", []):
        f = item["fixture"]
        goals = item.get("goals", {}) or {}
        score = item.get("score", {}) or {}
        ht = (score.get("halftime") or {}) if isinstance(score, dict) else {}
        stats = item.get("statistics", [])
        league = item.get("league", {})

        teams_item = item.get("teams", {}) or {}
        home_team = db.query(Team).filter(Team.api_football_id == teams_item["home"]["id"]).first()
        away_team = db.query(Team).filter(Team.api_football_id == teams_item["away"]["id"]).first()
        if not home_team or not away_team:
            continue

        matchday = league.get("round", "")
        if isinstance(matchday, str):
            try:
                matchday = int(matchday.split()[-1])
            except (ValueError, IndexError):
                matchday = None

        kickoff = f.get("date") or f.get("timestamp")
        if isinstance(kickoff, str):
            try:
                kickoff = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
            except ValueError:
                kickoff = datetime.now(timezone.utc)
        elif isinstance(kickoff, (int, float)):
            kickoff = datetime.fromtimestamp(kickoff, tz=timezone.utc)
        else:
            kickoff = datetime.now(timezone.utc)

        status_info = f.get("status", {}) or {}
        status = status_info.get("short", "NS")
        minute = status_info.get("elapsed")

        existing = (
            db.query(Match)
            .filter(
                Match.season_id == season_id,
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
            )
            .first()
        )
        if existing:
            existing.status = status
            existing.minute = minute
            existing.home_goals = goals.get("home")
            existing.away_goals = goals.get("away")
            existing.home_goals_ht = ht.get("home")
            existing.away_goals_ht = ht.get("away")
            existing.updated_at = datetime.now(timezone.utc)
            matches.append(existing)
        else:
            match = Match(
                competition_id=competition_id,
                season_id=season_id,
                matchday=matchday,
                kickoff_utc=kickoff,
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                status=status,
                minute=minute,
                home_goals=goals.get("home"),
                away_goals=goals.get("away"),
                home_goals_ht=ht.get("home"),
                away_goals_ht=ht.get("away"),
                source="api-football",
                external_id=str(f["id"]),
                ingested_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(match)
            db.flush()
            matches.append(match)

    db.commit()
    _upsert_freshness(
        db,
        f"fixtures_{api_league_id}_{season_year}",
        success=True,
        rows=len(matches),
    )
    return matches


def ingest_standings(
    db: Session,
    client: ApiFootballClient,
    season_id: int,
    api_league_id: int,
    season_year: int,
) -> list[StandingsSnapshot]:
    """Fetch standings and store a snapshot."""
    data = client.get_standings(api_league_id, season_year)
    snapshots = []

    for item in data.get("response", []):
        league = item.get("league", {})
        for standing_list in league.get("standings", []):
            for entry in standing_list:
                team = db.query(Team).filter(Team.api_football_id == entry["team"]["id"]).first()
                if not team:
                    continue

                snap = StandingsSnapshot(
                    season_id=season_id,
                    computed_at=datetime.now(timezone.utc),
                    as_of_matchday=entry.get("all", {}).get("played"),
                    position=entry.get("rank", 0),
                    team_id=team.id,
                    played=entry.get("all", {}).get("played"),
                    won=entry.get("all", {}).get("win"),
                    drawn=entry.get("all", {}).get("draw"),
                    lost=entry.get("all", {}).get("lose"),
                    goals_for=entry.get("all", {}).get("goals", {}).get("for"),
                    goals_against=entry.get("all", {}).get("goals", {}).get("against"),
                    points=entry.get("points"),
                    form=entry.get("form"),
                )
                db.add(snap)
                db.flush()
                snapshots.append(snap)

    db.commit()
    _upsert_freshness(
        db,
        f"standings_{api_league_id}_{season_year}",
        success=True,
        rows=len(snapshots),
    )
    return snapshots


def ingest_players(
    db: Session,
    client: ApiFootballClient,
    season_id: int,
    api_league_id: int,
    season_year: int,
) -> list[Player]:
    """Fetch players for a league/season and upsert."""
    data = client.get_players(api_league_id, season_year)
    players = []

    for item in data.get("response", []):
        p = item.get("player", {})
        stats_list = item.get("statistics", [])

        player = db.query(Player).filter(Player.api_football_id == p.get("id")).first()
        if not player:
            birth = p.get("birth", {}) or {}
            try:
                bd = date.fromisoformat(birth.get("date", "")) if birth.get("date") else None
            except (ValueError, TypeError):
                bd = None
            player = Player(
                api_football_id=p.get("id"),
                name=p.get("name", "Unknown"),
                nationality=p.get("nationality"),
                birth_date=bd,
                position=p.get("position"),
            )
            db.add(player)
            db.flush()
            players.append(player)

        for s in stats_list:
            team_info = s.get("team", {}) or {}
            team = db.query(Team).filter(Team.api_football_id == team_info.get("id")).first()
            if not team:
                continue
            games = s.get("games", {}) or {}
            goals_s = s.get("goals", {}) or {}
            shots_s = s.get("shots", {}) or {}
            passes_s = s.get("passes", {}) or {}
            cards = s.get("cards", {}) or {}

            existing_stat = (
                db.query(PlayerSeasonStat)
                .filter(
                    PlayerSeasonStat.player_id == player.id,
                    PlayerSeasonStat.team_id == team.id,
                    PlayerSeasonStat.season_id == season_id,
                )
                .first()
            )
            if existing_stat:
                existing_stat.appearances = games.get("appearences")
                existing_stat.lineups = games.get("lineups")
                existing_stat.minutes = games.get("minutes")
                existing_stat.goals = goals_s.get("total")
                existing_stat.assists = goals_s.get("assists")
                existing_stat.shots = shots_s.get("total")
                existing_stat.shots_on_target = shots_s.get("on")
                existing_stat.passes = passes_s.get("total")
                existing_stat.yellows = cards.get("yellow")
                existing_stat.reds = cards.get("red")
                existing_stat.rating = games.get("rating")
            else:
                stat = PlayerSeasonStat(
                    player_id=player.id,
                    team_id=team.id,
                    season_id=season_id,
                    snapshot_date=date.today(),
                    appearances=games.get("appearences"),
                    lineups=games.get("lineups"),
                    minutes=games.get("minutes"),
                    goals=goals_s.get("total"),
                    assists=goals_s.get("assists"),
                    shots=shots_s.get("total"),
                    shots_on_target=shots_s.get("on"),
                    passes=passes_s.get("total"),
                    yellows=cards.get("yellow"),
                    reds=cards.get("red"),
                    rating=games.get("rating"),
                )
                db.add(stat)

    db.commit()
    _upsert_freshness(
        db,
        f"players_{api_league_id}_{season_year}",
        success=True,
        rows=len(players),
    )
    return players


def full_ingest(api_league_id: int = 140, season_year: int | None = None) -> None:
    """Full pipeline: competition -> teams -> fixtures -> standings -> players."""
    from datetime import datetime

    client = ApiFootballClient()
    db = SessionLocal()

    try:
        # 1. Competition + seasons
        comp = seed_competition(db, client, api_league_id)

        # Determine available season
        seasons = (
            db.query(Season)
            .filter(Season.competition_id == comp.id)
            .order_by(Season.start_year.desc())
            .all()
        )
        if season_year:
            season = next((s for s in seasons if s.start_year == season_year), None)
        else:
            season = next((s for s in seasons if s.is_current), seasons[0] if seasons else None)

        if not season:
            print("No season found to ingest")
            return

        print(f"Ingesting season: {season.label} (year {season.start_year})")

        # 2. Teams
        teams = ingest_teams(db, client, comp.id, season.start_year, api_league_id)
        print(f"  Teams: {len(teams)} created/updated")

        # 3. Fixtures
        fixtures = ingest_fixtures(db, client, comp.id, season.id, api_league_id, season.start_year)
        print(f"  Fixtures: {len(fixtures)}")

        # 4. Standings
        standings = ingest_standings(db, client, season.id, api_league_id, season.start_year)
        print(f"  Standings snapshots: {len(standings)}")

        # 5. Players
        players = ingest_players(db, client, season.id, api_league_id, season.start_year)
        print(f"  Players: {len(players)} new")

        print("Full ingest complete ✅")

    finally:
        client.close()
        db.close()