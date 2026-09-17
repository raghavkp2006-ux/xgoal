#!/usr/bin/env python3
"""Phase 1.3 — Match ingestion via API-Football (v3)."""
import argparse, os, sys, warnings
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx, yaml
from app.database import SessionLocal
from app.models import Competition, Match, Season, Team


def api_key():
    """The API-Football key. There is deliberately no default value.

    ``settings`` reads ``API_FOOTBALL_KEY`` from the environment and
    ``backend/.env``; the ``os.getenv`` fallback keeps the job usable where the
    variable is exported directly. A missing key is a hard error rather than a
    silent fallback to a credential in the source tree.

    The import lives here so this file does not gain a fourth ``sys.path``-order
    lint violation for a two-line helper.
    """
    from app.config import settings

    key = settings.api_football_key or os.getenv("API_FOOTBALL_KEY")
    if not key:
        raise SystemExit(
            "API_FOOTBALL_KEY is not set: put it in backend/.env "
            "(see .env.example) or export it before running this job"
        )
    return key


BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_MAP = {"SP1":140,"SP2":141,"E0":39,"E1":40,"E2":41,"E3":42,"D1":78,"D2":79,"I1":135,"I2":136,"F1":61,"F2":62,"N1":88,"P1":94,"T1":203,"B1":144,"G1":197,"SC0":179,"SC1":180}

def load_aliases():
    p = Path(__file__).resolve().parent.parent.parent / "db" / "aliases" / "football_data_co_uk.yaml"
    if not p.exists(): return {}
    d = yaml.safe_load(open(p, encoding="utf-8")) or {}
    return d.get("football_data_co_uk", {}) if isinstance(d, dict) else {}
def api_get(endpoint, params=None):
    # httpx is the client app/api_client.py already uses; follow_redirects keeps
    # the behaviour requests had by default (httpx does not follow them).
    r = httpx.get(
        f"{BASE_URL}/{endpoint}",
        headers={"x-apisports-key": api_key()},
        params=params,
        timeout=30,
        verify=False,
        follow_redirects=True,
    )
    return r.json()

def resolve_team(db, aliases, home_api, away_api):
    """Resolve API-Football team payloads to Team rows, caching the API ids.

    Matching order: api_football_id -> alias map -> canonical name.
    """
    hi, ai = home_api.get("id"), away_api.get("id")
    hn, an = home_api.get("name", ""), away_api.get("name", "")
    ht = db.query(Team).filter(Team.api_football_id == hi).first()
    at = db.query(Team).filter(Team.api_football_id == ai).first()
    if ht and at:
        return ht, at

    for side in ("home", "away"):
        sid = hi if side == "home" else ai
        sname = hn if side == "home" else an
        t = db.query(Team).filter(Team.api_football_id == sid).first()
        if t is None:
            c = aliases.get(sname)
            if c:
                t = db.query(Team).filter(Team.canonical_name == c).first()
            if not t:
                t = db.query(Team).filter(Team.canonical_name == sname).first()
            if t and not t.api_football_id:
                t.api_football_id = sid
                db.flush()
        if side == "home":
            ht = t
        else:
            at = t
    return ht, at


def map_status(s):
    if s in {"FT","AET","PEN","AWD","WO"}: return "FT"
    return {"NS":"NS","1H":"1H","HT":"HT","2H":"2H","ET":"ET","PST":"PST","CANC":"CANC","ABD":"ABD","SUSP":"SUSP","INT":"INT","TBD":"TBD"}.get(s,"NS")

def parse_kickoff(s):
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except: return datetime.now(timezone.utc)

def ingest_one(db, aliases, comp, season, league_id, yr):
    data = api_get("fixtures", {"league": league_id, "season": yr})
    fixtures = data.get("response", [])
    if not fixtures: return {"inserted":0,"skipped":0,"errors":[]}
    ins, skp, errs = 0, 0, []
    for f in fixtures:
        fix = f.get("fixture",{}); teams = f.get("teams",{}); goals = f.get("goals",{})
        score = f.get("score",{}).get("halftime",{})
        ht, at = resolve_team(db, aliases, teams.get("home",{}), teams.get("away",{}))
        if not ht or not at:
            errs.append(f"Unresolved: {teams.get('home',{}).get('name','?')} vs {teams.get('away',{}).get('name','?')}")
            skp += 1; continue
        if db.query(Match).filter(Match.season_id==season.id,Match.home_team_id==ht.id,Match.away_team_id==at.id).first():
            skp += 1; continue
        try:
            db.add(Match(competition_id=comp.id,season_id=season.id,
                kickoff_utc=parse_kickoff(fix.get("date","")),
                home_team_id=ht.id,away_team_id=at.id,
                status=map_status(fix.get("status",{}).get("short","NS")),
                home_goals=goals.get("home"),away_goals=goals.get("away"),
                home_goals_ht=score.get("home"),away_goals_ht=score.get("away"),
                referee=fix.get("referee") or None,
                source="api-football",external_id=str(fix.get("id")),
                ingested_at=datetime.now(timezone.utc),updated_at=datetime.now(timezone.utc)))
            ins += 1
        except Exception as e:
            errs.append(str(e)); skp += 1
    db.commit()
    return {"inserted": ins, "skipped": skp, "errors": errs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--div", type=str); p.add_argument("--season", type=int)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    api_key()  # fail fast: no network or database work without a key
    divs = [args.div] if args.div else (list(LEAGUE_MAP.keys()) if args.all else ["SP1","SP2"])
    yrs = [args.season] if args.season else list(range(2015, 2025))
    aliases = load_aliases()
    print(f"Loaded {len(aliases)} aliases")
    st = api_get("status")
    rq = st.get("response",{}).get("requests",{})
    print(f"API quota: {rq.get('current',0)}/{rq.get('limit_day',100)}")
    db = SessionLocal()
    try:
        total_ins, total_skp = 0, 0
        for div in divs:
            lid = LEAGUE_MAP.get(div)
            if not lid: print(f"  {div}: no league ID"); continue
            comp = db.query(Competition).filter(Competition.code == div).first()
            if not comp: print(f"  {div}: no comp"); continue
            for yr in yrs:
                s = db.query(Season).filter(Season.competition_id==comp.id,Season.start_year==yr).first()
                if not s: print(f"  {div} {yr}: no season"); continue
                r = ingest_one(db, aliases, comp, s, lid, yr)
                total_ins += r["inserted"]; total_skp += r["skipped"]
                e = f", {len(r['errors'])} errs" if r["errors"] else ""
                print(f"  {div} {yr}: +{r['inserted']} ins, {r['skipped']} skp{e}")
        print(f"\n=== Summary: {total_ins} inserted, {total_skp} skipped ===")
    finally: db.close()

if __name__ == "__main__":
    main()