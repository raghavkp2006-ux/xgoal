#!/usr/bin/env python3
"""Direct CSV ingestion from football-data.co.uk.

Downloads every (division, season) CSV once (parallel), caches the raw bytes
under backend/data/csv_cache/, then inserts the fixtures with resolved teams.

Notes
-----
* Idempotent: a fixture already present for a season is skipped, so the job can
  be re-run at any time to fill gaps.
* `matches` carries UniqueConstraint(season_id, home_team_id, away_team_id) plus
  a no_self_play CHECK, so for split-format leagues (SC0/SC1) only the FIRST
  meeting of a repeated home/away pairing can be stored. Those leagues
  therefore land with ~50-58% of their raw rows.

Usage:
    python -m jobs.ingest_direct --div SP1
    python -m jobs.ingest_direct --all
"""
import csv, io, os, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore")
# Windows consoles default to cp1252; force UTF-8 so accented team names never crash
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx; import yaml
from app.database import SessionLocal
from app.models import Competition, Match, Season, Team

PROJ = Path(__file__).resolve().parent.parent
ALIAS_FILE = PROJ.parent / "db" / "aliases" / "football_data_co_uk.yaml"
CACHE_DIR = PROJ / "data" / "csv_cache"
DIVS = ["SP1","SP2","E0","E1","E2","E3","D1","D2","I1","I2","F1","F2","N1","P1","T1","B1","G1","SC0","SC1"]

def season_codes(n=10):
    return [f"{(25-1-i)%100:02d}{(25-i)%100:02d}" for i in range(n)]

def load_aliases():
    d = yaml.safe_load(open(ALIAS_FILE, encoding="utf-8")) or {}
    return d.get("football_data_co_uk", {}) if isinstance(d, dict) else {}

def decode_bytes(raw):
    """football-data.co.uk serves UTF-8 for recent files, cp1252 for older ones."""
    if raw is None:
        return None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def download_csv(sc, div, retries=3):
    """Return raw CSV bytes (or None). Bytes are cached so decoding can change later."""
    url = f"https://www.football-data.co.uk/mmz4281/{sc}/{div}.csv"
    for attempt in range(retries):
        try:
            # httpx does not follow redirects by default; requests did.
            r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"},
                          timeout=20, verify=False, follow_redirects=True)
            if r.status_code == 200 and len(r.content) > 200:
                return r.content
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def fetch_cached(sc, div):
    """Download with a local disk cache (raw bytes) so re-runs never hit the network."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / f"{sc}_{div}.csv"
    if cf.exists() and cf.stat().st_size > 200:
        return decode_bytes(cf.read_bytes())
    raw = download_csv(sc, div)
    if raw:
        try:
            cf.write_bytes(raw)
        except Exception:
            pass
    return decode_bytes(raw)


def parse_csv(content):
    if not content: return []
    if content.startswith("\ufeff"): content = content[1:]
    lines = content.strip().split("\n")
    while lines and not lines[-1].strip(): lines.pop()
    return list(csv.DictReader(io.StringIO("\n".join(lines))))

def parse_date(val):
    if not val: return None
    for fmt in ("%d/%m/%Y","%d/%m/%y","%Y-%m-%d","%d/%m/%Y %H:%M"):
        try: return datetime.strptime(val.strip(), fmt).date()
        except: continue
    return None

def parse_time(val):
    if not val or not val.strip(): return None
    v = val.strip()
    if v.count(":") == 2: v = v.rsplit(":", 1)[0]
    try: datetime.strptime(v, "%H:%M"); return v
    except: return None

def parse_int(val):
    if val is None or (isinstance(val, str) and not val.strip()): return None
    try: return int(float(str(val).strip()))
    except: return None

def ingest_file(db, aliases, comp, season, rows, teams):
    ins = 0; skp = 0; errs = []
    has_time = "Time" in (rows[0] if rows else {})
    existing = set(db.query(Match.home_team_id, Match.away_team_id)
                   .filter(Match.season_id == season.id).all())
    batch = []
    for row in rows:
        if not row.get("Div") and not row.get("Date") and not row.get("HomeTeam"): continue
        rh = (row.get("HomeTeam") or "").strip(); ra = (row.get("AwayTeam") or "").strip()
        if not rh or not ra: skp += 1; continue
        ch = aliases.get(rh); ca = aliases.get(ra)
        if not ch or not ca: errs.append(f"No alias: {rh}/{ra}"); skp += 1; continue
        ht = teams.get(ch); at = teams.get(ca)
        if not ht or not at: errs.append(f"No team: {ch}/{ca}"); skp += 1; continue
        if ht.id == at.id:
            errs.append(f"SELF-PLAY: '{rh}' vs '{ra}' both -> '{ch}'")
            skp += 1
            continue
        if (ht.id, at.id) in existing: skp += 1; continue
        md = parse_date(row.get("Date"))
        if not md: skp += 1; continue
        mt = parse_time(row.get("Time")) if has_time else None
        hh, mm = 20, 0
        if mt:
            try: hh, mm = int(mt.split(":")[0]), int(mt.split(":")[1])
            except: pass
        ko = datetime(md.year, md.month, md.day, hh, mm, tzinfo=timezone.utc)
        fthg = parse_int(row.get("FTHG")); ftag = parse_int(row.get("FTAG"))
        st = "FT" if (fthg is not None and ftag is not None) else "NS"
        batch.append(Match(competition_id=comp.id, season_id=season.id, kickoff_utc=ko,
            home_team_id=ht.id, away_team_id=at.id, status=st,
            home_goals=fthg, away_goals=ftag, source="football-data.co.uk",
            ingested_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc)))
        existing.add((ht.id, at.id)); ins += 1
    if batch:
        db.add_all(batch); db.commit()
    return ins, skp, errs
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--div", type=str); p.add_argument("--season", type=str)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    codes = [args.season] if args.season else season_codes(10)
    divs = [args.div] if args.div else (DIVS if args.all else ["SP1","SP2"])
    aliases = load_aliases()
    print(f"Loaded {len(aliases)} aliases")
    db = SessionLocal()
    try:
        # --- Bulk-load competitions (1 query instead of 19) ---
        existing_comps = {c.code: c for c in db.query(Competition).all()}
        comps = {}
        new_comps = []
        for div in DIVS:
            c = existing_comps.get(div)
            if not c:
                c = Competition(code=div, name=div, country="", tier=1)
                new_comps.append(c)
            comps[div] = c
        if new_comps:
            db.add_all(new_comps)
            db.commit()
            for c in new_comps:
                comps[c.code] = c
        print(f"Competitions ready: {len(comps)}", flush=True)

        # --- Bulk-load seasons (1 query, then one bulk insert) ---
        wanted = {}   # (comp_id, start_year) -> Season
        for c in comps.values():
            for sc in codes:
                yr = int(sc[:2]) + 2000
                wanted[(c.id, yr)] = None
        for s in db.query(Season).all():
            key = (s.competition_id, s.start_year)
            if key in wanted:
                wanted[key] = s
        new_seasons = []
        for (cid, yr), s in wanted.items():
            if s is None:
                new_seasons.append(Season(
                    competition_id=cid, label=f"{yr}/{yr+1-2000}", start_year=yr,
                    start_date=date(yr, 8, 1), end_date=date(yr+1, 5, 31)))
        if new_seasons:
            db.add_all(new_seasons)
            db.commit()
            for s in new_seasons:
                wanted[(s.competition_id, s.start_year)] = s
        season_map = wanted
        print(f"Seasons ready: {len(season_map)}", flush=True)

        # Load all teams once (was per-file, 190 redundant queries)
        teams = {t.canonical_name: t for t in db.query(Team).all()}
        print(f"Teams loaded: {len(teams)}", flush=True)
        total_ins = 0; total_skp = 0

        # Parallel prefetch of every CSV (network-bound, so threads help a lot)
        tasks = [(div, sc) for div in divs for sc in codes]
        cache = {}
        print(f"Fetching {len(tasks)} CSV files (6 threads)...", flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(fetch_cached, sc, div): (div, sc) for div, sc in tasks}
            done = 0
            for fut in as_completed(futs):
                div, sc = futs[fut]
                done += 1
                try:
                    cache[(div, sc)] = fut.result()
                except Exception as e:
                    cache[(div, sc)] = None
                if done % 20 == 0 or done == len(tasks):
                    got = sum(1 for c in cache.values() if c)
                    print(f"  fetched {done}/{len(tasks)} ({got} ok)", flush=True)

        missing = [f"{d}/{s}" for (d, s), c in cache.items() if not c]
        if missing:
            print(f"  unavailable ({len(missing)}): {', '.join(missing[:15])}"
                  f"{' ...' if len(missing) > 15 else ''}", flush=True)

        for div in divs:
            comp = comps[div]
            for sc in codes:
                yr = int(sc[:2]) + 2000
                s = season_map.get((comp.id, yr))
                if not s: continue
                content = cache.get((div, sc))
                if not content:
                    continue
                rows = parse_csv(content)
                if not rows: continue
                try:
                    ins, skp, errs = ingest_file(db, aliases, comp, s, rows, teams)
                except Exception as e:
                    db.rollback()
                    print(f"  {div} {sc}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
                    continue
                total_ins += ins; total_skp += skp
                print(f"  {div} {sc}: +{ins} ins, {skp} skp, {len(errs)} errs", flush=True)
                for e in errs[:5]: print(f"    {e}", flush=True)
        print(f"\n=== Total: {total_ins} inserted, {total_skp} skipped ===", flush=True)
    finally: db.close()

if __name__ == "__main__": main()