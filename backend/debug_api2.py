"""Debug API responses for teams, fixtures, standings, players."""
import httpx, os, json
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("API_FOOTBALL_KEY")
headers = {"x-apisports-key": key}
base = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")

season = 2025
league_id = 140

print(f"=== Teams for {league_id}/{season} ===")
r = httpx.get(f"{base}/teams", headers=headers, params={"league": league_id, "season": season})
data = r.json()
print(f"  Status: {r.status_code}, Results: {data.get('results', 0)}")
if data.get("results", 0) > 0:
    print(f"  First team: {data['response'][0]['team']['name']}")

print(f"\n=== Fixtures for {league_id}/{season} ===")
r = httpx.get(f"{base}/fixtures", headers=headers, params={"league": league_id, "season": season})
data = r.json()
print(f"  Status: {r.status_code}, Results: {data.get('results', 0)}")

print(f"\n=== Standings for {league_id}/{season} ===")
r = httpx.get(f"{base}/standings", headers=headers, params={"league": league_id, "season": season})
data = r.json()
print(f"  Status: {r.status_code}, Results: {data.get('results', 0)}")

print(f"\n=== Players for {league_id}/{season} ===")
r = httpx.get(f"{base}/players", headers=headers, params={"league": league_id, "season": season})
data = r.json()
print(f"  Status: {r.status_code}, Results: {data.get('results', 0)}")