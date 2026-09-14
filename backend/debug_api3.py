"""Check what seasons have data available."""
import httpx, os, json
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("API_FOOTBALL_KEY")
headers = {"x-apisports-key": key}
base = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")

# Get seasons for La Liga
r = httpx.get(f"{base}/leagues", headers=headers, params={"id": 140})
data = r.json()
seasons = data['response'][0]['seasons']
print("Available seasons:")
for s in seasons:
    print(f"  Year: {s['year']}, Current: {s.get('current')}, Start: {s.get('start')}, End: {s.get('end')}")

# Try the most recent season that has data
latest = max(s['year'] for s in seasons)
print(f"\nLatest season: {latest}")

# Check teams for recent seasons
for yr in [2024, 2023, 2022, latest]:
    r = httpx.get(f"{base}/teams", headers=headers, params={"league": 140, "season": yr})
    print(f"\nTeams for {yr}: {r.json().get('results', 0)}")
    if r.json().get('results', 0) > 0:
        print(f"  First: {r.json()['response'][0]['team']['name']}")
        break