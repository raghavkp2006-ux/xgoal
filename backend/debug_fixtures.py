"""Debug fixtures response structure."""
import httpx, os, json
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("API_FOOTBALL_KEY")
headers = {"x-apisports-key": key}
base = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")

r = httpx.get(f"{base}/fixtures", headers=headers, params={"league": 140, "season": 2024})
data = r.json()
print(f"Results: {data.get('results', 0)}")
if data.get("results", 0) > 0:
    f = data['response'][0]
    print(f"Top-level keys: {list(f.keys())}")
    print(f"Fixture keys: {list(f.get('fixture', {}).keys())}")
    print(f"Teams: {f.get('teams', {})}")
    print(json.dumps(f, indent=2)[:2000])