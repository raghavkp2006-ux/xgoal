"""Quick API key verification."""
import httpx, os, json
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("API_FOOTBALL_KEY")
headers = {"x-apisports-key": key}
base = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")

# /leagues?id=140
r = httpx.get(f"{base}/leagues", headers=headers, params={"id": 140})
data = r.json()
for league in data.get("response", []):
    l = league["league"]
    country = league.get("country", {})
    country_name = country.get("name", "Unknown") if isinstance(country, dict) else country
    print(f"  League: {l['name']} ({country_name})")
    print(f"  Seasons: {[s['season'] for s in league['seasons']]}")
    print(f"  Current season: {[s['season'] for s in league['seasons'] if s.get('current')]}")