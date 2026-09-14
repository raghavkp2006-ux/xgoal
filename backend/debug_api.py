"""Debug API response structure for leagues."""
import httpx, os, json
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("API_FOOTBALL_KEY")
headers = {"x-apisports-key": key}
base = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")

r = httpx.get(f"{base}/leagues", headers=headers, params={"id": 140})
data = r.json()
league = data["response"][0]

# Print league info
print("=== League Info ===")
print(json.dumps(league["league"], indent=2))
print("\n=== Country ===")
print(json.dumps(league.get("country"), indent=2))
print("\n=== Seasons (first 2) ===")
print(json.dumps(league["seasons"][:2], indent=2))
print(f"\nTotal seasons: {len(league['seasons'])}")