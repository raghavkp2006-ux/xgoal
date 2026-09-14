"""Verify API-Football key works against /status and /leagues?id=140."""
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("API_FOOTBALL_KEY")
base_url = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")

headers = {"x-apisports-key": key}

print(f"Testing API key against {base_url}...")

# Test /status
r = httpx.get(f"{base_url}/status", headers=headers)
print(f"\n/status -> {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"  Account: {data.get('response', {}).get('account', {}).get('firstname')} {data.get('response', {}).get('account', {}).get('lastname', '')}")
    print(f"  Requests today: {data.get('response', {}).get('requests', {}).get('current')} / {data.get('response', {}).get('requests', {}).get('limit_day')}")
    print(f"  Subscribed until: {data.get('response', {}).get('subscription', {}).get('end')}")
else:
    print(f"  Error: {r.text[:200]}")

# Test /leagues?id=140
r = httpx.get(f"{base_url}/leagues", headers=headers, params={"id": 140})
print(f"\n/leagues?id=140 -> {r.status_code}")
if r.status_code == 200:
    data = r.json()
    items = data.get("response", [])
    for league in items:
        print(f"  League: {league['league']['name']} ({league['league']['country']})")
        print(f"  Seasons available: {[s['season'] for s in league['seasons']]}")