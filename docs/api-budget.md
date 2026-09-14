# API-Football Budget

## Free Tier Limits
- **100 requests per day** (hard stop — API returns error, not overage billing)
- **10 requests per minute** (sustained bursting can get key firewalled)

## Verified on Day 1
- [ ] Call `/status` — record `x-ratelimit-requests-limit` and `x-ratelimit-requests-remaining`
- [ ] Call `/leagues?id=140` — confirm accessible seasons
- [ ] Call `/fixtures?league=140&season=2026` — confirm current season accessible

## Budget Allocation

| Job | Cadence | Requests/day (matchday) | Requests/day (idle) |
|---|---|---|---|
| Live poller `/fixtures?live=all&league=140` | every 5 min, only inside live window | ~48 (max 4h) | 0 |
| Fixture refresh `/fixtures?league=140&season=X` | 2×/day | 2 | 2 |
| Match events `/fixtures/events?fixture=N` | per finished match, once | ~10 | 0 |
| Nightly reconcile (last 7 days) | 1×/day | 1 | 1 |
| Player snapshot `/players?league=140&season=X&page=N` | weekly, ~28 pages | 0 (Tue) | 28 (Tue only) |
| Dev/manual headroom | — | ~15 | ~60 |
| **Peak total** | | **~76** | **~31** |

## Enforcement
- `RateLimitedClient` wrapper logs to `api_request_log`
- Refuses calls at **90 requests/day** (10% safety margin)
- Token bucket enforces 10/min limit
- Reads `x-ratelimit-requests-remaining` from response headers
- Never retries a 429 immediately — sustained bursting can get key firewalled

## Self-Gating
The live poller queries Neon first for whether any fixture is currently in a live window.
If no live fixtures exist, it exits without making an API call.
This single decision makes the budget work.

## Escape Hatch
If 100/day proves too tight: $19/mo Pro tier (7,500 requests/day) for the final month.