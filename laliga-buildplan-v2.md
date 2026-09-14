# La Liga Analytics Platform — Build Plan v2

**What this document is:** a review of the v1 plan (33 flaws, ranked by severity) followed by a full rewrite with concrete schema, API contracts, quota budgets, model specs, and testable acceptance gates.

**Read order:** Part 1 tells you what was wrong. Parts 2–8 are the plan you actually build from. If you only read one thing, read Part 1 flaws F1–F5 — they are the ones that would have killed the project in week 11.

---

# PART 1 — FLAW REGISTER

## Fatal — these break the project outright

### F1. The tournament simulator is the wrong feature for a league
The plan's headline feature (Phase 5) is a **single-elimination bracket** built from "top N La Liga teams." La Liga is a double round-robin league. A knockout bracket of La Liga teams is a fantasy scenario with no real-world referent, no ground truth to validate against, and no reason for a reader to trust it.

Worse, it's mechanically incompatible with the Phase 2 model:
- The model outputs **win/draw/loss**. Knockout matches cannot draw. The plan never says what to do with the ~25% draw mass. Renormalising W/L is wrong (it inflates the favourite); it silently biases every bracket.
- Seeding, home/away legs, and neutral venues are all unspecified, but the model has a **home advantage** feature — so every simulated matchup needs a venue decision the plan doesn't make.

**Fix:** replace it with a **season simulator** — simulate the remaining league fixtures 10,000× and report title / top-4 / top-6 / relegation probabilities per team. This is what FiveThirtyEight, Opta, and every serious football model actually publish. It has real ground truth (you can check it against reality every week), it's more impressive, and it removes the venue and draw problems entirely because you're simulating real scheduled fixtures. Keep the bracket as an optional **Copa del Rey mode** (a real Spanish knockout competition) if you have time. See Part 7.

---

### F2. A W/D/L classifier physically cannot drive a league simulator
Even after fixing F1, the model in Phase 2 is insufficient. League tables are ordered by points, then by tiebreakers that require **goals scored and conceded**. A model that outputs P(home win) / P(draw) / P(away win) gives you points but no goal difference, so you cannot resolve the table.

**Fix:** the primary model must be a **scoreline model** (Poisson / Dixon–Coles style) that emits an expected goals rate for each side, from which you sample actual scorelines. W/D/L probabilities are then *derived* by summing the score matrix. One model, both outputs, no inconsistency between the fixtures page and the simulator. The XGBoost classifier becomes a *comparison* model, not the backbone. See Part 6.

---

### F3. The player stats page has no data source
Phase 1 creates a `player_match_stats` table. Phase 1 never populates it. Phase 4 then builds "Player stats page with filters (team, position, stat thresholds)" on top of it.

- football-data.co.uk CSVs contain **zero player-level data**. Match-level only.
- Kaggle La Liga datasets with player stats exist but are stale, unlicensed for redistribution, and inconsistently maintained.
- API-Football *does* have player stats on the free tier, but `/players?league=140&season=X` is paginated at ~20 players/page — roughly 25–30 requests to snapshot one season's La Liga squad stats. Against a **100 request/day** cap, that's a quarter of your daily budget for one refresh.

**Fix:** scope player stats to **current season only**, populated by a **weekly** batched snapshot job, and label it in the UI as "updated weekly." Drop `player_match_stats` (per-match granularity is unobtainable) and replace it with `player_season_stats` (cumulative, snapshotted). Historical player analysis is out of scope — say so in the README rather than leaving a broken page.

---

### F4. The API budget is never computed, and the polling design breaks it
Phase 3 says "respect free-tier rate limits — poll less frequently." That is not a design. The actual constraints on API-Football's free plan:

- **100 requests per day**, hard stop (the API returns an error rather than billing you).
- **10 requests per minute**; sustained bursting can get your key firewalled without warning.
- Free plans are **limited in available seasons** — you cannot backfill 10 seasons of history through this API.

A naive "poll every minute during match windows" on a Saturday (La Liga fixtures spread roughly 14:00–22:00 CET) is ~480 requests. You are 5× over budget before any other call.

**Fix:** an explicit request budget (Part 3), a `api_request_log` table, and a hard client-side governor that refuses the call at 90 requests/day. Critically — **one `/fixtures?live=all&league=140` call returns every live La Liga match at once**, not one call per match. Design around bulk endpoints. Full budget table in Part 3.

---

### F5. Deployment is last, and the recommended database self-destructs
Two separate problems:

**(a) Ordering.** Deployment is Phase 6, week 14. But the Phase 4 checkpoint says *"Someone unfamiliar with the project can open the deployed link."* The plan contradicts itself. More importantly, deferring deployment to the end is the single most common way solo projects die — every environment problem (CORS, cold starts, env vars, migrations against a remote DB, build failures) arrives at once in the final week, when you have no slack.

**(b) Render's free Postgres is deleted 30 days after creation** (with a 14-day grace period to upgrade). On a 15-week project, choosing it means your database vanishes around week 5 — probably right after you finish the ingestion work in Phase 1. Render's free *web services* also spin down after 15 minutes of inactivity with a ~30–60s cold start, and free background workers require a paid instance, so an in-app scheduler is unreliable.

**Fix:** deploy at the end of **Phase 0**, before any features exist, and redeploy continuously thereafter. Use **Neon** for Postgres (free tier scales to zero after ~5 min idle, resumes in under a second, doesn't expire). Avoid Supabase free for this specific project — it **pauses the whole project after 7 days of inactivity** and needs a manual dashboard restore, which is exactly the failure mode that makes a portfolio link dead when a recruiter clicks it. Run scheduled jobs on **GitHub Actions cron**, which is independent of your web service's uptime. Full topology in Part 2.

---

## Major — these produce a broken or indefensible result

### M6. No point-in-time feature discipline
The plan correctly says "split by season, don't shuffle." That prevents *one* kind of leakage and misses the more likely one: the feature builder itself.

"Current league position" and "recent form (last 5 games)" are only meaningful **as of a specific date**. If you compute them with a `GROUP BY team` over the whole `matches` table, every training row for the 2019–20 season silently contains end-of-season information. The model will look excellent and be worthless.

**Fix:** every feature function takes an `as_of` timestamp and may only read matches with `kickoff_utc < as_of`. Enforce it with a test (Part 3, Step 2.2) that recomputes features against a date-filtered database copy and asserts equality.

### M7. The evaluation protocol is too weak to support the claim
- **Accuracy is the wrong headline metric.** La Liga base rates are roughly 46% home / 25% draw / 29% away. Always predicting "home win" scores ~46% accuracy with zero skill. A good model lands around 52–54%. Reporting "53% accurate" without the baseline sounds like a coin flip to a reader.
- **No RPS.** W/D/L is *ordinal* (away < draw < home). The standard football-forecasting metric is the **Ranked Probability Score**, which penalises predicting a home win when the result was an away win more than when it was a draw. Log loss doesn't.
- **A single train/test split** on ~3,800 matches gives you an evaluation set of ~380 matches. The standard error on that is large enough that a 0.01 log-loss difference is noise.
- **The market baseline is free and unused.** The football-data.co.uk CSVs ship with **closing odds** from multiple bookmakers (`B365CH/CD/CA`, `PSCH/PSCD/PSCA`, `AvgCH/AvgCD/AvgCA`). De-vig them and you have the sharpest publicly available forecast to benchmark against. "My model gets within 0.02 log-loss of the closing line" is a far stronger claim than "my model beats Elo."

**Fix:** walk-forward evaluation across ≥5 seasons, reporting log loss, RPS, Brier, and calibration error, against three baselines: uniform, base-rate, Elo, and de-vigged closing odds. Targets in Part 6.

### M8. Calibration is checked but never fixed
Phase 2.4 asks "are your 70% predictions right 70% of the time?" and then does nothing with the answer. A tree ensemble on 3,800 rows will be miscalibrated — usually overconfident.

**Fix:** fit isotonic regression (or Platt scaling for the multiclass case, via one-vs-rest) on a held-out calibration split, applied *after* the model, and report reliability diagrams and expected calibration error before and after. This is a cheap step that visibly improves your numbers and is a strong signal that you know what you're doing.

### M9. 3,800 matches is too few for XGBoost, and the plan doesn't notice
10 seasons × 380 matches = 3,800 rows, with maybe 15–25 features. Gradient boosting will overfit hard. The plan offers no regularisation strategy, no cross-validation scheme, no feature-count discipline.

**Fix:** train on **all divisions available from football-data.co.uk** — 20+ leagues, 10 seasons, roughly 60,000–80,000 matches — with a league identifier and league-specific home-advantage term. The *product* stays La Liga-only; the *training corpus* doesn't have to be. This is the single highest-leverage change to model quality in the whole plan and costs you one afternoon of extra ingestion.

### M10. "Inconsistent team name spellings" is named as the hard part, then left unsolved
The plan flags this correctly and gives no mechanism. The actual problem: football-data.co.uk uses `Ath Bilbao`, `Vallecano`, `Sociedad`, `Espanol`, `Betis`; API-Football uses `Athletic Club`, `Rayo Vallecano`, `Real Sociedad`, `Espanyol`, `Real Betis`. Fuzzy matching alone will confidently merge `Real Madrid` with `Real Sociedad` at some threshold.

**Fix:** a canonical `teams` table keyed to API-Football's `team_id`, plus a `team_aliases(source, raw_name → team_id)` table seeded from a **checked-in YAML file** you build once by hand (~60 distinct names across 10 seasons including relegated clubs). Fuzzy matching *proposes*, a human *approves*, and the result is version-controlled so ingestion is reproducible. Any unmatched name **fails the ingestion loudly** rather than inserting a duplicate team.

### M11. Promoted teams have no history (cold start)
Three teams enter La Liga every August with zero rows in your La Liga history. The plan's feature builder will produce nulls or zeros for them, and the model will treat "no data" as "average team" — when promoted sides historically underperform the league mean substantially.

**Fix:** ingest **Segunda División (SP2)** alongside SP1 so promoted teams carry a prior-division rating, and apply an explicit promotion discount when transferring ratings across divisions (fit the discount from history — it's a nice, small, honest piece of analysis for the README).

### M12. Predictions are never logged, so live model performance is unrecoverable
The plan generates predictions on demand for the fixtures page and stores nothing. That means at the end of the project you cannot answer the most obvious interview question: *"how did it actually do?"* And you can't retrofit it — the pre-match predictions are gone.

**Fix:** a `predictions` table written by a job that runs **before each matchday**, storing model version, feature snapshot hash, and probabilities. Start it in Phase 2, week 6, not later. By your demo you'll have 8+ weeks of genuine out-of-sample results and a "model scoreboard" page that almost no portfolio project has.

### M13. No model versioning or reproducibility
No artifact storage, no random seeds, no training-data snapshot, no way to answer "which model made this prediction."

**Fix:** `model_versions` table (id, algorithm, trained_at, train window, metrics JSON, artifact path, git SHA, seed); artifacts committed to the repo (a Dixon–Coles model is kilobytes) or stored as GitHub release assets; every prediction row references a `model_version_id`.

### M14. Standings are treated as trivial; La Liga's tiebreakers aren't
"Compute live standings from matches" is one bullet. But La Liga does **not** use overall goal difference first the way the Premier League does — it resolves ties by **head-to-head results between the tied clubs first**, then head-to-head goal difference, then overall goal difference, then goals scored. Head-to-head only applies once the tied clubs have played each other twice, which means mid-season your tiebreak logic has a branch. Ties among three or more clubs use a mini-table of matches among only those clubs.

This matters doubly for the simulator: 10,000 simulated seasons each need the tiebreaker applied correctly, or your title/top-4 probabilities are wrong at the margins where they're most interesting.

**Fix:** implement tiebreakers as a standalone, unit-tested pure function with fixtures drawn from real historical seasons that went to a tiebreak. **Verify the current rules against the RFEF/LaLiga competition regulations before implementing** — they have been amended before.

### M15. No tests, no CI, no data-quality gates
One "small validation script" is mentioned. For a portfolio project, a real test suite is a differentiator, and for a data pipeline, it's the only thing standing between you and a demo full of silently wrong numbers.

**Fix:** pytest + GitHub Actions CI from Phase 0. Three test layers: unit (tiebreakers, Elo, feature functions), data-quality (SQL assertions run after every ingestion), and contract (API response shape). See Part 3.

### M16. The live pipeline has no state machine and ignores real-world messiness
Not addressed anywhere: postponed and rescheduled fixtures (routine in La Liga), abandoned matches, VAR score corrections that *decrease* a score after it was reported, kickoff times crossing midnight UTC, and the fact that re-running the poller must not duplicate rows.

**Fix:** explicit `match_status` enum mirroring API-Football's status codes (`NS, 1H, HT, 2H, ET, PEN, FT, AET, PEN_FT, PST, CANC, ABD, SUSP, INT`), all writes as idempotent upserts on `(source, external_id)`, all timestamps stored as `timestamptz` in UTC with display-time conversion in the frontend only, and a nightly reconciliation job that re-fetches the last 7 days of finished fixtures to catch corrections.

### M17. Public endpoints have no caching, no rate limiting, no auth
A public FastAPI service proxying a 100-request/day metered API is a quota bomb. One crawler and your demo is dead for the rest of the day.

**Fix:** the API **never calls API-Football synchronously**. Every user-facing read is served from Postgres, populated by scheduled jobs. Add response caching (ETag + `Cache-Control`), per-IP rate limiting (slowapi), and CORS locked to your Vercel domain.

### M18. Nothing monitors whether the pipeline is still alive
The classic portfolio failure: cron breaks in week 16, you don't notice, and the recruiter sees a standings table frozen three weeks in the past.

**Fix:** a `data_freshness` table updated by every job; `GET /health` returning per-source staleness; a "Data updated X minutes ago" element in the UI footer; and a GitHub Actions workflow that fails loudly (email notification) when a job errors.

### M19. Checkpoints are subjective, not gates
"Sane, calibrated probabilities," "meaningful (non-random-looking) probability distributions," "visually coherent" — none of these can be passed or failed. A gate you can rationalise past isn't a gate.

**Fix:** every checkpoint in Part 3 is a numeric or boolean assertion. Example: *"walk-forward log loss ≤ 1.02 across 5 held-out seasons, and strictly below the Elo baseline on ≥4 of 5."*

### M20. Timeline and dependency diagram contradict each other
The diagram draws Phase 3 with a `↓` from Phase 2 while the annotation says it "runs in parallel conceptually." Phases 0–6 sum to 15 weeks with zero slack for a project explicitly budgeted at "3–4 months." There is no buffer, no cut list, and no statement of how many hours per week are assumed.

**Fix:** a real DAG (Part 2), effort stated in **person-days** rather than calendar weeks, an explicit hours/week assumption, a 2-week buffer, and a cut ladder (Part 8).

---

## Moderate — these degrade quality or cost you time

**D21. `POST /predict {home_team, away_team}` is the wrong contract.** Names are ambiguous ("Athletic" vs "Atlético"), there's no `as_of` date so form features are undefined, no model version in the response, and POST prevents HTTP caching on what is a pure read. → Use `GET /api/v1/predictions/{match_id}` for scheduled fixtures (precomputed and stored) and `POST /api/v1/predictions/hypothetical` with team **IDs** and an explicit `as_of` for arbitrary matchups.

**D22. The Monte Carlo understates uncertainty.** Sampling each match independently from a *fixed* set of model parameters captures outcome randomness only. Real forecast uncertainty also includes parameter uncertainty. → Bootstrap or sample team-strength parameters once per simulation run, then sample outcomes within it. Without this your title probabilities will be too confident.

**D23. 10,000 simulations on request will time out.** On a 512 MB free instance, a naive Python loop over 10,000 × ~200 remaining fixtures is minutes. → Vectorise with NumPy (draw the whole `(n_sims, n_fixtures)` Poisson matrix at once — seconds), and **precompute nightly** into a `simulation_runs` table. The endpoint reads a cached row.

**D24. The feature set makes forward simulation expensive, and the plan doesn't notice.** If features include "current league position" and "form over last 5," then simulating matchday 20 requires recomputing features from 10,000 divergent simulated histories. That's a 10,000× blowup nobody budgeted. → The Dixon–Coles attack/defence parameterisation sidesteps this: strengths are static within a simulation run, so the whole season vectorises. Another reason F2's fix is the right architecture.

**D25. Frontend phase makes no technical decisions.** No data-fetching strategy, no caching, no chart library, no handling of the backend's 30–60s cold start. → TanStack Query + Next.js App Router with `revalidate` on server components; Recharts or visx for charts; a skeleton-loading state that survives a 60s cold start without looking broken.

**D26. No indexes.** "Player stats page with filters (team, position, stat thresholds)" over an unindexed table on 0.5 GB of shared Postgres. → Index definitions are in Part 4.

**D27. The odds data is both an unused asset and an untripped trap.** It's your best benchmark (M7) — but if you feed odds in *as features*, you've built a model that predicts the bookmaker, not the match, and your evaluation becomes meaningless. → Benchmark only. If you want a market-informed variant, build it as a clearly separate second model and report both.

**D28. No xG discussion.** Modern football models lean on expected goals; football-data.co.uk has none. Understat/FBref have xG but require scraping with real ToS and brittleness risk. → Use **shots and shots-on-target**, which *are* in the CSVs (`HS/AS/HST/AST`), as a free and decent shot-quality proxy. Note the xG limitation honestly in the README — knowing what you'd add with more data is itself a good signal.

**D29. Head-to-head record is a weak feature at this sample size.** Two matches per season means a 10-season H2H is ~20 matches, heavily confounded by squad turnover, and largely redundant with team strength. → Keep it if you like, but test whether it earns its place; be prepared to drop it.

**D30. Licensing and ToS are unaddressed.** football-data.co.uk is free for personal use and expects attribution; API-Football's terms govern redistribution and caching of their data on a public site. → Read both, add an attribution footer, and keep the framing analytical rather than betting-adjacent (you're displaying bookmaker odds as a model benchmark, which is a different thing from a tipping site, but the presentation should make that obvious).

**D31. Draw prediction is genuinely hard, and not flagging it will make you think you failed.** A well-calibrated football model rarely assigns a draw the highest probability — draws cluster around 25–30% and almost never top the distribution. If you judge the model on argmax accuracy you'll conclude it "can't predict draws" and start breaking it trying to fix that. → State this up front in the plan and in the README, and evaluate on RPS/log loss where draw *probability* is what's scored.

**D32. The calendar is ignored.** Starting in September, international breaks (September, October, November, March) mean stretches of 10–14 days with **no La Liga fixtures at all**. If you schedule your live-features work into a break, you have nothing to test against. → Pull the fixture calendar in Phase 0 and place Phase 3 work on a window with at least two matchdays in it.

**D33. No MVP definition.** There's no answer to "it's week 12 and I'm behind — what ships?" → Cut ladder in Part 8.

---

# PART 2 — CORRECTED ARCHITECTURE

## Data sources

| Source | Use | Coverage | Cost / limit | Notes |
|---|---|---|---|---|
| football-data.co.uk | All historical training data | 20+ divisions, 10+ seasons; results, half-time, shots, SoT, corners, fouls, cards, referee, full odds incl. closing | Free, personal use, attribute | `https://www.football-data.co.uk/mmz4281/{SSSS}/{DIV}.csv` where `SSSS` is e.g. `2526`, `DIV` is `SP1`/`SP2`/`E0`… No player data. |
| API-Football (v3) | Current season fixtures, live scores, events, teams, players | La Liga = league id `140` (verify via `/leagues`) | **Free: 100 req/day, 10 req/min**, limited historical seasons | Base `https://v3.football.api-sports.io/`, header `x-apisports-key`, GET only. Response 200 with empty `response[]` is normal — check the `errors` field. |
| Neon Postgres | Storage | 0.5 GB, 100 CU-hours/mo free | Free | Scales to zero after ~5 min idle, resumes sub-second, does not expire. |

**Rejected:** Render free Postgres (deleted after 30 days), Supabase free (pauses whole project after 7 days idle — kills a dormant portfolio link), Kaggle player datasets (stale, unclear licensing).

## Deployment topology (stood up in Phase 0, not Phase 6)

```
GitHub repo (monorepo: /backend /frontend /jobs /ml)
  ├── GitHub Actions: CI (pytest, ruff, mypy) on every push
  ├── GitHub Actions: cron jobs ──────────► API-Football ──► Neon Postgres
  │     • live poller (matchday windows only)
  │     • nightly reconcile + standings rebuild
  │     • weekly player snapshot
  │     • pre-matchday prediction writer
  │     • nightly season simulation
  ├── Render free web service (FastAPI) ──► Neon Postgres  [reads only]
  └── Vercel (Next.js) ───────────────────► FastAPI
```

Why jobs live in GitHub Actions rather than the web service: Render's free tier spins down after 15 min idle and free background workers require a paid instance, so an in-process scheduler will miss executions. Actions cron is free on public repos, runs independently of your API's uptime, and gives you logs and failure emails for nothing. Two caveats to design around: **scheduled workflows can be delayed by several minutes under load** (fine for 5–10 min polling, not for second-level precision), and **Actions disables cron on repos with no activity for 60 days** (irrelevant while you're building, relevant for the portfolio afterlife — a monthly commit or a calendar reminder handles it).

Cold start: Render free spins down after 15 min, and the next request takes 30–60s. You have 750 instance-hours/month, and a month is ~730 hours — so keeping one service permanently warm consumes essentially your entire free allowance and leaves nothing for redeploys. Pick one: accept the cold start and build a frontend skeleton state that tolerates it gracefully, or pay $7/month for an always-on instance when you're ready to share the link. For a link you'll actively send to people, $7 is the right call.

## Dependency DAG (replaces the contradictory v1 diagram)

```
P0 Setup + deploy skeleton
      │
      ├──────────────┬───────────────────┐
      ▼              ▼                   ▼
P1 Historical    P1b Current-season   (P4 frontend shell can
   ingestion         ingestion          start any time after P0)
   (CSV)             (API-Football)
      │                   │
      ▼                   ▼
P2 Model  ◄───────── P3 Live pipeline      ← P2 and P3 are genuinely
   (needs P1 only)      (needs P1b only)      parallel; do P3 during a
      │                   │                   week with real matchdays
      ├───────────────────┤
      ▼                   ▼
P4 Frontend (needs P2 + P3 endpoints)
      │
      ▼
P5 Season simulator (needs P2 goals model; UI reuses P4 patterns)
      │
      ▼
P6 Hardening, docs, writeup
```

---

# PART 3 — THE PLAN

**Effort is stated in person-days at ~15 focused hours/week.** At that rate this is a **17–19 week** project including buffer, not 15. If you have 30 hours/week, halve the calendar. Do not compress the person-day estimates; compress the calendar.

---

## Phase 0 — Setup & skeleton deploy (4 person-days)

1. Monorepo: `/backend` (FastAPI), `/frontend` (Next.js), `/jobs`, `/ml`, `/db`. Single repo — cross-cutting changes are constant on a solo project and split repos will cost you more than they save.
2. Python 3.12, `uv` or Poetry, pinned lockfile. FastAPI, SQLAlchemy 2.x, Alembic, pandas, numpy, scipy, scikit-learn, xgboost, httpx, pydantic-settings, pytest, ruff, mypy.
3. Neon project, connection string in `.env`, Alembic initialised with one empty migration that runs cleanly.
4. **API-Football key verification — do this on day one:** call `/status` and record `x-ratelimit-requests-limit` / `-remaining`. Then call `/leagues?id=140` and confirm **exactly which seasons your free key can access**. If it can't reach the current season, the entire live half of the project is dead and you need to know now, not in week 7.
5. Pull the current-season fixture calendar (1 request) and **write down the international break dates.** Schedule Phase 3 around them (D32).
6. Next.js scaffold, one page hitting `GET /health`.
7. **Deploy all three tiers now**: Neon (already remote), FastAPI to Render, Next.js to Vercel. Wire CORS. Get the live URL working with nothing but a health check on it.
8. GitHub Actions CI: ruff + mypy + pytest on every push. One trivial passing test.

**Gate (all must be true):**
- [ ] Public Vercel URL renders data fetched from the public Render URL, which read from Neon.
- [ ] `alembic upgrade head` runs against Neon from a clean checkout.
- [ ] CI green on a fresh clone.
- [ ] Documented in `docs/api-budget.md`: your key's daily limit, accessible seasons, and league id.
- [ ] Season fixture calendar saved; Phase 3 dates chosen.

---

## Phase 1 — Data foundation (12 person-days)

### 1.1 Schema (2d)
Implement the DDL in Part 4 as Alembic migrations. Note what changed from v1: `seasons` and `competitions` tables added; `player_match_stats` **replaced** by `player_season_stats` (F3); `team_aliases`, `predictions`, `model_versions`, `simulation_runs`, `api_request_log`, `data_freshness` added; unique constraints on every external id for idempotent upserts.

### 1.2 Team identity resolution (2d) — do this *before* bulk ingestion
1. Pull canonical teams from API-Football for the current season → `teams` (1 request).
2. Extract every distinct `HomeTeam`/`AwayTeam` string across all SP1 + SP2 CSVs (~90 names).
3. Fuzzy-match (rapidfuzz) to propose mappings; **review every one by hand**; write to `db/aliases/football_data_co_uk.yaml`, committed.
4. Teams that exist only in history (Deportivo, Sporting Gijón, Málaga…) get canonical rows with `source='manual'`.
5. Loader raises `UnknownTeamError` on any unmapped name. Never auto-create.

**Do not skip the manual review.** Fuzzy matching will pair `Real Madrid` with `Real Sociedad` and `Athletic` with `Atlético` at plausible thresholds, and those errors are invisible downstream — they'll just make your model slightly worse forever.

### 1.3 Historical ingestion (3d)
- Download SP1 + SP2, 10 seasons, **plus all other available divisions** for the training corpus (M9). Cache raw CSVs in `data/raw/` and commit a manifest with checksums so ingestion is reproducible.
- Known CSV gotchas to handle explicitly: `Date` format changes between seasons (`26/08/13` vs `11/08/2023`) — parse with `dayfirst=True` and try both patterns; a `Time` column that only exists from ~2019 onward; a BOM on the first header cell; trailing empty rows; blank odds columns for postponed fixtures; bookmaker columns appearing and disappearing between seasons (never assume a fixed column set — select by name, tolerate absence).
- Upsert on `(competition_id, season_id, home_team_id, away_team_id)`.
- Store the de-vigged closing-odds implied probabilities in `matches` for later benchmarking (normalise `1/odds` across the three outcomes).

### 1.4 Current season ingestion (1d)
`/fixtures?league=140&season=2026` returns the **whole season in one request**. `/teams?league=140&season=2026` is one more. Budget: ~5 requests total. Match to canonical teams by API-Football `team_id` directly.

### 1.5 Data quality suite (2d)
A `jobs/validate.py` run after every ingestion, failing the job on any violation:
- Every completed season has exactly 380 SP1 matches (fewer means missing rows; more means duplicates).
- Every team plays exactly 38 matches, 19 home and 19 away, per completed season.
- `status='FT'` ⟹ both scores non-null; `status='NS'` ⟹ both scores null.
- No two matches share `(season, home_team, away_team)`.
- Zero orphaned foreign keys.
- Every `matches.kickoff_utc` falls inside its season's date bounds.
- Goals within a sane range (0–15), all non-negative.
- Reconstructed final standings for each historical season match Wikipedia's for **at least three spot-checked seasons** (this is your end-to-end correctness proof, and it catches alias errors that nothing else will).

### 1.6 Freshness + observability (1d)
`data_freshness` table, updated by every job; `GET /health` surfaces it.

### 1.7 Deploy (1d)
Run migrations and ingestion against production Neon. `GET /api/v1/teams` live on the public URL.

**Gate:**
- [ ] `python -m jobs.validate` exits 0 against production.
- [ ] ≥ 3,700 La Liga matches (10 seasons) and ≥ 55,000 total training matches ingested.
- [ ] Reconstructed final tables match published tables for 3 sampled seasons, exactly, including tiebreak ordering.
- [ ] Re-running full ingestion changes zero rows (idempotency proof — assert on row count and a content hash).
- [ ] Zero unmapped team names.

---

## Phase 2 — Prediction model (15 person-days)

### 2.1 Elo baseline (2d)
Plain Python. Standard update `R' = R + K(S − E)`, `E = 1/(1 + 10^(−ΔR/400))`, draws as S=0.5, plus a home-field bonus added to the home rating before computing E. Tune K (start 20) and home advantage (start ~65 Elo) on a validation window. Add margin-of-victory scaling if you want, and note that raw Elo produces a win probability, not W/D/L — convert with a draw model (an ordered logit on ΔR is the clean way; document the choice).

### 2.2 Point-in-time feature builder (3d)
Signature: `build_features(match_id, as_of: datetime) -> dict`, permitted to read only `kickoff_utc < as_of`.

Feature set (keep it small — ~15 features on this sample size):
- Rolling points-per-game, goals for/against, shots and shots-on-target for/against over last 5 and last 10 matches, computed **separately for home and away venue** (home form at home is more predictive than overall form).
- Days of rest since last match.
- Matchday number (proxy for how settled the season is).
- Elo rating differential going into the match.
- Promotion flag / prior-division adjusted rating for newly promoted sides (M11).
- Division indicator for the multi-league corpus.

Deliberately excluded: current league position (redundant with rolling points and leakage-prone if computed carelessly); H2H (D29 — test it, expect to drop it); bookmaker odds (D27 — benchmark only).

**Leakage test (write it before the features):** for a sample of 200 matches, build features against a database view filtered to `kickoff_utc < match.kickoff_utc`, and assert byte-identical output to the full-database build. If they differ, you have leakage. This test is the most valuable thing in your test suite.

### 2.3 Primary model — Dixon–Coles goals model (4d)
Fit by maximum likelihood: per-team attack `α_i` and defence `β_i` parameters plus a global home advantage `γ`, with

```
λ_home = exp(α_home − β_away + γ)
λ_away = exp(α_away − β_home)
```

Two refinements that matter:
- **Low-score correction `τ(x,y,λ,μ,ρ)`** — plain independent Poisson underestimates 0-0, 1-1, and 1-0/0-1 frequencies. Dixon–Coles corrects exactly these four cells.
- **Exponential time decay** on the likelihood, `w = exp(−ξ·Δdays)`, so recent matches dominate. Tune ξ by walk-forward validation; start near 0.003/day (≈ half-life 8 months) and expect the optimum somewhere in 0.001–0.005.

Identifiability: constrain `mean(α) = 0` (or `sum(α) = 0`), or the fit won't converge to a unique solution. Use `scipy.optimize.minimize` with L-BFGS-B, fixed seed, and assert convergence.

This model gives you a full **score matrix** (P of each scoreline up to 10-10), from which W/D/L, over/under 2.5, and both-teams-to-score all fall out by summation — and, crucially, the goal samples the simulator needs (F2).

### 2.4 Comparison model — XGBoost / logistic regression (2d)
Multinomial logistic regression first (it will be closer to XGBoost than you expect on this data). Then XGBoost with aggressive regularisation: `max_depth` 3–4, `min_child_weight` ≥ 10, `subsample` 0.8, `colsample_bytree` 0.8, early stopping on a validation fold. Hyperparameters chosen by walk-forward CV, never by test-set performance.

### 2.5 Calibration (1d)
Isotonic regression per class (one-vs-rest, then renormalise) fitted on a dedicated calibration split that is **not** the test set. Report reliability diagrams and expected calibration error, before and after, for every model.

### 2.6 Evaluation protocol (2d)
**Walk-forward, not a single split.** For each of the last 5 seasons: train on everything strictly before it, calibrate on the last 20% of the training window, predict every match of the held-out season. Report the pooled and per-season metrics.

Metrics: log loss (primary), **RPS** (football-standard, ordinal-aware), Brier, ECE, accuracy (reported *with* baselines so it can't be misread).

Baselines, all four: uniform (log loss 1.0986); base rate (~1.04–1.05); Elo; de-vigged closing odds (the sharp benchmark, roughly 0.95–0.97 log loss on top European leagues).

Realistic targets — write these into the README as your success criteria *before* you see the results:
- Beat base-rate log loss on all 5 held-out seasons.
- Beat Elo on ≥4 of 5.
- Land within ~0.03 log loss of the de-vigged closing line. Beating the closing line consistently would be an extraordinary result; if you appear to, you have leakage — go look for it.
- Accuracy ~51–54%. Do not chase higher.
- ECE < 0.03 post-calibration.
- **Expect the model to rarely make a draw its argmax prediction.** This is correct behaviour, not a bug (D31).

Report all of it including the parts that don't flatter you. A candidate who reports "my XGBoost did not beat my Dixon–Coles baseline, here's why I think that is" reads as far more credible than one reporting a suspiciously good number.

### 2.7 Serving + prediction logging (1d)
- Retrain the final production model on **all** data with hyperparameters frozen from CV. Persist artifact + `model_versions` row.
- `GET /api/v1/predictions/{match_id}` — reads precomputed rows.
- `POST /api/v1/predictions/hypothetical` — team ids + `as_of`.
- **Prediction writer job (start it now, M12):** every day at 06:00 UTC, write predictions for all fixtures in the next 8 days to `predictions`. This begins accumulating your out-of-sample record from week 6 instead of week 15.

**Gate:**
- [ ] Leakage test passes.
- [ ] Walk-forward numbers hit the targets above, committed to `docs/model-evaluation.md` with reliability diagrams.
- [ ] Full training run reproduces identical metrics from a fixed seed on a clean checkout.
- [ ] `predictions` table is accumulating rows daily in production.

---

## Phase 3 — Live pipeline (8 person-days) — schedule on a real matchday week

### 3.1 Request budget (0.5d)
Write `docs/api-budget.md` and enforce it in code. Working budget against 100/day:

| Job | Cadence | Requests/day (matchday) | Requests/day (idle) |
|---|---|---|---|
| Live poller `/fixtures?live=all&league=140` | every 5 min, only inside a live window | ~48 (max 4h of windows) | 0 |
| Fixture refresh `/fixtures?league=140&season=X` | 2×/day | 2 | 2 |
| Match events `/fixtures/events?fixture=N` | per finished match, once | ~10 | 0 |
| Nightly reconcile (last 7 days) | 1×/day | 1 | 1 |
| Player snapshot `/players?league=140&season=X&page=N` | weekly, ~28 pages | 0 (schedule Tue) | 28 (Tue only) |
| Dev/manual headroom | — | ~15 | ~60 |
| **Peak total** | | **~76** | **~31** |

Enforcement: a `RateLimitedClient` wrapper that logs to `api_request_log`, refuses at 90/day, respects the 10/min limit with a token bucket, and honours `x-ratelimit-requests-remaining` from response headers rather than trusting its own count. Never retry a 429 tightly — sustained bursting can get your key firewalled.

### 3.2 Live poller (2d)
Runs on GitHub Actions cron every 5 minutes, but **self-gates**: it queries Neon first for whether any fixture is currently in a live window, and exits without an API call if not. That single decision is what makes the budget work.

Idempotent upsert on `(source, external_id)`, full status enum (M16), and it must handle a score *decreasing* (VAR) without treating it as corrupt data.

### 3.3 Standings (2d)
Computed from `matches`, never fetched. Pure function `compute_standings(matches, season) -> list[TableRow]` implementing La Liga tiebreakers (M14): points → head-to-head points among tied clubs (only once all relevant fixtures are played) → head-to-head goal difference → overall goal difference → goals scored. Handle 3+ way ties via a mini-table.

Unit tests seeded with real seasons that went to a tiebreak, plus a synthetic 3-way tie. Verify current rules against the official regulations first.

Materialise to `standings_snapshots` after each poller run so the read endpoint is a single indexed query.

### 3.4 Events + reconciliation (2d)
Fetch events once per match after `FT` (not during — that would triple your budget). Nightly reconcile re-fetches the last 7 days of finished fixtures to pick up corrections.

### 3.5 Weekly player snapshot (1.5d)
`/players?league=140&season=X` paginated → `player_season_stats` with `snapshot_date`. Tuesday 04:00 UTC (post-weekend, pre-midweek). Handle mid-season transfers: a player row is `(player_id, team_id, season_id)`, so a January move creates a second row rather than overwriting.

**Gate:**
- [ ] Poller ran through a full real matchday; scores tracked within 5 minutes of reality throughout.
- [ ] Daily API consumption stayed under 80 on the heaviest day, verified from `api_request_log`.
- [ ] Computed standings match LaLiga's official table exactly (including ordering) on 3 consecutive days.
- [ ] Poller re-run on identical data produces zero row changes.
- [ ] Tiebreaker function passes all unit tests including the 3-way case.

---

## Phase 4 — Frontend (12 person-days)

Stack: Next.js App Router, TypeScript, Tailwind, TanStack Query for client fetching, Recharts for charts. Server components with `revalidate` where the data is slow-moving (standings, historical stats); client fetching only for live scores.

- **4.1 Design pass first (1d).** One hour choosing type, spacing, and a two-colour palette before writing components saves the "polish pass" from being a rewrite. Dark mode is nearly free with Tailwind and reads well in screenshots.
- **4.2 Dashboard (2d).** Live scores, standings, next fixtures with model probabilities. Must survive a 60s cold start: skeletons, not spinners; never a blank page. Footer shows data freshness (M18).
- **4.3 Fixtures + predictions (2d).** Probability bars, expected scoreline, and the model's most likely correct scores from the Dixon–Coles score matrix — that last one is visually distinctive and costs nothing extra, since you already computed the matrix.
- **4.4 Model scoreboard page (2d).** *The differentiator.* Your logged pre-match predictions (M12) scored against actual results as they come in: running log loss, RPS, calibration curve, and a comparison against the closing-odds benchmark over the same fixtures. Almost no portfolio project has this, because almost no one logs predictions early enough to have the data.
- **4.5 Team + player explorer (3d).** Filters, sorting, form charts. Label player data "updated weekly" (F3).
- **4.6 Nav, responsive, error states, empty states, a11y pass (2d).**

**Gate:**
- [ ] Every page loads with real production data; no mocks anywhere.
- [ ] Lighthouse performance ≥ 85, accessibility ≥ 95 on mobile.
- [ ] Every page has a defined loading, empty, and error state; unplugging the backend produces a clear message, not a crash.
- [ ] Someone unfamiliar with the project can state what the app does after 10 seconds on the landing page.

---

## Phase 5 — Season simulator (10 person-days)

Replaces the knockout bracket entirely (F1). Spec in Part 7.

- **5.1 Vectorised engine (3d).** NumPy, 10,000 seasons, seeded.
- **5.2 Parameter uncertainty (1d).** Bootstrap team strengths per run (D22).
- **5.3 Nightly precompute job + storage (1d).** Write to `simulation_runs`; endpoint reads cache (D23).
- **5.4 Validation (2d).** Back-test: run the simulator from matchday 20 of each of the last 5 completed seasons and check the actual champion/relegated teams fall in plausible probability regions. Also check internal consistency — every team's finish-position probabilities sum to 1, and expected final points across sims sit near the actual outcome.
- **5.5 UI (2d).** Table of title / top-4 / top-6 / relegation probabilities, a stacked finish-position distribution per team, and expected final points.
- **5.6 What-if mode (1d).** Force a result or a set of results, re-run, show the delta. Cheap once the engine is vectorised, and it demos beautifully.

**Gate:**
- [ ] 10,000 seasons in < 10 seconds locally.
- [ ] Finish-position probabilities sum to 1.00 ± 0.001 for every team.
- [ ] Back-test: on ≥4 of 5 historical seasons, the eventual champion was in the simulator's top 2 by title probability as of matchday 20.
- [ ] Endpoint responds in < 500 ms (serving cached results).

---

## Phase 6 — Hardening, docs, writeup (8 person-days)

1. **Hardening (2d):** per-IP rate limiting, ETag/`Cache-Control` on all reads, CORS locked to the Vercel origin, structured logging, Sentry free tier, secrets audit (confirm no API key ever reached the client bundle).
2. **README (2d):** architecture diagram, data-source table with attribution, model evaluation results with the reliability plot, honest limitations (no xG, no injury/lineup data, weekly-only player stats, free-tier cold start), and "what I'd do with a paid API tier."
3. **`docs/model-evaluation.md` (1d):** the full protocol and every number, including the ones that didn't work.
4. **Demo assets (1d):** a 60-second video and a static screenshot for the CV, so your portfolio survives the backend being asleep.
5. **Technical writeup (2d):** the strongest single artifact here is not "I built a football predictor" — it's *"I benchmarked my model against the closing line and here's the gap."* That's a real, falsifiable, quantitative claim and it demonstrates you understand what a good forecast is.

**Buffer: 2 weeks.** Not optional. Something in Phase 1 or 3 will take twice as long as estimated.

---

# PART 4 — SCHEMA

```sql
-- ============ reference ============
CREATE TABLE competitions (
    id              SERIAL PRIMARY KEY,
    code            TEXT NOT NULL UNIQUE,        -- 'SP1', 'SP2'
    name            TEXT NOT NULL,
    country         TEXT NOT NULL,
    tier            SMALLINT NOT NULL,
    api_football_id INTEGER UNIQUE               -- 140 for La Liga
);

CREATE TABLE seasons (
    id              SERIAL PRIMARY KEY,
    competition_id  INTEGER NOT NULL REFERENCES competitions(id),
    label           TEXT NOT NULL,               -- '2025-26'
    start_year      SMALLINT NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    is_current      BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (competition_id, start_year)
);

CREATE TABLE teams (
    id              SERIAL PRIMARY KEY,
    canonical_name  TEXT NOT NULL UNIQUE,
    short_name      TEXT,
    api_football_id INTEGER UNIQUE,              -- NULL for historical-only clubs
    founded         SMALLINT,
    venue_name      TEXT,
    logo_url        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Solves M10. Seeded from a committed YAML file; never auto-populated.
CREATE TABLE team_aliases (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,                   -- 'football_data_co_uk' | 'api_football'
    raw_name    TEXT NOT NULL,
    team_id     INTEGER NOT NULL REFERENCES teams(id),
    UNIQUE (source, raw_name)
);

-- ============ matches ============
CREATE TYPE match_status AS ENUM (
    'NS','1H','HT','2H','ET','BT','P','FT','AET','PEN',
    'PST','CANC','ABD','SUSP','INT','TBD','AWD','WO'
);

CREATE TABLE matches (
    id                  BIGSERIAL PRIMARY KEY,
    competition_id      INTEGER NOT NULL REFERENCES competitions(id),
    season_id           INTEGER NOT NULL REFERENCES seasons(id),
    matchday            SMALLINT,
    kickoff_utc         TIMESTAMPTZ NOT NULL,
    home_team_id        INTEGER NOT NULL REFERENCES teams(id),
    away_team_id        INTEGER NOT NULL REFERENCES teams(id),
    status              match_status NOT NULL DEFAULT 'NS',
    minute              SMALLINT,

    home_goals          SMALLINT CHECK (home_goals BETWEEN 0 AND 20),
    away_goals          SMALLINT CHECK (away_goals BETWEEN 0 AND 20),
    home_goals_ht       SMALLINT,
    away_goals_ht       SMALLINT,

    home_shots          SMALLINT,  away_shots          SMALLINT,
    home_shots_on_tgt   SMALLINT,  away_shots_on_tgt   SMALLINT,
    home_corners        SMALLINT,  away_corners        SMALLINT,
    home_fouls          SMALLINT,  away_fouls          SMALLINT,
    home_yellows        SMALLINT,  away_yellows        SMALLINT,
    home_reds           SMALLINT,  away_reds           SMALLINT,
    referee             TEXT,

    -- de-vigged closing-odds implied probabilities: the benchmark, NOT a feature (D27)
    closing_p_home      NUMERIC(5,4),
    closing_p_draw      NUMERIC(5,4),
    closing_p_away      NUMERIC(5,4),

    source              TEXT NOT NULL,           -- 'football_data_co_uk' | 'api_football'
    external_id         TEXT,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT no_self_play CHECK (home_team_id <> away_team_id),
    CONSTRAINT finished_has_score CHECK (
        status NOT IN ('FT','AET','PEN') OR (home_goals IS NOT NULL AND away_goals IS NOT NULL)
    ),
    UNIQUE (season_id, home_team_id, away_team_id),
    UNIQUE (source, external_id)
);

CREATE INDEX idx_matches_kickoff       ON matches (kickoff_utc);
CREATE INDEX idx_matches_season_status ON matches (season_id, status);
CREATE INDEX idx_matches_home_time     ON matches (home_team_id, kickoff_utc);
CREATE INDEX idx_matches_away_time     ON matches (away_team_id, kickoff_utc);
CREATE INDEX idx_matches_live          ON matches (status) WHERE status IN ('1H','HT','2H','ET','P');

CREATE TABLE match_events (
    id            BIGSERIAL PRIMARY KEY,
    match_id      BIGINT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    minute        SMALLINT NOT NULL,
    extra_minute  SMALLINT,
    team_id       INTEGER REFERENCES teams(id),
    player_id     INTEGER,
    assist_id     INTEGER,
    event_type    TEXT NOT NULL,                 -- Goal | Card | subst | Var
    detail        TEXT,
    external_id   TEXT,
    UNIQUE (match_id, minute, event_type, player_id, detail)
);

-- ============ players (current season only — see F3) ============
CREATE TABLE players (
    id              SERIAL PRIMARY KEY,
    api_football_id INTEGER UNIQUE,
    name            TEXT NOT NULL,
    nationality     TEXT,
    birth_date      DATE,
    position        TEXT
);

CREATE TABLE player_season_stats (
    id                SERIAL PRIMARY KEY,
    player_id         INTEGER NOT NULL REFERENCES players(id),
    team_id           INTEGER NOT NULL REFERENCES teams(id),
    season_id         INTEGER NOT NULL REFERENCES seasons(id),
    snapshot_date     DATE NOT NULL,
    appearances       SMALLINT, lineups SMALLINT, minutes INTEGER,
    goals             SMALLINT, assists SMALLINT,
    shots             SMALLINT, shots_on_target SMALLINT,
    passes            INTEGER,  pass_accuracy NUMERIC(5,2),
    yellows           SMALLINT, reds SMALLINT,
    rating            NUMERIC(4,2),
    UNIQUE (player_id, team_id, season_id, snapshot_date)   -- transfers create a 2nd row
);
CREATE INDEX idx_pss_lookup ON player_season_stats (season_id, team_id, snapshot_date DESC);
CREATE INDEX idx_pss_goals  ON player_season_stats (season_id, goals DESC);

-- ============ standings ============
CREATE TABLE standings_snapshots (
    id            BIGSERIAL PRIMARY KEY,
    season_id     INTEGER NOT NULL REFERENCES seasons(id),
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    as_of_matchday SMALLINT,
    position      SMALLINT NOT NULL,
    team_id       INTEGER NOT NULL REFERENCES teams(id),
    played SMALLINT, won SMALLINT, drawn SMALLINT, lost SMALLINT,
    goals_for SMALLINT, goals_against SMALLINT, points SMALLINT,
    form          TEXT,                          -- 'WWDLW'
    UNIQUE (season_id, computed_at, team_id)
);
CREATE INDEX idx_standings_latest ON standings_snapshots (season_id, computed_at DESC);

-- ============ models & predictions ============
CREATE TABLE model_versions (
    id                SERIAL PRIMARY KEY,
    name              TEXT NOT NULL,             -- 'dixon_coles' | 'xgb_wdl' | 'elo'
    version           TEXT NOT NULL,
    git_sha           TEXT NOT NULL,
    random_seed       INTEGER NOT NULL,
    trained_at        TIMESTAMPTZ NOT NULL,
    train_start       DATE NOT NULL,
    train_end         DATE NOT NULL,
    n_train_matches   INTEGER NOT NULL,
    hyperparameters   JSONB NOT NULL,
    eval_metrics      JSONB NOT NULL,            -- log_loss, rps, brier, ece per season
    artifact_path     TEXT NOT NULL,
    is_production     BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (name, version)
);

-- Written BEFORE kickoff. This table is the whole point of M12.
CREATE TABLE predictions (
    id                BIGSERIAL PRIMARY KEY,
    match_id          BIGINT NOT NULL REFERENCES matches(id),
    model_version_id  INTEGER NOT NULL REFERENCES model_versions(id),
    predicted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    as_of             TIMESTAMPTZ NOT NULL,      -- feature cutoff
    p_home NUMERIC(6,5) NOT NULL,
    p_draw NUMERIC(6,5) NOT NULL,
    p_away NUMERIC(6,5) NOT NULL,
    expected_home_goals NUMERIC(4,2),
    expected_away_goals NUMERIC(4,2),
    score_matrix      JSONB,                     -- full Dixon-Coles grid
    feature_hash      TEXT NOT NULL,
    CONSTRAINT probs_sum_to_one CHECK (abs(p_home + p_draw + p_away - 1) < 0.001),
    UNIQUE (match_id, model_version_id, as_of)
);
CREATE INDEX idx_predictions_match ON predictions (match_id);

CREATE TABLE simulation_runs (
    id                SERIAL PRIMARY KEY,
    season_id         INTEGER NOT NULL REFERENCES seasons(id),
    model_version_id  INTEGER NOT NULL REFERENCES model_versions(id),
    run_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    n_simulations     INTEGER NOT NULL,
    random_seed       INTEGER NOT NULL,
    as_of_matchday    SMALLINT NOT NULL,
    forced_results    JSONB,                     -- what-if mode
    results           JSONB NOT NULL             -- per team: title/top4/top6/releg/pos dist/exp pts
);
CREATE INDEX idx_simruns_latest ON simulation_runs (season_id, run_at DESC);

-- ============ operations ============
CREATE TABLE api_request_log (
    id              BIGSERIAL PRIMARY KEY,
    requested_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    endpoint        TEXT NOT NULL,
    params          JSONB,
    status_code     SMALLINT,
    quota_remaining INTEGER,                     -- from x-ratelimit-requests-remaining
    duration_ms     INTEGER
);
CREATE INDEX idx_api_log_day ON api_request_log (requested_at);

CREATE TABLE data_freshness (
    source          TEXT PRIMARY KEY,            -- 'live_scores' | 'standings' | 'players' | ...
    last_success_at TIMESTAMPTZ,
    last_attempt_at TIMESTAMPTZ,
    last_error      TEXT,
    rows_affected   INTEGER
);
```

---

# PART 5 — API CONTRACT

All read endpoints are served **from Postgres only** — never a synchronous call to API-Football (M17). All responses carry `ETag` and `Cache-Control`.

```
GET  /health                                  → {status, db, sources:[{name, age_seconds}]}
GET  /api/v1/teams?season=2026
GET  /api/v1/teams/{team_id}                  → profile, form, season stats
GET  /api/v1/matches?season=&from=&to=&team_id=&status=
GET  /api/v1/matches/live                     → currently in-play (cache 30s)
GET  /api/v1/matches/{match_id}               → detail + events + prediction
GET  /api/v1/standings?season=2026            → latest snapshot, tiebreak-ordered
GET  /api/v1/predictions/{match_id}           → precomputed, with model_version
POST /api/v1/predictions/hypothetical         → {home_team_id, away_team_id, as_of?} (D21)
GET  /api/v1/models                           → versions + eval metrics
GET  /api/v1/models/scoreboard?season=2026    → live scored performance vs baselines (M12)
GET  /api/v1/simulation?season=2026           → cached latest run
POST /api/v1/simulation/whatif                → {forced_results:[{match_id, home_goals, away_goals}]}
GET  /api/v1/players?season=&team_id=&position=&min_minutes=&sort=&limit=&offset=
```

Conventions: versioned prefix; integer IDs everywhere, never team names; cursor or offset pagination with a hard `limit` cap of 100; RFC 7807 problem+json errors; per-IP rate limit 60 req/min via slowapi; CORS restricted to the Vercel origin plus localhost.

---

# PART 6 — MODEL SPECIFICATION

**Primary: Dixon–Coles.** Attack/defence per team + global home advantage, low-score correction ρ, exponential time decay ξ, identifiability constraint `sum(α)=0`. Fit with L-BFGS-B, fixed seed, convergence asserted. Emits a full score matrix → W/D/L by summation, plus goal samples for the simulator.

**Comparison: multinomial logistic + XGBoost** on the point-in-time features from 2.2, heavily regularised, walk-forward CV, multi-league corpus.

**Baseline: Elo**, hand-rolled.

**Calibration:** isotonic per class on a dedicated split, applied post-hoc to every model.

**Evaluation:** walk-forward over 5 seasons. Log loss (primary), RPS (ordinal-aware, the football standard), Brier, ECE, accuracy-with-baselines.

**Benchmarks:** uniform 1.0986 · base rate ~1.045 · Elo ~1.00–1.02 · your model target ≤1.00 · de-vigged closing line ~0.95–0.97. *(These are approximate reference points for top European leagues, not guarantees — establish your own baselines from your own data and report those.)*

**Anti-leakage rules, enforced by tests:**
1. No feature reads a match with `kickoff_utc >= as_of`.
2. No bookmaker odds as features.
3. Calibration split ≠ test split.
4. Hyperparameters selected on validation folds only; the test season is touched exactly once.
5. Final production model retrained on everything with frozen hyperparameters, and reported metrics come from the walk-forward run, not from the final fit.

**Known limitations to state in the README (this is a strength, not an admission):** no xG (shots-on-target as proxy); no lineup, injury, or suspension data; no European-competition fixture congestion signal beyond rest days; no manager changes; no transfer-window squad-strength adjustment; player stats weekly only.

---

# PART 7 — SEASON SIMULATOR SPECIFICATION

**Input:** current `matches` state for the season, remaining fixtures, fitted Dixon–Coles parameters.

**Algorithm:**
```
for run in 1..10_000:
    θ = sample_team_strengths()          # bootstrap → parameter uncertainty (D22)
    λ = compute_lambdas(remaining_fixtures, θ)   # (n_fixtures, 2)
    goals = poisson_sample(λ)                    # vectorised across all fixtures
    apply_low_score_correction(goals)
    table = current_table + accumulate(goals)
    positions[run] = rank_with_laliga_tiebreakers(table)
```
Vectorise the outer loop too — sample the entire `(10_000, n_fixtures, 2)` goal tensor in one call. Target: under 10 seconds locally, then cache.

**Output per team:** P(champion), P(top 4), P(top 6), P(relegation), full finish-position distribution, expected final points with a credible interval.

**Correctness checks:**
- Every team's position distribution sums to 1.
- Position probabilities across teams sum to 1 for each position.
- Expected total points across all teams ≈ 3 × 380 × (1 − draw_rate/3)… simpler check: total points across the table equals `3×(matches − draws) + 2×draws` in every simulated season.
- Back-test from matchday 20 on 5 completed seasons.

**Serving:** nightly precompute → `simulation_runs`; endpoint reads the cached row. What-if runs use a reduced 2,000 simulations for interactivity and say so in the UI.

**Optional Copa del Rey mode** (only if ahead of schedule): a real knockout with real draws, using the score matrix to sample 90-minute results and an explicit extra-time/penalties rule for level scores (extra time as a scaled-down λ, penalties as a coin flip weighted slightly by team strength). This is the *correct* way to do the thing v1 wanted to do, and doing it correctly requires the goals model — which is why it comes last.

---

# PART 8 — RISK REGISTER & CUT LADDER

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Free API key can't access current season | Medium | Fatal to live features | Verify on **day 1** (Phase 0.4). Fallback: football-data.co.uk updates weekly, so the app degrades to weekly refresh rather than dying. |
| 100 req/day proves too tight | Medium | High | Budget + governor + self-gating poller. Escape hatch: $19/mo Pro tier (7,500/day) for the final month only. |
| Team alias errors corrupt training data silently | Medium | High | Manual review + fail-loud on unmapped + standings reconciliation against published tables. |
| Model doesn't beat Elo | Medium | Low | This is a *result*, not a failure. Report it honestly with analysis. It's still a portfolio-worthy writeup. |
| Render cold start ruins the demo | High | Medium | Skeleton states; a demo video that doesn't depend on the live service; $7/mo when actively sharing the link. |
| Phase 1 overruns | High | Medium | The 2-week buffer exists for exactly this. |
| GitHub Actions cron drifts or pauses | Low | Medium | 5-min cadence tolerates minutes of drift; monthly commit keeps the repo active. |
| Scope creep into multi-league | Medium | Medium | Multi-league is **training corpus only**. The product stays La Liga. |

## Cut ladder — what ships if you're behind

**Tier 1 — must ship (this is the MVP):** Phase 1 data + Phase 2 model with honest walk-forward evaluation + a fixtures page showing predictions + deployed link + README. This alone is a strong project.

**Tier 2 — ship if possible:** live scores, computed standings, model scoreboard page.

**Tier 3 — the differentiators:** season simulator, what-if mode.

**Tier 4 — cut first:** player explorer, match events, Copa del Rey mode, blog post.

Cut from the bottom. **Never cut the evaluation rigour to buy features** — a well-evaluated model with two pages beats a shaky model with six.

---

# APPENDIX A — Known alias mappings (starter set)

Verify and extend during Phase 1.2. Left column is football-data.co.uk's spelling.

```yaml
football_data_co_uk:
  "Ath Bilbao":   "Athletic Club"
  "Ath Madrid":   "Atlético Madrid"
  "Sociedad":     "Real Madrid"        # ← WRONG ON PURPOSE. Fix it. This is the
                                       #   exact class of error fuzzy matching makes
                                       #   and eyeballing catches. Correct: Real Sociedad.
  "Espanol":      "Espanyol"
  "Betis":        "Real Betis"
  "Vallecano":    "Rayo Vallecano"
  "Celta":        "Celta Vigo"
  "La Coruna":    "Deportivo La Coruña"
  "Sp Gijon":     "Sporting Gijón"
  "Alaves":       "Deportivo Alavés"
  "Cadiz":        "Cádiz"
  "Malaga":       "Málaga"
  "Almeria":      "Almería"
  "Leganes":      "Leganés"
  "Gimnastic":    "Gimnàstic Tarragona"
```

# APPENDIX B — La Liga tiebreakers

Applied in order for clubs level on points:
1. Head-to-head points among the tied clubs (only once all matches between them have been played)
2. Head-to-head goal difference among the tied clubs
3. Overall goal difference
4. Overall goals scored

For three or more tied clubs, build a mini-table of matches among only those clubs and apply the same order. If head-to-head fixtures are incomplete, fall through to overall goal difference.

**Verify against the current RFEF / LaLiga competition regulations before implementing** — these rules have been amended in the past, and the simulator's title probabilities depend on getting them exactly right.

# APPENDIX C — Sources

- API-Football rate limits and pricing: https://www.api-football.com/pricing · https://www.api-football.com/news/post/how-ratelimit-works
- API-Football docs (caching guidance): https://www.api-football.com/documentation-v3
- football-data.co.uk data: https://www.football-data.co.uk/mmz4281/2526/SP1.csv (pattern: `/mmz4281/{season}/{div}.csv`)
- Render free tier behaviour: https://render.com/articles/platforms-with-a-real-free-tier-for-developers-in-2026
- Neon vs Supabase free tiers: https://neon.com/guides/neon-vs-supabase-free-plan
- Dixon, M. & Coles, S. (1997), *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, Applied Statistics 46(2) — the source for the low-score correction and time decay.
- Constantinou & Fenton (2012), *Solving the Problem of Inadequate Scoring Rules for Assessing Probabilistic Football Forecast Models* — the case for RPS.

*Free-tier limits and hosting terms change frequently. Re-verify every number in Part 2 on the day you start.*
