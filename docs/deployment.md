# Deployment

Topology: Neon Postgres, a Render free web service for FastAPI, and Vercel for
Next.js. Scheduled jobs run in GitHub Actions, not inside the Render service.
The repository-root `render.yaml` is the Blueprint; its `rootDir` is `backend`.

## 1. Neon

In the Neon project, open **Connect / Connection Details**, select the intended
branch, database, and role, and copy its PostgreSQL connection string. Keep the
TLS query parameters, including `sslmode=require`. Never put it in frontend env
variables, source control, or command output.

`backend/app/database.py:get_engine()` accepts `postgresql://` and `postgres://`
and rewrites either to `postgresql+psycopg2://`. An explicit `+psycopg2` is also
accepted but is not required. For this runbook use `postgresql://` or
`postgresql+psycopg2://`: Alembic's `env.py` passes the URL directly to SQLAlchemy
and does not perform the application's `postgres://` normalization.

From a clean checkout at the repository root, use PowerShell and Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .\backend
Set-Location backend
$neonSecret = Read-Host "Neon connection string" -AsSecureString
$neonPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($neonSecret)
try {
    $env:DATABASE_URL = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($neonPointer)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($neonPointer)
}
..\.venv\Scripts\python.exe -m alembic upgrade head
Remove-Item Env:DATABASE_URL
```

Run migrations deliberately before serving the corresponding application
version. They change the schema; the Render build command does not run them.
The migration env reads `DATABASE_URL` through settings from the environment
or `backend/.env`; the command above uses the environment without writing a file.

## 2. Render

Create a **Blueprint** from this repository using the root `render.yaml`.
It creates a Python web service on the **free** plan, builds in `backend`, pins
Python 3.12.8, and uses `/health` as its health-check path. There is no Render
database resource: the service connects to Neon.

Configure these environment-variable names in the service dashboard:

- `DATABASE_URL`
- `CORS_ORIGIN`
- `LOG_LEVEL`
- `RATE_LIMIT_DEFAULT`
- `RATE_LIMIT_EXPENSIVE`

`DATABASE_URL` and `CORS_ORIGIN` are manual (`sync: false`) Blueprint inputs.
Set `CORS_ORIGIN` to the exact Vercel production origin and any explicitly allowed
preview origins, separated by commas. Whitespace and empty entries are ignored;
localhost is always allowed. A wildcard origin prevents startup.
The Blueprint supplies the logging and rate-limit defaults. `PYTHON_VERSION`
is pinned by the Blueprint. Rate limits default to 60 requests/minute per IP
per endpoint, with 10/minute on each expensive POST; `/health` is exempt.

The start command enables proxy headers and trusts all forwarding peers as
specified for this Blueprint. Use this service behind Render's managed ingress.
For a directly reachable local deployment, restrict trusted forwarding peers:

```powershell
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --proxy-headers --forwarded-allow-ips "127.0.0.1"
```

The key function reads `request.client.host`, after uvicorn resolves trusted
proxy headers. It does not independently trust `X-Forwarded-For`.

The buildplan's free-service assumption is suspension after about 15 idle
minutes and a 30-60 second cold start. GitHub Actions remains responsible for
scheduled jobs; an in-process scheduler cannot reliably run on a sleeping service.

## 3. Vercel

Import the repository with **Root Directory: frontend** and the Next.js preset.
Set `NEXT_PUBLIC_API_URL` to the HTTPS Render service URL, without a trailing
slash, in the applicable Production/Preview environments. Rebuild the frontend
after changing this public variable. It is a public backend address, not a secret.
Add each intended Vercel origin to Render's `CORS_ORIGIN`.

Requests from `/`, `/standings`, and `/fixtures` run in Vercel server components:
health, competitions, seasons, standings, teams, upcoming matches, and individual
fixture predictions. Requests from `/scoreboard` and `/simulator` run in the
browser: model versions, teams, logged predictions, simulation, upcoming matches,
and the what-if POST. Server requests can share Vercel egress IPs and rate-limit
buckets; browser requests use their resolved client IP. CORS applies to browser
requests, not server-to-server fetches.

## 4. Verify

After deployment, enter public URLs locally; no credentials are needed:

```powershell
$apiBase = Read-Host "Render HTTPS service URL, without a trailing slash"
$vercelOrigin = Read-Host "Exact Vercel HTTPS origin, without a trailing slash"
curl.exe -sS -i --max-time 90 "$apiBase/health"
curl.exe -sS -i --max-time 90 -X OPTIONS `
    -H "Origin: $vercelOrigin" `
    -H "Access-Control-Request-Method: POST" `
    -H "Access-Control-Request-Headers: content-type" `
    "$apiBase/api/v1/simulation/whatif"
```

Health should return HTTP 200 and JSON with `status: "ok"`, `db: true`, and a
`sources` array with per-source age/error fields. Inspect the ages: `ok` means
database connectivity, not that every scheduled job is fresh.
Preflight should return HTTP 200, `Access-Control-Allow-Origin` equal to the
entered Vercel origin, `Access-Control-Allow-Methods` including `POST`, and
`Access-Control-Allow-Headers` including `content-type`. An unlisted origin
should not receive an allow-origin header; its preflight returns HTTP 400.

## 5. Rollback and known limits

Redeploy the previous Render deployment and restore the previous Vercel
deployment through their dashboards. Retain the Neon connection and environment
settings. Do not automatically downgrade migrations, reset the database, or
reload data as an application rollback; keep migrations backward-compatible.

Render's free-tier cold start can take 30-60 seconds; allow a full cold start
before treating a first request as failed. Neon can also resume after inactivity.
The limiter uses process-local memory: counters reset on service restart and are
not shared across multiple workers. Shared Vercel egress addresses can hit a
common per-IP bucket. Explicit preview-origin entries require updating when
preview URLs change. Only the two intentional POSTs (hypothetical prediction and
what-if simulation) are registered; neither persists its result to the database.
