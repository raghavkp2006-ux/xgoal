"use client";

import { useEffect, useState } from "react";
import { AlertCircle, ArrowDown, ArrowUp, RotateCcw } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatDateTime } from "@/lib/format-date";

type Metric = "p_champion" | "p_top4" | "p_top6" | "p_relegation";
type TeamSimulation = {
  team_id: number;
  team: string;
  p_champion: number;
  p_top4: number;
  p_top6: number;
  p_relegation: number;
  expected_final_points: number;
  expected_final_points_95ci: { lower: number; upper: number; level: number };
  finish_position_distribution: Record<string, number>;
};
type TeamDelta = Pick<TeamSimulation, Metric | "team_id" | "expected_final_points">;
type SimulationResponse = {
  season_id: number;
  n_simulations: number;
  as_of_matchday: number;
  run_at: string | null;
  model_version_id: number;
  context?: { model_version?: string | null; fixture_calendar_note?: string | null };
  teams: TeamSimulation[];
  deltas?: TeamDelta[];
};
type Fixture = {
  id: number;
  home_team_id: number;
  away_team_id: number;
  kickoff_utc: string;
  status: string;
};
type ForcedResult = { match_id: number; home_goals: number; away_goals: number };
type LoadState = "loading" | "ready" | "empty" | "error";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const metrics: { key: Metric; label: string }[] = [
  { key: "p_champion", label: "Champion" },
  { key: "p_top4", label: "Top 4" },
  { key: "p_top6", label: "Top 6" },
  { key: "p_relegation", label: "Relegation" },
];

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function Delta({ value, points = false }: { value: number; points?: boolean }) {
  const display = points ? value : value * 100;
  const zero = Math.abs(display) < 0.2;
  return (
    <span className={`inline-flex items-center gap-0.5 whitespace-nowrap text-xs ${zero ? "text-muted-foreground" : display > 0 ? "text-emerald-300" : "text-rose-300"}`}>
      {!zero && (display > 0 ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />)}
      {zero ? "0.0" : Math.abs(display).toFixed(1)}{points ? " pts" : " pp"}
    </span>
  );
}

function SimulatorSkeleton() {
  return (
    <div className="w-full space-y-6 animate-pulse" role="status" aria-label="Loading season simulation">
      <div className="h-10 w-64 rounded bg-muted" />
      <div className="h-28 rounded-xl bg-muted" />
      <div className="h-72 rounded-xl bg-muted" />
      <div className="h-96 rounded-xl bg-muted" />
      <p className="text-sm text-muted-foreground">Loading the cached simulation. A cold backend can take over a minute to start.</p>
    </div>
  );
}

export default function SimulatorPage() {
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<SimulationResponse | null>(null);
  const [scenario, setScenario] = useState<SimulationResponse | null>(null);
  const [scenarioForcedCount, setScenarioForcedCount] = useState(0);
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [forced, setForced] = useState<ForcedResult[]>([]);
  const [selectedFixtureId, setSelectedFixtureId] = useState<number | null>(null);
  const [homeGoals, setHomeGoals] = useState(0);
  const [awayGoals, setAwayGoals] = useState(0);
  const [selectedTeamId, setSelectedTeamId] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [scenarioError, setScenarioError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        // No short timeout: the loading state also covers a 60s backend cold start.
        const response = await fetch(`${API}/api/v1/simulation`, { cache: "no-store", signal: controller.signal });
        if (response.status === 404) {
          setState("empty");
          return;
        }
        if (!response.ok) throw new Error(`Backend request failed (${response.status}).`);
        const data = (await response.json()) as SimulationResponse;
        if (!Array.isArray(data.teams) || data.teams.length === 0) {
          setState("empty");
          return;
        }
        setBaseline(data);
        setSelectedTeamId(data.teams[0].team_id);
        setState("ready");

        const fixturesResponse = await fetch(
          `${API}/api/v1/matches?season_id=${data.season_id}&status=NS&limit=500`,
          { cache: "no-store", signal: controller.signal },
        );
        if (!fixturesResponse.ok) throw new Error(`Unable to load upcoming fixtures (${fixturesResponse.status}).`);
        const upcoming = ((await fixturesResponse.json()) as Fixture[])
          .filter((fixture) => fixture.status === "NS")
          .sort((a, b) => a.kickoff_utc.localeCompare(b.kickoff_utc) || a.id - b.id);
        setFixtures(upcoming);
        setSelectedFixtureId(upcoming[0]?.id ?? null);
      } catch (cause) {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : "Unable to reach the backend.");
        setState("error");
      }
    }
    void load();
    return () => controller.abort();
  }, []);

  const data = scenario ?? baseline;
  const names = new Map(baseline?.teams.map((team) => [team.team_id, team.team]) ?? []);
  const deltas = new Map(scenario?.deltas?.map((delta) => [delta.team_id, delta]) ?? []);
  const chosen = data?.teams.find((team) => team.team_id === selectedTeamId) ?? data?.teams[0];
  const availableFixtures = fixtures.filter((fixture) => !forced.some((result) => result.match_id === fixture.id));
  const fixtureLabel = (fixture: Fixture) =>
    `${formatDateTime(fixture.kickoff_utc)} · ${names.get(fixture.home_team_id) ?? fixture.home_team_id} vs ${names.get(fixture.away_team_id) ?? fixture.away_team_id}`;

  function addForced() {
    const fixture = availableFixtures.find((item) => item.id === selectedFixtureId) ?? availableFixtures[0];
    if (!fixture || forced.length >= 50) return;
    if (![homeGoals, awayGoals].every((goal) => Number.isInteger(goal) && goal >= 0 && goal <= 20)) {
      setScenarioError("Enter whole-number scores from 0 to 20.");
      return;
    }
    setForced((current) => [...current, { match_id: fixture.id, home_goals: homeGoals, away_goals: awayGoals }]);
    setSelectedFixtureId(availableFixtures.find((item) => item.id !== fixture.id)?.id ?? null);
    setHomeGoals(0);
    setAwayGoals(0);
    setScenarioError(null);
  }

  async function runWhatIf() {
    if (!forced.length) return;
    setRunning(true);
    setScenarioError(null);
    try {
      const response = await fetch(`${API}/api/v1/simulation/whatif`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ forced_results: forced }),
      });
      if (!response.ok) {
        const problem = await response.json().catch(() => null);
        throw new Error(problem?.detail ?? `What-if request failed (${response.status}).`);
      }
      setScenario((await response.json()) as SimulationResponse);
      setScenarioForcedCount(forced.length);
    } catch (cause) {
      setScenarioError(cause instanceof Error ? cause.message : "Unable to run the scenario.");
    } finally {
      setRunning(false);
    }
  }

  function reset() {
    setForced([]);
    setScenario(null);
    setScenarioForcedCount(0);
    setScenarioError(null);
    setSelectedFixtureId(fixtures[0]?.id ?? null);
  }

  if (state === "loading") return <SimulatorSkeleton />;
  if (state === "empty") return (
    <div className="w-full space-y-6">
      <h1 className="text-3xl font-semibold">Season simulator</h1>
      <Card><CardContent className="p-8 text-sm text-muted-foreground">No simulation run is available yet. The nightly precompute job will populate the baseline.</CardContent></Card>
    </div>
  );
  if (state === "error" || !data) return (
    <div className="w-full space-y-6">
      <h1 className="text-3xl font-semibold">Season simulator</h1>
      <Card className="border-rose-400/20"><CardContent className="flex gap-3 p-6 text-sm text-rose-200">
        <AlertCircle className="size-5 shrink-0" />
        <p>Simulation is unavailable. {error} Check that the xgoal backend is running and try again.</p>
      </CardContent></Card>
    </div>
  );

  return (
    <div className="w-full space-y-7">
      <header className="space-y-2">
        <div className="flex items-center gap-3"><h1 className="text-3xl font-semibold">Season simulator</h1><Badge variant="muted">La Liga</Badge></div>
        <p className="text-sm text-muted-foreground">{scenario
          ? `What-if run: 2,000 simulations with ${scenarioForcedCount} forced result(s). Base column = the same 2,000-simulation run without your forced results, so it differs slightly from the 10,000-simulation figures.`
          : <>Simulated {data.n_simulations.toLocaleString()} times, as of matchday {data.as_of_matchday}, model version {data.context?.model_version ?? data.model_version_id}. Run {data.run_at ? formatDateTime(data.run_at) : "—"}.</>}</p>
        {data.context?.fixture_calendar_note && <p className="text-xs text-muted-foreground">{data.context.fixture_calendar_note}</p>}
      </header>

      <Card>
        <CardHeader><CardTitle>What-if results</CardTitle><p className="text-sm text-muted-foreground">Pick an upcoming fixture and force its score. What-if runs use 2,000 simulations.</p></CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <label className="min-w-56 flex-1 space-y-1 text-xs text-muted-foreground">Upcoming fixture
              <select className="mt-1 w-full rounded border border-border bg-background p-2 text-sm text-foreground" value={availableFixtures.some((item) => item.id === selectedFixtureId) ? selectedFixtureId! : availableFixtures[0]?.id ?? ""} onChange={(event) => setSelectedFixtureId(Number(event.target.value))}>
                {availableFixtures.map((fixture) => <option key={fixture.id} value={fixture.id}>{fixtureLabel(fixture)}</option>)}
              </select>
            </label>
            <label className="space-y-1 text-xs text-muted-foreground">Home goals
              <input type="number" min={0} max={20} step={1} value={homeGoals} onChange={(event) => setHomeGoals(Number(event.target.value))} className="mt-1 block w-20 rounded border border-border bg-background p-2 text-sm text-foreground" />
            </label>
            <label className="space-y-1 text-xs text-muted-foreground">Away goals
              <input type="number" min={0} max={20} step={1} value={awayGoals} onChange={(event) => setAwayGoals(Number(event.target.value))} className="mt-1 block w-20 rounded border border-border bg-background p-2 text-sm text-foreground" />
            </label>
            <button onClick={addForced} disabled={!availableFixtures.length || forced.length >= 50 || running} className="rounded bg-muted px-4 py-2 text-sm disabled:opacity-50">Add result</button>
          </div>
          {forced.length > 0 && <ul className="space-y-2 text-sm">{forced.map((result) => {
            const fixture = fixtures.find((item) => item.id === result.match_id);
            return <li key={result.match_id} className="flex items-center justify-between gap-3 rounded border border-border p-2">
              <span>{fixture ? fixtureLabel(fixture) : result.match_id} · {result.home_goals}–{result.away_goals}</span>
              <button onClick={() => { setForced((current) => current.filter((item) => item.match_id !== result.match_id)); setSelectedFixtureId(result.match_id); }} className="text-muted-foreground hover:text-foreground" aria-label={`Remove match ${result.match_id}`}>Remove</button>
            </li>;
          })}</ul>}
          <div className="flex flex-wrap gap-3">
            <button onClick={() => void runWhatIf()} disabled={!forced.length || running} className="rounded bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50">{running ? "Running 2,000 simulations…" : "Run what-if"}</button>
            <button onClick={reset} disabled={running} className="inline-flex items-center gap-2 rounded border border-border px-4 py-2 text-sm disabled:opacity-50"><RotateCcw className="size-4" />Reset</button>
          </div>
          {scenarioError && <p role="alert" className="text-sm text-rose-300">{scenarioError}</p>}
        </CardContent>
      </Card>

      <Card className="overflow-hidden">
        <CardHeader><CardTitle>Final table probabilities</CardTitle><p className="text-sm text-muted-foreground">Select a team to see its finish-position distribution. What-if deltas compare paired unforced and forced 2,000-simulation runs with the same seed.</p></CardHeader>
        <CardContent className="overflow-x-auto p-0">
          <table className="w-full min-w-[1140px] table-fixed text-sm">
            <colgroup>
              <col className="w-[240px]" />
              {metrics.map((metric) => <col key={metric.key} className="w-[160px]" />)}
              <col />
            </colgroup>
            <thead className="border-b border-border bg-muted/50 text-left"><tr><th className="p-3">Team</th>{metrics.map((metric) => <th key={metric.key} className="p-3 text-right">P({metric.label})</th>)}<th className="p-3 text-right">Expected points (95% interval)</th></tr></thead>
            <tbody>{data.teams.map((team) => {
              const delta = deltas.get(team.team_id);
              return <tr key={team.team_id} onClick={() => setSelectedTeamId(team.team_id)} className={`cursor-pointer border-b border-border/50 hover:bg-muted/30 ${chosen?.team_id === team.team_id ? "bg-primary/10" : ""}`}>
                <th scope="row" className="whitespace-nowrap p-3 text-left font-medium"><button onClick={() => setSelectedTeamId(team.team_id)}>{team.team}</button></th>
                {metrics.map((metric) => <td key={metric.key} className="p-3 text-right tabular-nums">
                  <div className="whitespace-nowrap leading-5">{percent(team[metric.key])}</div>
                  {scenario && <div className="flex h-5 items-center justify-end gap-2 whitespace-nowrap text-xs leading-5">
                    <span className="text-muted-foreground">base {delta ? percent(team[metric.key] - delta[metric.key]) : "unavailable"}</span>
                    {delta && <Delta value={delta[metric.key]} />}
                  </div>}
                </td>)}
                <td className="p-3 text-right tabular-nums">
                  <div className="whitespace-nowrap leading-5">{team.expected_final_points.toFixed(1)} <span className="text-muted-foreground">({team.expected_final_points_95ci.lower.toFixed(0)}–{team.expected_final_points_95ci.upper.toFixed(0)})</span></div>
                  {scenario && <div className="flex h-5 items-center justify-end gap-2 whitespace-nowrap text-xs leading-5">
                    <span className="text-muted-foreground">base {delta ? (team.expected_final_points - delta.expected_final_points).toFixed(1) : "unavailable"}</span>
                    {delta && <Delta value={delta.expected_final_points} points />}
                  </div>}
                </td>
              </tr>;
            })}</tbody>
          </table>
        </CardContent>
      </Card>

      {chosen && <Card>
        <CardHeader><CardTitle>{chosen.team}: finish-position distribution</CardTitle></CardHeader>
        <CardContent className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={Array.from({ length: 20 }, (_, index) => ({ position: index + 1, probability: (chosen.finish_position_distribution[String(index + 1)] ?? 0) * 100 }))}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="position" label={{ value: "Final position", position: "insideBottom", offset: -5 }} />
              <YAxis tickFormatter={(value: number) => `${value}%`} />
              <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, "Probability"]} />
              <Bar dataKey="probability" fill="hsl(174 72% 45%)" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>}
    </div>
  );
}
