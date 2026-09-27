"use client";

import React, { useState, useEffect, useRef } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { AlertCircle, ArrowUp, ArrowDown, Minus, Play, Plus, X, Loader2 } from "lucide-react";

// --- Types ---

interface TeamSimulation {
  team_id: number;
  team: string;
  p_champion: number;
  p_top4: number;
  p_top6: number;
  p_relegation: number;
  finish_position_distribution: Record<string, number>;
  expected_final_points: number;
  current_points: number;
  current_goal_difference: number;
  played: number;
}

interface SimulationContext {
  competition: string;
  season: string;
  as_of_matchday: number;
  derived_calendar: boolean;
  fixture_calendar_note: string;
  parameter_bootstrap_fits: number;
  generated_at: string;
}

interface SimulationResponse {
  id?: number;
  season_id: number;
  model_version_id: number;
  n_simulations: number;
  random_seed: number;
  as_of_matchday: number;
  run_at: string;
  teams: TeamSimulation[];
  simulated_matches: number;
  completed_matches: number;
  context: SimulationContext;
  parameter_uncertainty: any;
  baseline_teams?: TeamSimulation[];
  note?: string;
  forced_results?: any[];
}

interface ForcedResult {
  home_team: string;
  away_team: string;
  home_goals: number;
  away_goals: number;
}

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// --- Components ---

function GradientBar({ value, variant = "primary" }: { value: number; variant?: "primary" | "rose" }) {
  const isRose = variant === "rose";
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-xs tabular-nums w-10 text-right">
        {(value * 100).toFixed(1)}%
      </span>
      <div className="w-24 bg-muted/30 rounded-full h-2 overflow-hidden shrink-0">
        <div
          className={cn(
            "h-full rounded-full",
            isRose ? "bg-gradient-to-r from-rose-400/20 to-rose-400" : "bg-gradient-to-r from-primary/20 to-primary"
          )}
          style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
        />
      </div>
    </div>
  );
}

function AnimatedDelta({ value, type }: { value: number; type: "champion" | "relegation" }) {
  const [displayValue, setDisplayValue] = useState(0);

  const isNeutral = Math.abs(value) < 0.001;
  const isGood = type === "champion" ? value > 0 : value < 0;

  useEffect(() => {
    if (isNeutral) {
      setDisplayValue(0);
      return;
    }

    let startTimestamp: number | null = null;
    const duration = 400; // ms

    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      setDisplayValue(value * easeOut);

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        setDisplayValue(value);
      }
    };

    requestAnimationFrame(step);
  }, [value, isNeutral]);

  if (isNeutral) {
    return (
      <span className="font-mono text-xs text-muted-foreground flex items-center justify-end gap-1">
        <Minus className="w-3 h-3" />
        0.0%
      </span>
    );
  }

  const formattedValue = Math.abs(displayValue * 100).toFixed(1) + "%";

  return (
    <span
      className={cn(
        "font-mono text-xs flex items-center justify-end gap-0.5 tabular-nums",
        isGood ? "text-emerald-300" : "text-rose-300"
      )}
    >
      {value > 0 ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
      {formattedValue}
    </span>
  );
}

function PageHeader() {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <h1 className="text-3xl font-semibold tracking-tight">Season Simulator</h1>
        <Badge variant="muted" className="bg-primary/10 text-primary border-primary/20">
          Monte Carlo
        </Badge>
      </div>
      <p className="text-sm text-muted-foreground">
        10,000 season simulations simulating remaining fixtures based on current team ratings and home advantage.
      </p>
    </div>
  );
}

function MetadataCard({ data }: { data: SimulationResponse }) {
  return (
    <Card className="bg-card border-border">
      <CardHeader className="p-5 pb-0 gap-1.5">
        <CardTitle className="text-base font-semibold tracking-tight">Simulation Run Metadata</CardTitle>
      </CardHeader>
      <CardContent className="p-5">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Simulations</p>
            <p className="font-mono text-sm">{data.n_simulations?.toLocaleString()}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Matches (Played / Rem)</p>
            <p className="font-mono text-sm">{data.completed_matches} / {data.simulated_matches}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Random Seed</p>
            <p className="font-mono text-sm">{data.random_seed}</p>
          </div>
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Generated At</p>
            <p className="font-mono text-sm">{data.run_at ? new Date(data.run_at).toLocaleString() : ""}</p>
          </div>
        </div>
        {data.context?.fixture_calendar_note && (
          <div className="mt-4 pt-4 border-t border-border flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
            <p className="text-xs text-muted-foreground">
              {data.context.fixture_calendar_note}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ProbabilityTable({
  data,
  isWhatIf,
}: {
  data: SimulationResponse;
  isWhatIf: boolean;
}) {
  const getBaselineTeam = (teamId: number) => {
    if (!data.baseline_teams) return null;
    return data.baseline_teams.find((t) => t.team_id === teamId);
  };

  return (
    <Card className="bg-card border-border overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse min-w-[800px]">
          <thead>
            <tr className="border-b border-border bg-muted/50">
              <th className="px-5 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">#</th>
              <th className="px-3 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">Team</th>
              <th className="px-3 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground text-right">Pts (xPts)</th>
              <th className="px-3 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">P(Champ)</th>
              {isWhatIf && <th className="px-3 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground text-right">Δ Champ</th>}
              <th className="px-3 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground text-right">P(Top 4)</th>
              <th className="px-3 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground text-right">P(Top 6)</th>
              <th className="px-5 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">P(Relegation)</th>
              {isWhatIf && <th className="px-5 py-3 text-xs font-medium uppercase tracking-wide text-muted-foreground text-right">Δ Releg</th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {data.teams?.map((team, idx) => {
              const baseline = getBaselineTeam(team.team_id);
              const dChamp = baseline ? team.p_champion - baseline.p_champion : 0;
              const dReleg = baseline ? team.p_relegation - baseline.p_relegation : 0;

              return (
                <tr
                  key={team.team_id}
                  className="transition-shadow duration-150 hover:shadow-md hover:shadow-primary/5 hover:bg-muted/10 group"
                >
                  <td className="px-5 py-3 font-mono text-xs text-muted-foreground">{idx + 1}</td>
                  <td className="px-3 py-3 text-sm font-medium">{team.team}</td>
                  <td className="px-3 py-3 text-right">
                    <span className="font-mono text-sm">{team.current_points}</span>
                    <span className="font-mono text-xs text-muted-foreground ml-1">({team.expected_final_points.toFixed(1)})</span>
                  </td>
                  <td className="px-3 py-3">
                    <GradientBar value={team.p_champion} variant="primary" />
                  </td>
                  {isWhatIf && (
                    <td className="px-3 py-3">
                      <AnimatedDelta value={dChamp} type="champion" />
                    </td>
                  )}
                  <td className="px-3 py-3 font-mono text-xs tabular-nums text-right">
                    {(team.p_top4 * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-3 font-mono text-xs tabular-nums text-right">
                    {(team.p_top6 * 100).toFixed(1)}%
                  </td>
                  <td className="px-5 py-3">
                    <GradientBar value={team.p_relegation} variant="rose" />
                  </td>
                  {isWhatIf && (
                    <td className="px-5 py-3">
                      <AnimatedDelta value={dReleg} type="relegation" />
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function FinishDistributionChart({ teams }: { teams: TeamSimulation[] }) {
  const [group, setGroup] = useState<"top6" | "bottom6">("top6");

  const selectedTeams = group === "top6" ? teams.slice(0, 6) : teams.slice(-6);
  const positions = group === "top6" ? [1, 2, 3, 4, 5, 6] : [15, 16, 17, 18, 19, 20];

  const colors = [
    "hsl(174 72% 45%)",
    "hsl(217 90% 60%)",
    "hsl(280 70% 60%)",
    "hsl(340 70% 60%)",
    "hsl(30 90% 60%)",
    "hsl(100 60% 50%)",
  ];

  const chartData = positions.map((pos) => {
    const d: any = { position: `Pos ${pos}` };
    selectedTeams.forEach((t) => {
      d[t.team] = ((t.finish_position_distribution && t.finish_position_distribution[String(pos)]) || 0) * 100;
    });
    return d;
  });

  return (
    <Card className="bg-card border-border">
      <CardHeader className="p-5 pb-0 flex flex-row items-center justify-between gap-1.5">
        <CardTitle className="text-base font-semibold tracking-tight">Finish Position Distribution</CardTitle>
        <div className="flex bg-muted/50 rounded-md p-1">
          <button
            onClick={() => setGroup("top6")}
            className={cn(
              "px-3 py-1 text-xs font-medium rounded-sm transition-colors",
              group === "top6" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
            )}
          >
            Top 6
          </button>
          <button
            onClick={() => setGroup("bottom6")}
            className={cn(
              "px-3 py-1 text-xs font-medium rounded-sm transition-colors",
              group === "bottom6" ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
            )}
          >
            Bottom 6
          </button>
        </div>
      </CardHeader>
      <CardContent className="p-5 h-[350px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 29% 18%)" vertical={false} />
            <XAxis dataKey="position" tick={{ fontSize: 11, fill: "hsl(215 20% 65%)" }} tickLine={false} axisLine={false} />
            <YAxis
              tick={{ fontSize: 11, fill: "hsl(215 20% 65%)" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip
              cursor={{ fill: "hsl(217 25% 15% / 0.4)" }}
              contentStyle={{
                background: "hsl(222 40% 10%)",
                border: "1px solid hsl(217 29% 18%)",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value: number, name: string) => [`${value.toFixed(1)}%`, name]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {selectedTeams.map((t, idx) => (
              <Bar key={t.team_id} dataKey={t.team} stackId="a" fill={colors[idx % colors.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function WhatIfPanel({
  teams,
  onRun,
  onClear,
  isLoading,
  isActive,
}: {
  teams: TeamSimulation[];
  onRun: (results: ForcedResult[]) => void;
  onClear: () => void;
  isLoading: boolean;
  isActive: boolean;
}) {
  const [forcedResults, setForcedResults] = useState<ForcedResult[]>([
    { home_team: teams[0]?.team || "", away_team: teams[1]?.team || "", home_goals: 0, away_goals: 0 },
  ]);

  const teamNames = teams.map((t) => t.team).sort();

  const addMatch = () => {
    setForcedResults([
      ...forcedResults,
      { home_team: teamNames[0], away_team: teamNames[1], home_goals: 0, away_goals: 0 },
    ]);
  };

  const removeMatch = (index: number) => {
    setForcedResults(forcedResults.filter((_, i) => i !== index));
  };

  const updateMatch = (index: number, field: keyof ForcedResult, value: any) => {
    const newResults = [...forcedResults];
    newResults[index] = { ...newResults[index], [field]: value };
    setForcedResults(newResults);
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader className="p-5 pb-4 gap-1.5 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-semibold tracking-tight">What-If Scenarios</CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Force specific match results to see how they impact the probabilities.
            </p>
          </div>
          {isActive && (
            <Badge variant="muted" className="bg-primary/10 text-primary border-primary/20">
              Active Mode
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-5 space-y-4">
        {forcedResults.map((result, idx) => (
          <div key={idx} className="flex flex-col sm:flex-row sm:items-center gap-3 p-3 bg-muted/20 border border-border rounded-md">
            <div className="flex-1">
              <select
                className="w-full bg-background border border-border rounded-md px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-primary"
                value={result.home_team}
                onChange={(e) => updateMatch(idx, "home_team", e.target.value)}
              >
                {teamNames.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2 justify-center">
              <input
                type="number"
                min="0"
                className="w-16 bg-background border border-border rounded-md px-3 py-1.5 text-sm font-mono text-center outline-none focus:ring-1 focus:ring-primary"
                value={result.home_goals}
                onChange={(e) => updateMatch(idx, "home_goals", parseInt(e.target.value) || 0)}
              />
              <span className="text-muted-foreground font-mono">-</span>
              <input
                type="number"
                min="0"
                className="w-16 bg-background border border-border rounded-md px-3 py-1.5 text-sm font-mono text-center outline-none focus:ring-1 focus:ring-primary"
                value={result.away_goals}
                onChange={(e) => updateMatch(idx, "away_goals", parseInt(e.target.value) || 0)}
              />
            </div>
            <div className="flex-1">
              <select
                className="w-full bg-background border border-border rounded-md px-3 py-1.5 text-sm outline-none focus:ring-1 focus:ring-primary"
                value={result.away_team}
                onChange={(e) => updateMatch(idx, "away_team", e.target.value)}
              >
                {teamNames.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={() => removeMatch(idx)}
              disabled={forcedResults.length === 1}
              className="p-1.5 text-muted-foreground hover:text-rose-400 hover:bg-rose-400/10 rounded-md transition-colors disabled:opacity-50"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}

        <div className="flex items-center justify-between pt-2">
          <button
            onClick={addMatch}
            className="flex items-center gap-1.5 text-sm font-medium text-primary hover:text-primary/80 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Add Match
          </button>
          
          <div className="flex items-center gap-3">
            {isActive && (
              <button
                onClick={onClear}
                className="px-4 py-2 text-sm font-medium border border-border rounded-md hover:bg-muted/50 transition-colors"
              >
                Clear What-If
              </button>
            )}
            <button
              onClick={() => onRun(forcedResults)}
              disabled={isLoading}
              className="px-4 py-2 text-sm font-medium bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors flex items-center gap-2 disabled:opacity-70"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              Run Scenario
            </button>
          </div>
        </div>

        <div className="mt-2 text-xs text-muted-foreground flex items-center gap-1.5 border-t border-border pt-4">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          Note: Uses 2,000 simulations (not 10,000). Session-only — not persisted.
        </div>
      </CardContent>
    </Card>
  );
}

function SimulatorSkeleton() {
  return (
    <div className="w-full space-y-8 animate-pulse">
      <div className="space-y-3">
        <div className="h-10 bg-muted rounded w-64" />
        <div className="h-5 bg-muted rounded w-96" />
      </div>
      <div className="h-40 bg-muted rounded-xl" />
      <div className="h-[600px] bg-muted rounded-xl" />
    </div>
  );
}

function ErrorState({ error }: { error: string }) {
  return (
    <Card className="bg-rose-400/10 border-rose-400/20">
      <CardContent className="p-6 flex flex-col items-center justify-center text-center space-y-4">
        <AlertCircle className="w-12 h-12 text-rose-400" />
        <div className="space-y-1">
          <h3 className="text-lg font-medium text-rose-300">Simulation Error</h3>
          <p className="text-sm text-rose-300/80">{error}</p>
        </div>
      </CardContent>
    </Card>
  );
}

// --- Main Page ---

export default function SimulatorPage() {
  const [baselineData, setBaselineData] = useState<SimulationResponse | null>(null);
  const [currentData, setCurrentData] = useState<SimulationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isWhatIfLoading, setIsWhatIfLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchBaseline() {
      try {
        const res = await fetch(`${API}/api/v1/simulation`);
        if (!res.ok) throw new Error(`Failed to load simulation data (${res.status})`);
        const data = await res.json();
        setBaselineData(data);
        setCurrentData(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    }
    fetchBaseline();
  }, []);

  const runWhatIf = async (forcedResults: ForcedResult[]) => {
    if (!baselineData) return;
    setIsWhatIfLoading(true);
    try {
      const res = await fetch(`${API}/api/v1/simulation/whatif`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ forced_results: forcedResults }),
      });
      if (!res.ok) throw new Error("Failed to run what-if simulation");
      const whatIfData = await res.json();
      setCurrentData(whatIfData);
    } catch (err: any) {
      console.error(err);
      // Fallback or toast error? For now just log it or set a small local error state
      alert(err.message);
    } finally {
      setIsWhatIfLoading(false);
    }
  };

  const clearWhatIf = () => {
    setCurrentData(baselineData);
  };

  if (isLoading) return <SimulatorSkeleton />;
  if (error) return <ErrorState error={error} />;
  if (!currentData || !baselineData) return null;

  const isWhatIfActive = !!currentData.forced_results;

  return (
    <div className="w-full space-y-8">
      <PageHeader />

      <MetadataCard data={currentData} />

      <WhatIfPanel
        teams={baselineData.teams || []}
        onRun={runWhatIf}
        onClear={clearWhatIf}
        isLoading={isWhatIfLoading}
        isActive={isWhatIfActive}
      />

      <div className="grid grid-cols-1 gap-8">
        <ProbabilityTable data={currentData} isWhatIf={isWhatIfActive} />
        <FinishDistributionChart teams={currentData.teams || []} />
      </div>
    </div>
  );
}
