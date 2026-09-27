"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Gauge, ListChecks, ListOrdered, Scale, Target, TrendingUp } from "lucide-react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// Uninformed baselines: guessing the uniform 1/3 chance for every match.
const BASELINE_LOG_LOSS = Math.log(3);
const BASELINE_ACCURACY = 100 / 3;
const EPSILON = 1e-9;
const UNIFORM = { home: 1 / 3, draw: 1 / 3, away: 1 / 3 };

type ModelVersion = {
  id: number;
  name: string;
  version: string;
  is_production: boolean;
  trained_at: string;
  train_start: string;
  train_end: string;
  n_train_matches: number;
  prediction_count: number;
};

type PredictionLog = {
  id: number;
  match_id: number;
  model_version_id: number;
  model_name: string;
  model_version: string;
  as_of: string;
  p_home: number;
  p_draw: number;
  p_away: number;
  expected_home_goals: number | null;
  expected_away_goals: number | null;
  kickoff_utc: string;
  status: string;
  home_team_id: number;
  away_team_id: number;
  home_goals: number | null;
  away_goals: number | null;
  // De-vigged closing-odds implied probabilities — the market benchmark
  // (buildplan Part 4.4/6). Null when the source row has no closing odds.
  closing_p_home: number | null;
  closing_p_draw: number | null;
  closing_p_away: number | null;
};

type Team = { id: number; name: string };

type Outcome = "H" | "D" | "A";

type ScoredPrediction = PredictionLog & {
  outcome: Outcome;
  pOfOutcome: number;
  predictedOutcome: Outcome;
  correct: boolean;
  logLoss: number;
  rps: number;
  hasMarket: boolean;
  marketLogLoss: number | null;
  marketRps: number | null;
};

type LoadState = "loading" | "ready" | "error";

function outcomeOf(homeGoals: number, awayGoals: number): Outcome {
  if (homeGoals > awayGoals) return "H";
  if (homeGoals < awayGoals) return "A";
  return "D";
}

function argmaxOutcome(pHome: number, pDraw: number, pAway: number): Outcome {
  if (pHome >= pDraw && pHome >= pAway) return "H";
  if (pAway >= pDraw && pAway >= pHome) return "A";
  return "D";
}

/**
 * Ranked Probability Score for a 3-outcome ordered (away < draw < home)
 * forecast: RPS = 0.5 * [(p_home - o_home)^2 + ((p_home+p_draw) - (o_home+o_draw))^2]
 * Unlike log loss, this penalises predicting a home win when the result was
 * an away win more than when it was a draw — the football-standard metric
 * (buildplan M7 / Part 6).
 */
function rpsOf(pHome: number, pDraw: number, pAway: number, outcome: Outcome): number {
  const oHome = outcome === "H" ? 1 : 0;
  const oDraw = outcome === "D" ? 1 : 0;
  const term1 = pHome - oHome;
  const term2 = pHome + pDraw - (oHome + oDraw);
  return 0.5 * (term1 * term1 + term2 * term2);
}

function mean(values: number[]): number {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function scorePredictions(predictions: PredictionLog[]): ScoredPrediction[] {
  return predictions
    .filter((p) => p.home_goals !== null && p.away_goals !== null)
    .map((p) => {
      const outcome = outcomeOf(p.home_goals as number, p.away_goals as number);
      const pOfOutcome = outcome === "H" ? p.p_home : outcome === "D" ? p.p_draw : p.p_away;
      const predictedOutcome = argmaxOutcome(p.p_home, p.p_draw, p.p_away);
      const hasMarket = p.closing_p_home !== null && p.closing_p_draw !== null && p.closing_p_away !== null;
      const marketPOfOutcome = hasMarket
        ? outcome === "H"
          ? (p.closing_p_home as number)
          : outcome === "D"
            ? (p.closing_p_draw as number)
            : (p.closing_p_away as number)
        : null;
      return {
        ...p,
        outcome,
        pOfOutcome,
        predictedOutcome,
        correct: predictedOutcome === outcome,
        logLoss: -Math.log(Math.max(pOfOutcome, EPSILON)),
        rps: rpsOf(p.p_home, p.p_draw, p.p_away, outcome),
        hasMarket,
        marketLogLoss: hasMarket ? -Math.log(Math.max(marketPOfOutcome as number, EPSILON)) : null,
        marketRps: hasMarket
          ? rpsOf(p.closing_p_home as number, p.closing_p_draw as number, p.closing_p_away as number, outcome)
          : null,
      };
    });
}

function runningSeries(scored: ScoredPrediction[]) {
  let logLossSum = 0;
  let correctSum = 0;
  let rpsSum = 0;
  let marketLogLossSum = 0;
  let marketRpsSum = 0;
  let marketN = 0;
  return scored.map((p, index) => {
    logLossSum += p.logLoss;
    correctSum += p.correct ? 1 : 0;
    rpsSum += p.rps;
    const n = index + 1;
    // The market cumulative average only ever advances over fixtures that
    // actually have closing odds, so the model-vs-market lines stay a fair,
    // same-fixtures comparison even if some rows are missing odds.
    if (p.marketLogLoss !== null && p.marketRps !== null) {
      marketLogLossSum += p.marketLogLoss;
      marketRpsSum += p.marketRps;
      marketN += 1;
    }
    return {
      index: n,
      label: `${p.home_team_id}-${p.away_team_id}`,
      date: new Date(p.kickoff_utc).toLocaleDateString(undefined, { day: "2-digit", month: "short" }),
      cumulativeLogLoss: logLossSum / n,
      cumulativeAccuracy: (correctSum / n) * 100,
      cumulativeRPS: rpsSum / n,
      cumulativeMarketLogLoss: marketN > 0 ? marketLogLossSum / marketN : null,
      cumulativeMarketRPS: marketN > 0 ? marketRpsSum / marketN : null,
    };
  });
}

const CALIBRATION_BIN_WIDTH = 0.2;

function calibrationBins(scored: ScoredPrediction[]) {
  const bins = Array.from({ length: Math.round(1 / CALIBRATION_BIN_WIDTH) }, (_, i) => ({
    lower: i * CALIBRATION_BIN_WIDTH,
    upper: (i + 1) * CALIBRATION_BIN_WIDTH,
    predictedSum: 0,
    observedSum: 0,
    n: 0,
  }));

  for (const p of scored) {
    const points: [number, number][] = [
      [p.p_home, p.outcome === "H" ? 1 : 0],
      [p.p_draw, p.outcome === "D" ? 1 : 0],
      [p.p_away, p.outcome === "A" ? 1 : 0],
    ];
    for (const [predicted, observed] of points) {
      const binIndex = Math.min(bins.length - 1, Math.floor(predicted / CALIBRATION_BIN_WIDTH));
      bins[binIndex].predictedSum += predicted;
      bins[binIndex].observedSum += observed;
      bins[binIndex].n += 1;
    }
  }

  return bins
    .filter((bin) => bin.n > 0)
    .map((bin) => ({
      range: `${Math.round(bin.lower * 100)}–${Math.round(bin.upper * 100)}%`,
      midpoint: ((bin.lower + bin.upper) / 2) * 100,
      predicted: (bin.predictedSum / bin.n) * 100,
      observed: (bin.observedSum / bin.n) * 100,
      n: bin.n,
    }));
}

function expectedCalibrationError(bins: ReturnType<typeof calibrationBins>, totalPoints: number) {
  if (!totalPoints) return 0;
  const weighted = bins.reduce((sum, bin) => sum + bin.n * Math.abs(bin.predicted - bin.observed), 0);
  return weighted / totalPoints / 100;
}

export default function ScoreboardPage() {
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [teams, setTeams] = useState<Map<number, Team>>(new Map());
  const [selectedVersionId, setSelectedVersionId] = useState<number | null>(null);
  const [predictions, setPredictions] = useState<PredictionLog[]>([]);
  const [predictionsLoading, setPredictionsLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [versionsResponse, teamsResponse] = await Promise.all([
          fetch(`${API}/api/v1/model-versions`, { cache: "no-store" }),
          fetch(`${API}/api/v1/teams`, { cache: "no-store" }),
        ]);
        if (!versionsResponse.ok) throw new Error(`Backend request failed (${versionsResponse.status}).`);
        if (!teamsResponse.ok) throw new Error(`Backend request failed (${teamsResponse.status}).`);
        const versionsData = (await versionsResponse.json()) as ModelVersion[];
        const teamsData = (await teamsResponse.json()) as Team[];
        if (cancelled) return;
        setVersions(versionsData);
        setTeams(new Map(teamsData.map((team) => [team.id, team])));
        // Predictions from different fitted models are never comparable in
        // aggregate, so default to the single version with the most resolved,
        // logged predictions rather than pooling every version together.
        const scoreable = versionsData.filter((v) => v.prediction_count > 0);
        const best = scoreable.sort((a, b) => b.prediction_count - a.prediction_count)[0];
        setSelectedVersionId(best?.id ?? null);
        setState("ready");
      } catch (cause) {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : "Unable to reach the backend.");
        setState("error");
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (selectedVersionId === null) {
      setPredictions([]);
      return;
    }
    let cancelled = false;
    async function loadPredictions() {
      setPredictionsLoading(true);
      try {
        const response = await fetch(
          `${API}/api/v1/predictions?model_version_id=${selectedVersionId}&resolved_only=true&limit=2000`,
          { cache: "no-store" },
        );
        if (!response.ok) throw new Error(`Backend request failed (${response.status}).`);
        const data = (await response.json()) as PredictionLog[];
        if (cancelled) return;
        setPredictions(data);
      } catch (cause) {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : "Unable to load predictions.");
        setState("error");
      } finally {
        if (!cancelled) setPredictionsLoading(false);
      }
    }
    loadPredictions();
    return () => {
      cancelled = true;
    };
  }, [selectedVersionId]);

  const scored = useMemo(() => scorePredictions(predictions), [predictions]);
  const series = useMemo(() => runningSeries(scored), [scored]);
  const bins = useMemo(() => calibrationBins(scored), [scored]);
  const ece = useMemo(() => expectedCalibrationError(bins, scored.length * 3), [bins, scored.length]);
  const selectedVersion = versions.find((v) => v.id === selectedVersionId) ?? null;

  // RPS's uninformed baseline depends on which outcome actually happened (it
  // is not outcome-invariant like log loss's ln(3)), so it is computed as the
  // mean RPS a uniform 1/3-1/3-1/3 forecast would have scored against the
  // actual outcome sequence in this dataset.
  const baselineRPS = useMemo(
    () => (scored.length ? mean(scored.map((p) => rpsOf(UNIFORM.home, UNIFORM.draw, UNIFORM.away, p.outcome))) : 0),
    [scored],
  );

  // Model vs. closing line must be compared over the exact same fixtures, so
  // this recomputes the model's own log loss/RPS restricted to only the
  // matches that actually have closing odds — not the full resolved set.
  const marketComparison = useMemo(() => {
    const matched = scored.filter((p) => p.hasMarket);
    if (!matched.length) return null;
    const modelLogLoss = mean(matched.map((p) => p.logLoss));
    const marketLogLoss = mean(matched.map((p) => p.marketLogLoss as number));
    const modelRps = mean(matched.map((p) => p.rps));
    const marketRps = mean(matched.map((p) => p.marketRps as number));
    return {
      n: matched.length,
      modelLogLoss,
      marketLogLoss,
      modelRps,
      marketRps,
      logLossGap: modelLogLoss - marketLogLoss,
      rpsGap: modelRps - marketRps,
    };
  }, [scored]);

  const overallLogLoss = series.length ? series[series.length - 1].cumulativeLogLoss : null;
  const overallAccuracy = series.length ? series[series.length - 1].cumulativeAccuracy : null;
  const overallRPS = series.length ? series[series.length - 1].cumulativeRPS : null;

  if (state === "error") {
    return (
      <div className="w-full space-y-8">
        <PageHeader />
        <Card className="border-rose-400/20">
          <CardContent className="flex gap-3 p-5 text-sm text-rose-200">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <div>
              <p className="font-medium">The model scoreboard is unavailable</p>
              <p className="mt-1 text-rose-200/75">{error} Check that the xgoal backend is running and try again.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (state === "loading") {
    return <ScoreboardSkeleton />;
  }

  return (
    <div className="w-full space-y-8">
      <PageHeader />

      <VersionPicker versions={versions} selectedId={selectedVersionId} onChange={setSelectedVersionId} />

      {!selectedVersion ? (
        <EmptyState />
      ) : predictionsLoading ? (
        <ScoreboardSkeleton compact />
      ) : scored.length === 0 ? (
        <NoResolvedState version={selectedVersion} />
      ) : (
        <>
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4" aria-label="Headline model metrics">
            <StatCard
              icon={Target}
              label="Accuracy"
              value={`${overallAccuracy!.toFixed(1)}%`}
              detail={`vs ${BASELINE_ACCURACY.toFixed(1)}% uninformed baseline`}
              positive={overallAccuracy! > BASELINE_ACCURACY}
            />
            <StatCard
              icon={Gauge}
              label="Log loss"
              value={overallLogLoss!.toFixed(3)}
              detail={`vs ${BASELINE_LOG_LOSS.toFixed(3)} uninformed baseline`}
              positive={overallLogLoss! < BASELINE_LOG_LOSS}
              lowerIsBetter
            />
            <StatCard
              icon={ListOrdered}
              label="RPS"
              value={overallRPS!.toFixed(3)}
              detail={`vs ${baselineRPS.toFixed(3)} uninformed baseline`}
              positive={overallRPS! < baselineRPS}
              lowerIsBetter
            />
            <StatCard icon={ListChecks} label="Matches scored" value={String(scored.length)} detail={`${selectedVersion.name} ${selectedVersion.version}`} />
            <StatCard
              icon={TrendingUp}
              label="Calibration error"
              value={ece.toFixed(3)}
              detail="mean |predicted − observed|, weighted by bin size"
              positive={ece < 0.1}
              lowerIsBetter
            />
            {marketComparison ? (
              <StatCard
                icon={Scale}
                label="Log loss gap vs. closing line"
                value={`${marketComparison.logLossGap >= 0 ? "+" : ""}${marketComparison.logLossGap.toFixed(3)}`}
                detail={`model − market, n=${marketComparison.n} fixtures with odds`}
                positive={marketComparison.logLossGap <= 0}
                lowerIsBetter
              />
            ) : (
              <StatCard icon={Scale} label="Log loss gap vs. closing line" value="—" detail="no fixtures with closing odds for this version" />
            )}
            {marketComparison ? (
              <StatCard
                icon={Scale}
                label="RPS gap vs. closing line"
                value={`${marketComparison.rpsGap >= 0 ? "+" : ""}${marketComparison.rpsGap.toFixed(3)}`}
                detail={`model − market, n=${marketComparison.n} fixtures with odds`}
                positive={marketComparison.rpsGap <= 0}
                lowerIsBetter
              />
            ) : (
              <StatCard icon={Scale} label="RPS gap vs. closing line" value="—" detail="no fixtures with closing odds for this version" />
            )}
          </section>
          {marketComparison && marketComparison.logLossGap < -0.02 ? (
            <p className="-mt-4 text-xs text-amber-300">
              This model is beating the closing line by more than 0.02 log loss on the fixtures it has odds for — per
              the buildplan, that&rsquo;s unusual enough to be a leakage smell, not a result to celebrate outright.
            </p>
          ) : null}

          <section className="grid gap-4 lg:grid-cols-2" aria-label="Running performance over time">
            <RunningChart
              title="Running log loss"
              subtitle="Cumulative average as each match resolves, chronologically, vs. the de-vigged closing line on the same fixtures."
              data={series}
              dataKey="cumulativeLogLoss"
              color="#2dd4bf"
              baseline={BASELINE_LOG_LOSS}
              formatValue={(v) => v.toFixed(3)}
              marketDataKey="cumulativeMarketLogLoss"
              marketLabel="Closing line"
            />
            <RunningChart
              title="Running RPS"
              subtitle="Cumulative Ranked Probability Score, the ordinal-aware football-standard metric, vs. the closing line."
              data={series}
              dataKey="cumulativeRPS"
              color="#a78bfa"
              baseline={baselineRPS}
              formatValue={(v) => v.toFixed(3)}
              marketDataKey="cumulativeMarketRPS"
              marketLabel="Closing line"
            />
            <div className="lg:col-span-2">
              <RunningChart
                title="Running accuracy"
                subtitle="Cumulative correct-outcome rate as each match resolves."
                data={series}
                dataKey="cumulativeAccuracy"
                color="#34d399"
                baseline={BASELINE_ACCURACY}
                formatValue={(v) => `${v.toFixed(1)}%`}
              />
            </div>
          </section>

          <CalibrationChart bins={bins} matchCount={scored.length} />

          <PredictionLedger scored={scored} teams={teams} />
        </>
      )}
    </div>
  );
}

function PageHeader() {
  return (
    <section className="flex flex-wrap items-end justify-between gap-4">
      <div className="space-y-2">
        <Badge variant="muted">Model scoreboard</Badge>
        <h1 className="text-3xl font-semibold tracking-tight">How well is xgoal forecasting?</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Every metric below is scoped to a single fitted model version. Predictions from different model fits are
          never pooled together, since they aren&rsquo;t comparable in aggregate.
        </p>
      </div>
    </section>
  );
}

function VersionPicker({
  versions,
  selectedId,
  onChange,
}: {
  versions: ModelVersion[];
  selectedId: number | null;
  onChange: (id: number) => void;
}) {
  if (!versions.length) return null;
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <label htmlFor="model-version" className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Scoring model version
          </label>
          <p className="mt-0.5 text-xs text-muted-foreground">Only versions with logged, resolved predictions can be scored.</p>
        </div>
        <select
          id="model-version"
          value={selectedId ?? ""}
          onChange={(event) => onChange(Number(event.target.value))}
          className="w-full rounded-lg border border-border bg-muted px-3 py-2 text-sm sm:w-auto sm:min-w-[320px]"
        >
          {versions.map((version) => (
            <option key={version.id} value={version.id} disabled={version.prediction_count === 0}>
              {version.name} {version.version} — {version.prediction_count} prediction{version.prediction_count === 1 ? "" : "s"}
              {version.is_production ? " · production" : ""}
              {version.prediction_count === 0 ? " (none logged)" : ""}
            </option>
          ))}
        </select>
      </CardContent>
    </Card>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  detail,
  positive,
  lowerIsBetter,
}: {
  icon: typeof Target;
  label: string;
  value: string;
  detail: string;
  positive?: boolean;
  lowerIsBetter?: boolean;
}) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          <Icon className="size-3.5" /> {label}
        </div>
        <p className="mt-2 text-3xl font-semibold tabular-nums tracking-tight">{value}</p>
        <p
          className={`mt-1.5 text-xs ${
            positive === undefined ? "text-muted-foreground" : positive ? "text-emerald-300" : "text-rose-300"
          }`}
        >
          {detail}
          {positive !== undefined ? (lowerIsBetter ? (positive ? " (better)" : " (worse)") : positive ? " (better)" : " (worse)") : ""}
        </p>
      </CardContent>
    </Card>
  );
}

function RunningChart({
  title,
  subtitle,
  data,
  dataKey,
  color,
  baseline,
  formatValue,
  marketDataKey,
  marketLabel,
}: {
  title: string;
  subtitle: string;
  data: ReturnType<typeof runningSeries>;
  dataKey: "cumulativeLogLoss" | "cumulativeAccuracy" | "cumulativeRPS";
  color: string;
  baseline: number;
  formatValue: (value: number) => string;
  marketDataKey?: "cumulativeMarketLogLoss" | "cumulativeMarketRPS";
  marketLabel?: string;
}) {
  const withBaseline = data.map((point) => ({ ...point, baseline }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <p className="text-sm text-muted-foreground">{subtitle}</p>
      </CardHeader>
      <CardContent className="h-64 pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={withBaseline} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 29% 18%)" vertical={false} />
            <XAxis
              dataKey="index"
              tick={{ fontSize: 11, fill: "hsl(215 20% 65%)" }}
              tickLine={false}
              axisLine={false}
              label={{ value: "match #, chronological", position: "insideBottom", offset: -2, fontSize: 11, fill: "hsl(215 20% 65%)" }}
            />
            <YAxis tick={{ fontSize: 11, fill: "hsl(215 20% 65%)" }} tickLine={false} axisLine={false} width={44} />
            <Tooltip
              contentStyle={{ background: "hsl(222 40% 10%)", border: "1px solid hsl(217 29% 18%)", borderRadius: 8, fontSize: 12 }}
              labelFormatter={(label, payload) => (payload?.[0] ? `${payload[0].payload.date} · match #${label}` : `match #${label}`)}
              formatter={(value: number, name: string) => [value === null || value === undefined ? "—" : formatValue(value), name]}
            />
            <Line
              type="monotone"
              dataKey="baseline"
              stroke="hsl(215 20% 65%)"
              strokeDasharray="4 4"
              dot={false}
              isAnimationActive={false}
              name="Baseline"
            />
            {marketDataKey ? (
              <Line
                type="monotone"
                dataKey={marketDataKey}
                stroke="#fbbf24"
                strokeWidth={2}
                strokeDasharray="2 2"
                dot={false}
                isAnimationActive={false}
                connectNulls
                name={marketLabel ?? "Closing line"}
              />
            ) : null}
            <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} name={title} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function CalibrationChart({ bins, matchCount }: { bins: ReturnType<typeof calibrationBins>; matchCount: number }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Calibration</CardTitle>
        <p className="text-sm text-muted-foreground">
          Predicted probability vs. how often that outcome actually happened, binned in 20-point buckets across every
          home/draw/away probability the model issued. The dashed line is perfect calibration.
        </p>
      </CardHeader>
      <CardContent className="h-72 pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={bins} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 29% 18%)" vertical={false} />
            <XAxis dataKey="range" tick={{ fontSize: 11, fill: "hsl(215 20% 65%)" }} tickLine={false} axisLine={false} />
            <YAxis
              tick={{ fontSize: 11, fill: "hsl(215 20% 65%)" }}
              tickLine={false}
              axisLine={false}
              width={44}
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
            />
            <Tooltip
              contentStyle={{ background: "hsl(222 40% 10%)", border: "1px solid hsl(217 29% 18%)", borderRadius: 8, fontSize: 12 }}
              formatter={(value: number, name: string, entry) => {
                if (name === "Observed rate") return [`${value.toFixed(1)}% (n=${entry.payload.n})`, name];
                return [`${value.toFixed(1)}%`, name];
              }}
            />
            <Bar dataKey="observed" name="Observed rate" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
            <Line type="monotone" dataKey="midpoint" name="Perfect calibration" stroke="hsl(215 20% 65%)" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
      <p className="px-5 pb-5 text-xs text-muted-foreground">
        n = {matchCount} matches from a single frozen model fit — bin-level noise at this sample size is large, so
        read this as a rough shape, not a precise measurement (buildplan M7).
      </p>
    </Card>
  );
}

function PredictionLedger({ scored, teams }: { scored: ScoredPrediction[]; teams: Map<number, Team> }) {
  const rows = [...scored].reverse();
  return (
    <Card className="overflow-hidden">
      <CardHeader className="border-b border-border">
        <CardTitle>Resolved predictions</CardTitle>
        <p className="text-sm text-muted-foreground">Most recent first · {rows.length} matches this model has been scored against.</p>
      </CardHeader>
      <CardContent className="overflow-x-auto p-0">
        <table className="w-full min-w-[720px] text-sm">
          <thead className="bg-muted/50 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-5 py-3 font-medium">Fixture</th>
              <th className="px-3 py-3 text-center font-medium">Predicted H/D/A</th>
              <th className="px-3 py-3 text-center font-medium">Actual</th>
              <th className="px-3 py-3 text-center font-medium">Pick</th>
              <th className="px-5 py-3 text-right font-medium">Log loss</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const home = teams.get(row.home_team_id)?.name ?? `Team ${row.home_team_id}`;
              const away = teams.get(row.away_team_id)?.name ?? `Team ${row.away_team_id}`;
              return (
                <tr key={row.id} className="border-t border-border/80">
                  <td className="px-5 py-3">
                    <p className="font-medium">
                      {home} <span className="text-muted-foreground">vs</span> {away}
                    </p>
                    <p className="text-xs text-muted-foreground">{new Date(row.kickoff_utc).toLocaleDateString()}</p>
                  </td>
                  <td className="px-3 py-3 text-center font-mono text-xs text-muted-foreground">
                    {(row.p_home * 100).toFixed(0)} / {(row.p_draw * 100).toFixed(0)} / {(row.p_away * 100).toFixed(0)}
                  </td>
                  <td className="px-3 py-3 text-center font-medium tabular-nums">
                    {row.home_goals}–{row.away_goals} <span className="text-muted-foreground">({row.outcome})</span>
                  </td>
                  <td className="px-3 py-3 text-center">
                    <Badge variant={row.correct ? "success" : "danger"}>{row.predictedOutcome}</Badge>
                  </td>
                  <td className="px-5 py-3 text-right font-mono text-xs text-muted-foreground">{row.logLoss.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function EmptyState() {
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 px-5 py-14 text-center">
        <span className="grid size-11 place-items-center rounded-full bg-muted text-muted-foreground">
          <Gauge className="size-5" />
        </span>
        <div>
          <p className="font-medium">No scoreable model versions yet</p>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">
            The backend has fitted model versions, but none of them have any logged predictions to score.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function NoResolvedState({ version }: { version: ModelVersion }) {
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 px-5 py-14 text-center">
        <span className="grid size-11 place-items-center rounded-full bg-muted text-muted-foreground">
          <Gauge className="size-5" />
        </span>
        <div>
          <p className="font-medium">No resolved predictions for {version.name} {version.version}</p>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">
            This model version has logged predictions, but none of the matches they refer to have finished yet.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function ScoreboardSkeleton({ compact = false }: { compact?: boolean }) {
  return (
    <div className="w-full space-y-8" aria-label="Loading model scoreboard">
      {!compact ? (
        <div className="space-y-3">
          <div className="h-5 w-36 animate-pulse rounded bg-muted" />
          <div className="h-9 w-96 animate-pulse rounded bg-muted" />
          <div className="h-4 w-full max-w-2xl animate-pulse rounded bg-muted" />
        </div>
      ) : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <Card key={i}>
            <CardContent className="space-y-3 p-5">
              <div className="h-3 w-20 animate-pulse rounded bg-muted" />
              <div className="h-8 w-16 animate-pulse rounded bg-muted" />
              <div className="h-3 w-32 animate-pulse rounded bg-muted" />
            </CardContent>
          </Card>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        {Array.from({ length: 2 }, (_, i) => (
          <Card key={i}>
            <CardHeader>
              <div className="h-4 w-32 animate-pulse rounded bg-muted" />
              <div className="h-3 w-48 animate-pulse rounded bg-muted" />
            </CardHeader>
            <CardContent className="h-64">
              <div className="h-full w-full animate-pulse rounded bg-muted" />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
