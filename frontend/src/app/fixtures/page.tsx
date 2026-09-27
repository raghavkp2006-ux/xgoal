import { AlertCircle, CalendarClock, HelpCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type Competition = { id: number; code: string };
type Season = { id: number; label: string; is_current: boolean };
type Team = { id: number; name: string; short_name: string | null };
type Match = {
  id: number;
  kickoff_utc: string;
  status: string;
  home_team_id: number;
  away_team_id: number;
  home_goals: number | null;
  away_goals: number | null;
};
type Prediction = {
  match_id: number | null;
  model_name: string;
  model_version: string;
  p_home: number;
  p_draw: number;
  p_away: number;
  expected_home_goals: number | null;
  expected_away_goals: number | null;
};

const MATCH_LIMIT = 20;
const FINISHED_STATUSES = new Set(["FT", "AET", "PEN"]);

export const dynamic = "force-dynamic";

async function request<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Backend request failed (${response.status}).`);
  return response.json() as Promise<T>;
}

async function getPrediction(api: string, matchId: number): Promise<Prediction | null> {
  const response = await fetch(`${api}/api/v1/predictions/${matchId}`, { cache: "no-store" });
  if (!response.ok) return null;
  return (await response.json()) as Prediction;
}

async function getFixturesData() {
  const api = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
  const competitions = await request<Competition[]>(`${api}/api/v1/competitions`);
  const sp1 = competitions.find((competition) => competition.code === "SP1");
  if (!sp1) throw new Error("SP1 competition is not available from the backend.");
  const seasons = await request<Season[]>(`${api}/api/v1/competitions/${sp1.id}/seasons`);
  const season = seasons.find((item) => item.is_current);
  if (!season) throw new Error("The backend has no current SP1 season.");

  const [matches, teams] = await Promise.all([
    request<Match[]>(`${api}/api/v1/matches?season_id=${season.id}&limit=${MATCH_LIMIT}`),
    request<Team[]>(`${api}/api/v1/teams`),
  ]);

  const predictions = await Promise.all(matches.map((match) => getPrediction(api, match.id)));
  const predictionByMatch = new Map<number, Prediction>();
  matches.forEach((match, index) => {
    const prediction = predictions[index];
    if (prediction) predictionByMatch.set(match.id, prediction);
  });

  return { season, matches, teams: new Map(teams.map((team) => [team.id, team])), predictionByMatch };
}

export default async function FixturesPage() {
  try {
    const data = await getFixturesData();
    return (
      <div className="w-full space-y-8">
        <PageHeader season={data.season.label} />
        {data.matches.length ? (
          <div className="space-y-4">
            {data.matches.map((match) => (
              <MatchCard
                key={match.id}
                match={match}
                homeTeam={data.teams.get(match.home_team_id)}
                awayTeam={data.teams.get(match.away_team_id)}
                prediction={data.predictionByMatch.get(match.id) ?? null}
              />
            ))}
          </div>
        ) : (
          <EmptyState season={data.season.label} />
        )}
      </div>
    );
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : "Unable to load fixtures.";
    return (
      <div className="w-full space-y-8">
        <PageHeader />
        <Card className="border-rose-400/20">
          <CardContent className="flex gap-3 p-5 text-sm text-rose-200">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <div>
              <p className="font-medium">Fixtures are unavailable</p>
              <p className="mt-1 text-rose-200/75">{detail} Check that the xgoal backend is running and try again.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }
}

function PageHeader({ season }: { season?: string }) {
  return (
    <section className="flex flex-wrap items-end justify-between gap-4">
      <div className="space-y-2">
        <Badge variant="muted">Recent fixtures</Badge>
        <h1 className="text-3xl font-semibold tracking-tight">Results &amp; forecasts</h1>
        <p className="text-sm text-muted-foreground">The most recent La Liga matches, with a Dixon-Coles forecast wherever one has been logged.</p>
      </div>
      {season ? <Badge variant="success">{season} current season</Badge> : null}
    </section>
  );
}

function EmptyState({ season }: { season: string }) {
  return (
    <Card>
      <CardContent className="flex flex-col items-center gap-3 px-5 py-14 text-center">
        <span className="grid size-11 place-items-center rounded-full bg-muted text-muted-foreground">
          <CalendarClock className="size-5" />
        </span>
        <div>
          <p className="font-medium">No fixtures yet</p>
          <p className="mt-1 max-w-md text-sm text-muted-foreground">The backend identifies {season} as the current SP1 season, but has not returned any matches for it.</p>
        </div>
      </CardContent>
    </Card>
  );
}

function formatKickoff(iso: string) {
  return new Date(iso).toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

function MatchCard({ match, homeTeam, awayTeam, prediction }: { match: Match; homeTeam?: Team; awayTeam?: Team; prediction: Prediction | null }) {
  const played = FINISHED_STATUSES.has(match.status);
  const homeName = homeTeam?.name ?? `Team ${match.home_team_id}`;
  const awayName = awayTeam?.name ?? `Team ${match.away_team_id}`;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex-row items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <CalendarClock className="size-3.5" /> {formatKickoff(match.kickoff_utc)}
          </p>
          <CardTitle className="mt-1 text-lg">
            {homeName} <span className="text-muted-foreground">vs</span> {awayName}
          </CardTitle>
        </div>
        <div className="text-right">
          {played ? (
            <p className="text-2xl font-semibold tabular-nums">
              {match.home_goals} – {match.away_goals}
            </p>
          ) : (
            <Badge variant="warning">{match.status}</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        {prediction ? (
          <PredictionDetail prediction={prediction} homeName={homeName} awayName={awayName} />
        ) : (
          <div className="flex items-center gap-2 rounded-lg border border-dashed border-border/80 px-4 py-3 text-sm text-muted-foreground">
            <HelpCircle className="size-4 shrink-0" />
            No prediction has been logged for this fixture yet.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function PredictionDetail({ prediction, homeName, awayName }: { prediction: Prediction; homeName: string; awayName: string }) {
  const { p_home, p_draw, p_away, expected_home_goals, expected_away_goals } = prediction;
  return (
    <div className="space-y-4">
      <ProbabilityBar homeLabel={homeName} awayLabel={awayName} pHome={p_home} pDraw={p_draw} pAway={p_away} />
      {expected_home_goals !== null && expected_away_goals !== null ? (
        <p className="text-sm text-muted-foreground">
          Expected scoreline{" "}
          <span className="font-mono font-medium text-foreground">
            {expected_home_goals.toFixed(2)} – {expected_away_goals.toFixed(2)}
          </span>
        </p>
      ) : null}
      <p className="text-xs text-muted-foreground">
        {prediction.model_name} v{prediction.model_version}
      </p>
    </div>
  );
}

function ProbabilityBar({ homeLabel, awayLabel, pHome, pDraw, pAway }: { homeLabel: string; awayLabel: string; pHome: number; pDraw: number; pAway: number }) {
  const pct = (value: number) => `${(value * 100).toFixed(0)}%`;
  return (
    <div className="space-y-2">
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted" role="img" aria-label={`${homeLabel} win ${pct(pHome)}, draw ${pct(pDraw)}, ${awayLabel} win ${pct(pAway)}`}>
        <div className="h-full bg-primary" style={{ width: pct(pHome) }} />
        <div className="h-full bg-muted-foreground/40" style={{ width: pct(pDraw) }} />
        <div className="h-full bg-rose-400" style={{ width: pct(pAway) }} />
      </div>
      <div className="flex justify-between text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-primary" /> {homeLabel} {pct(pHome)}</span>
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-muted-foreground/40" /> Draw {pct(pDraw)}</span>
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-rose-400" /> {awayLabel} {pct(pAway)}</span>
      </div>
    </div>
  );
}
