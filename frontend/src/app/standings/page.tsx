import { AlertCircle, Trophy } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatDateTime } from "@/lib/format-date";

type Competition = { id: number; code: string };
type Season = { id: number; label: string; is_current: boolean };
type Team = { id: number; name: string };
type Standing = { id: number; computed_at: string; as_of_matchday: number | null; position: number; team_id: number; played: number | null; won: number | null; drawn: number | null; lost: number | null; goals_for: number | null; goals_against: number | null; points: number | null; form: string | null };

export const dynamic = "force-dynamic";

async function request<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Backend request failed (${response.status}).`);
  return response.json() as Promise<T>;
}

async function getStandings() {
  const api = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
  const competitions = await request<Competition[]>(`${api}/api/v1/competitions`);
  const sp1 = competitions.find((competition) => competition.code === "SP1");
  if (!sp1) throw new Error("SP1 competition is not available from the backend.");
  const seasons = await request<Season[]>(`${api}/api/v1/competitions/${sp1.id}/seasons`);
  const season = seasons.find((item) => item.is_current);
  if (!season) throw new Error("The backend has no current SP1 season.");
  const [standings, teams] = await Promise.all([request<Standing[]>(`${api}/api/v1/standings?season_id=${season.id}`), request<Team[]>(`${api}/api/v1/teams`)]);
  return { season, standings, teams: new Map(teams.map((team) => [team.id, team])) };
}

export default async function StandingsPage() {
  try {
    const data = await getStandings();
    return <div className="w-full space-y-8"><PageHeader season={data.season.label} />{data.standings.length ? <StandingsTable data={data} /> : <EmptyState season={data.season.label} />}</div>;
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : "Unable to load standings.";
    return <div className="w-full space-y-8"><PageHeader /><Card className="border-rose-400/20"><CardContent className="flex gap-3 p-5 text-sm text-rose-200"><AlertCircle className="mt-0.5 size-4 shrink-0" /><div><p className="font-medium">Standings are unavailable</p><p className="mt-1 text-rose-200/75">{detail} Check that the xgoal backend is running and try again.</p></div></CardContent></Card></div>;
  }
}

function PageHeader({ season }: { season?: string }) { return <section className="flex flex-wrap items-end justify-between gap-4"><div className="space-y-2"><Badge variant="muted">Competition table</Badge><h1 className="text-3xl font-semibold tracking-tight">La Liga standings</h1><p className="text-sm text-muted-foreground">Official positions from the latest snapshot returned by xgoal.</p></div>{season ? <Badge variant="success">{season} current season</Badge> : null}</section>; }
function StandingsTable({ data }: { data: Awaited<ReturnType<typeof getStandings>> }) { const latest = data.standings[0]; return <Card className="overflow-hidden"><CardHeader className="border-b border-border"><CardTitle>Table after matchday {latest.as_of_matchday ?? "—"}</CardTitle><p className="text-sm text-muted-foreground">Snapshot generated {formatDateTime(latest.computed_at)} · ordered by the backend tiebreaker result.</p></CardHeader><CardContent className="overflow-x-auto p-0"><table className="w-full min-w-[650px] text-sm"><thead className="bg-muted/50 text-left text-xs uppercase tracking-wide text-muted-foreground"><tr><th className="px-5 py-3 font-medium">#</th><th className="px-5 py-3 font-medium">Club</th><th className="px-3 py-3 text-center font-medium">P</th><th className="px-3 py-3 text-center font-medium">W</th><th className="px-3 py-3 text-center font-medium">D</th><th className="px-3 py-3 text-center font-medium">L</th><th className="px-3 py-3 text-center font-medium">GD</th><th className="px-5 py-3 text-right font-medium">Pts</th><th className="px-5 py-3 text-right font-medium">Form</th></tr></thead><tbody>{data.standings.map((row) => { const team = data.teams.get(row.team_id); const goalDifference = (row.goals_for ?? 0) - (row.goals_against ?? 0); return <tr key={row.id} className="border-t border-border/80"><td className="px-5 py-3 font-medium text-muted-foreground">{row.position}</td><td className="px-5 py-3 font-medium">{team?.name ?? `Team ${row.team_id}`}</td><td className="px-3 py-3 text-center text-muted-foreground">{row.played ?? "—"}</td><td className="px-3 py-3 text-center text-muted-foreground">{row.won ?? "—"}</td><td className="px-3 py-3 text-center text-muted-foreground">{row.drawn ?? "—"}</td><td className="px-3 py-3 text-center text-muted-foreground">{row.lost ?? "—"}</td><td className="px-3 py-3 text-center text-muted-foreground">{goalDifference > 0 ? `+${goalDifference}` : goalDifference}</td><td className="px-5 py-3 text-right font-semibold">{row.points ?? "—"}</td><td className="px-5 py-3 text-right font-mono text-xs tracking-[0.15em] text-muted-foreground">{row.form ?? "—"}</td></tr>; })}</tbody></table></CardContent></Card>; }
function EmptyState({ season }: { season: string }) { return <Card><CardContent className="flex flex-col items-center gap-3 px-5 py-14 text-center"><span className="grid size-11 place-items-center rounded-full bg-muted text-muted-foreground"><Trophy className="size-5" /></span><div><p className="font-medium">No standings snapshot yet</p><p className="mt-1 max-w-md text-sm text-muted-foreground">The backend identifies {season} as the current SP1 season, but has not returned a standings snapshot for it.</p></div></CardContent></Card>; }
