import { Activity, Database, Radio, ShieldCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type HealthResponse = {
  status: string;
  db: boolean;
  sources: { name: string; age_seconds: number | null; last_error: string | null }[];
};

export const dynamic = "force-dynamic";

async function getHealth(): Promise<{ health: HealthResponse | null; error: string | null }> {
  try {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const response = await fetch(`${backendUrl}/health`, { cache: "no-store" });
    if (!response.ok) return { health: null, error: `Health check failed (${response.status})` };
    return { health: (await response.json()) as HealthResponse, error: null };
  } catch (error) {
    return { health: null, error: error instanceof Error ? error.message : "Unknown connection error" };
  }
}

export default async function Home() {
  const { health, error } = await getHealth();
  const status = health?.db ? "Operational" : health ? "Degraded" : "Unreachable";
  const variant = health?.db ? "success" : health ? "warning" : "danger";

  return (
    <div className="w-full space-y-8">
      <section className="max-w-3xl space-y-4">
        <Badge variant="muted" className="gap-1.5"><Radio className="size-3" /> System overview</Badge>
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">La Liga, modelled clearly.</h1>
          <p className="max-w-2xl text-base leading-7 text-muted-foreground">A focused workspace for Dixon-Coles forecasts, competition context, and model health.</p>
        </div>
      </section>
      <section className="grid gap-4 sm:grid-cols-3" aria-label="Product capabilities">
        <Capability icon={Activity} title="Forecasts" detail="Scoreline probabilities" />
        <Capability icon={Database} title="Competition data" detail="Fixtures and standings" />
        <Capability icon={ShieldCheck} title="Model quality" detail="Measured, not assumed" />
      </section>
      <Card className="max-w-3xl">
        <CardHeader className="flex-row items-center justify-between">
          <div><CardTitle>Backend connection</CardTitle><p className="mt-1 text-sm text-muted-foreground">Live response from the xgoal API.</p></div>
          <Badge variant={variant}>{status}</Badge>
        </CardHeader>
        <CardContent>
          {error ? <div role="alert" className="rounded-lg border border-rose-400/20 bg-rose-400/10 p-4 text-sm text-rose-200">The backend could not be reached. {error}</div> : null}
          {health ? <div className="space-y-5"><div className="flex items-center gap-3 text-sm"><span className={`size-2.5 rounded-full ${health.db ? "bg-emerald-400" : "bg-amber-300"}`} /><span className="font-medium">Database {health.db ? "connected" : "unavailable"}</span><span className="text-muted-foreground">API status: {health.status}</span></div><div className="grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-2">{health.sources.length ? health.sources.map((source) => <div key={source.name} className="bg-card px-4 py-3 text-sm"><p className="font-medium">{source.name}</p><p className="mt-1 text-muted-foreground">{source.age_seconds !== null ? `Updated ${Math.round(source.age_seconds / 60)}m ago` : "No successful refresh"}</p>{source.last_error ? <p className="mt-2 text-xs text-rose-300">Latest error: {source.last_error}</p> : null}</div>) : <div className="bg-card px-4 py-3 text-sm text-muted-foreground">No source freshness records yet.</div>}</div></div> : null}
        </CardContent>
      </Card>
    </div>
  );
}

function Capability({ icon: Icon, title, detail }: { icon: typeof Activity; title: string; detail: string }) {
  return <Card className="bg-card/75"><CardContent className="flex items-center gap-3 p-4"><span className="grid size-9 place-items-center rounded-lg bg-primary/10 text-primary"><Icon className="size-4" /></span><div><p className="text-sm font-medium">{title}</p><p className="text-xs text-muted-foreground">{detail}</p></div></CardContent></Card>;
}
