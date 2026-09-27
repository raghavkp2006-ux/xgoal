import { Card, CardContent, CardHeader } from "@/components/ui/card";

export default function StandingsLoading() {
  return <div className="w-full space-y-8" aria-label="Loading standings"><div className="space-y-3"><div className="h-5 w-32 animate-pulse rounded bg-muted" /><div className="h-9 w-64 animate-pulse rounded bg-muted" /><div className="h-4 w-80 animate-pulse rounded bg-muted" /></div><Card><CardHeader><div className="h-5 w-44 animate-pulse rounded bg-muted" /><div className="h-4 w-72 animate-pulse rounded bg-muted" /></CardHeader><CardContent className="space-y-3">{Array.from({ length: 10 }, (_, index) => <div key={index} className="grid grid-cols-[36px_1fr_repeat(5,32px)_48px] gap-4"><div className="h-4 animate-pulse rounded bg-muted" /><div className="h-4 animate-pulse rounded bg-muted" />{Array.from({ length: 6 }, (_, cell) => <div key={cell} className="h-4 animate-pulse rounded bg-muted" />)}</div>)}</CardContent></Card></div>;
}
