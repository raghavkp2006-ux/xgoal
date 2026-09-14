"use client";

import { useEffect, useState } from "react";

type HealthResponse = {
  status: string;
  db: boolean;
  sources: { name: string; age_seconds: number | null; last_error: string | null }[];
};

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${backendUrl}/health`)
      .then((res) => res.json())
      .then((data) => {
        setHealth(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-2">xgoal</h1>
        <p className="text-gray-400">
          La Liga analytics powered by a Dixon-Coles scoreline model.
          Predictions, standings, and a full season simulator.
        </p>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-lg border border-gray-800 p-6 bg-gray-900/50">
          <h2 className="text-lg font-semibold mb-1">Model Predictions</h2>
          <p className="text-sm text-gray-400">
            Dixon-Coles attack/defence ratings with time decay and low-score
            correction.
          </p>
        </div>
        <div className="rounded-lg border border-gray-800 p-6 bg-gray-900/50">
          <h2 className="text-lg font-semibold mb-1">Live Standings</h2>
          <p className="text-sm text-gray-400">
            La Liga tiebreakers applied correctly — head-to-head, goal
            difference, goals scored.
          </p>
        </div>
        <div className="rounded-lg border border-gray-800 p-6 bg-gray-900/50">
          <h2 className="text-lg font-semibold mb-1">Season Simulator</h2>
          <p className="text-sm text-gray-400">
            10,000 Monte Carlo simulations of remaining fixtures with parameter
            uncertainty.
          </p>
        </div>
      </section>

      <section className="rounded-lg border border-gray-800 p-6 bg-gray-900/50">
        <h2 className="text-lg font-semibold mb-3">System Status</h2>
        {loading && (
          <div className="animate-pulse h-4 w-48 bg-gray-800 rounded" />
        )}
        {error && (
          <div className="text-red-400 text-sm">
            Backend unreachable: {error}
          </div>
        )}
        {health && (
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  health.db ? "bg-green-500" : "bg-red-500"
                }`}
              />
              <span>Database: {health.db ? "connected" : "disconnected"}</span>
            </div>
            {health.sources.map((source) => (
              <div key={source.name} className="flex items-center gap-2 ml-4">
                <span className="text-gray-500">{source.name}:</span>
                <span>
                  {source.age_seconds !== null
                    ? `${Math.round(source.age_seconds / 60)}m ago`
                    : "never"}
                </span>
                {source.last_error && (
                  <span className="text-red-400 text-xs">
                    ({source.last_error})
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}