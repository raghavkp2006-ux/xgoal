import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "xgoal — La Liga Analytics",
  description:
    "La Liga analytics platform with Dixon-Coles model predictions, season simulator, and live standings.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">
        <header className="border-b border-gray-800 px-6 py-4">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <a href="/" className="text-xl font-bold text-primary-400">
              xgoal
            </a>
            <nav className="flex gap-6 text-sm text-gray-400">
              <a href="/" className="hover:text-white transition-colors">
                Dashboard
              </a>
              <a href="/fixtures" className="hover:text-white transition-colors">
                Fixtures
              </a>
              <a href="/standings" className="hover:text-white transition-colors">
                Standings
              </a>
              <a href="/simulation" className="hover:text-white transition-colors">
                Simulator
              </a>
            </nav>
          </div>
        </header>
        <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
          {children}
        </main>
        <footer className="border-t border-gray-800 px-6 py-4 text-center text-xs text-gray-600">
          Data updated <span id="data-freshness">—</span> &middot; Powered by
          football-data.co.uk & API-Football
        </footer>
      </body>
    </html>
  );
}