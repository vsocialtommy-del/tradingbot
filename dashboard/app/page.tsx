"use client";

import { useEffect, useState } from "react";

import type { Signal } from "@/lib/supabase";

interface ApiResponse {
  buy: Signal | null;
  sell: Signal | null;
  fetchedAt: string;
}

const POLL_INTERVAL_MS = 10_000;

export default function Page() {
  const [data, setData] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    let cancelled = false;

    const fetchOnce = async () => {
      try {
        const res = await fetch("/api/signals", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = (await res.json()) as ApiResponse;
        if (!cancelled) {
          setData(json);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoading(false);
        }
      }
    };

    fetchOnce();
    const fetchTimer = setInterval(fetchOnce, POLL_INTERVAL_MS);
    // Tick a "now" state every second so the "X seconds ago" label
    // counts smoothly rather than jumping every poll.
    const nowTimer = setInterval(() => setNow(Date.now()), 1000);

    return () => {
      cancelled = true;
      clearInterval(fetchTimer);
      clearInterval(nowTimer);
    };
  }, []);

  const fetchedAtMs = data ? new Date(data.fetchedAt).getTime() : null;
  const secondsAgo =
    fetchedAtMs !== null ? Math.max(0, Math.floor((now - fetchedAtMs) / 1000)) : null;

  return (
    <main className="mx-auto max-w-5xl p-4 sm:p-8">
      <header className="mb-6 sm:mb-10">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">
          Trading Signals
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Live BUY / SELL zones from the XAUUSD bot. Auto-refreshes every 10 s.
        </p>
      </header>

      {loading && <p className="text-slate-500">Loading…</p>}

      {error && (
        <div className="rounded border border-red-300 bg-red-50 p-4 text-red-800 mb-6">
          <strong>Error:</strong> {error}
        </div>
      )}

      {data && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
            <SignalCard signal={data.buy} side="BUY" />
            <SignalCard signal={data.sell} side="SELL" />
          </div>
          <Footer
            currentPrice={data.buy?.currentPrice ?? data.sell?.currentPrice ?? null}
            secondsAgo={secondsAgo}
          />
        </>
      )}
    </main>
  );
}

interface SignalCardProps {
  signal: Signal | null;
  side: "BUY" | "SELL";
}

function SignalCard({ signal, side }: SignalCardProps) {
  const isBuy = side === "BUY";
  const palette = isBuy
    ? {
        border: "border-emerald-300",
        bg: "bg-emerald-50",
        text: "text-emerald-700",
        chip: "bg-emerald-100 text-emerald-800",
        emoji: "🟢",
      }
    : {
        border: "border-rose-300",
        bg: "bg-rose-50",
        text: "text-rose-700",
        chip: "bg-rose-100 text-rose-800",
        emoji: "🔴",
      };

  return (
    <section
      className={`rounded-xl border-2 ${palette.border} ${palette.bg} p-5 sm:p-6 shadow-sm`}
    >
      <h2 className={`text-xl font-semibold ${palette.text}`}>
        {palette.emoji} {side}
      </h2>

      {signal === null ? (
        <p className="text-slate-500 mt-4">No active signal.</p>
      ) : (
        <>
          <p className="mt-3 text-3xl sm:text-4xl font-bold tabular-nums">
            {fmtPrice(signal.zoneBottom)} – {fmtPrice(signal.zoneTop)}
          </p>

          <div className="mt-2 flex items-center gap-2 flex-wrap">
            {signal.patternType && (
              <span className="text-xs uppercase tracking-wider text-slate-500">
                {signal.patternType}
              </span>
            )}
            {signal.zoneStatus && (
              <span
                className={`text-xs uppercase tracking-wider px-2 py-0.5 rounded ${palette.chip}`}
              >
                {signal.zoneStatus}
              </span>
            )}
          </div>

          <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-2 text-base sm:text-lg">
            <Level label="Entry" value={signal.entryPrice} />
            <Level label="SL" value={signal.slPrice} />
            <Level label="TP1" value={signal.tp1Price} />
            <Level label="TP2" value={signal.tp2Price} />
            <Level label="TP3" value={signal.tp3Price} />
          </dl>

          {signal.distanceDollars !== null && (
            <p className="mt-4 text-sm text-slate-600">
              {fmtDollars(signal.distanceDollars)} away from entry
            </p>
          )}
        </>
      )}
    </section>
  );
}

function Level({ label, value }: { label: string; value: number | null }) {
  return (
    <>
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right tabular-nums font-medium">
        {value === null ? "—" : fmtPrice(value)}
      </dd>
    </>
  );
}

function Footer({
  currentPrice,
  secondsAgo,
}: {
  currentPrice: number | null;
  secondsAgo: number | null;
}) {
  return (
    <footer className="mt-8 text-sm text-slate-500 flex flex-col sm:flex-row sm:justify-between gap-1">
      <span>
        Current price:{" "}
        <span className="text-slate-700 tabular-nums font-medium">
          {currentPrice === null ? "—" : `$${fmtPrice(currentPrice)}`}
        </span>
      </span>
      <span>
        Updated{" "}
        {secondsAgo === null
          ? "—"
          : secondsAgo < 1
          ? "just now"
          : `${secondsAgo}s ago`}
      </span>
    </footer>
  );
}

function fmtPrice(n: number): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function fmtDollars(n: number): string {
  const sign = n >= 0 ? "+" : "−";
  return `${sign}$${Math.abs(n).toFixed(2)}`;
}
