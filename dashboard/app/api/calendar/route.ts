import { NextResponse, type NextRequest } from "next/server";

import { db, fail, isBot, needLogin } from "@/lib/db";

export const dynamic = "force-dynamic";

type Row = { trading_day: string; setup_key: string; profit: number | string };

// ?bot=M5&month=2026-11 -> P/L per trading day. A "trade" is one setup
// (layer 1 + layer 2 together), the same way the bot counts losses.
export async function GET(req: NextRequest) {
  const denied = await needLogin();
  if (denied) return denied;
  const bot = req.nextUrl.searchParams.get("bot");
  const month = req.nextUrl.searchParams.get("month") ?? "";
  if (!isBot(bot)) return fail("Unknown bot");
  if (!/^\d{4}-\d{2}$/.test(month)) return fail("month must be YYYY-MM");
  const sb = db(bot);
  if (!sb) return fail(`${bot} bot isn't set up in Vercel yet`, 500);

  const [y, m] = month.split("-").map(Number);
  const first = `${month}-01`;
  const last = new Date(Date.UTC(y, m, 0)).toISOString().slice(0, 10);
  const { data, error } = await sb.from("closed_trades")
    .select("trading_day, setup_key, profit")
    .eq("bot", bot).gte("trading_day", first).lte("trading_day", last)
    .range(0, 9999);
  if (error) return fail(error.message, 500);

  const setups = new Map<string, { day: string; net: number }>();
  for (const r of (data ?? []) as Row[]) {
    const k = `${r.trading_day}|${r.setup_key}`;
    const s = setups.get(k) ?? { day: r.trading_day, net: 0 };
    s.net += Number(r.profit);
    setups.set(k, s);
  }
  const days: Record<string, { pnl: number; trades: number; wins: number; losses: number }> = {};
  for (const s of setups.values()) {
    const d = (days[s.day] ??= { pnl: 0, trades: 0, wins: 0, losses: 0 });
    d.pnl += s.net;
    d.trades += 1;
    if (s.net > 0) d.wins += 1;
    else if (s.net < 0) d.losses += 1;
  }
  let total = 0, trades = 0, wins = 0, losses = 0;
  for (const d of Object.values(days)) {
    d.pnl = Math.round(d.pnl * 100) / 100;
    total += d.pnl; trades += d.trades; wins += d.wins; losses += d.losses;
  }
  return NextResponse.json(
    { bot, month, days, total: Math.round(total * 100) / 100, trades, wins, losses },
    { headers: { "Cache-Control": "no-store" } },
  );
}
