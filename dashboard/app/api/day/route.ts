import { NextResponse, type NextRequest } from "next/server";

import { db, fail, isBot, needLogin, tradesQuery } from "@/lib/db";

export const dynamic = "force-dynamic";

type Row = {
  position_id: number; setup_key: string; direction: string; volume: number | string;
  open_time: string; close_time: string; open_price: number | string | null;
  close_price: number | string | null; profit: number | string;
};

// ?bot=M5&date=2026-11-11 -> that day's trades, layers grouped by setup.
export async function GET(req: NextRequest) {
  const denied = await needLogin();
  if (denied) return denied;
  const bot = req.nextUrl.searchParams.get("bot");
  const date = req.nextUrl.searchParams.get("date") ?? "";
  if (!isBot(bot)) return fail("Unknown bot");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return fail("date must be YYYY-MM-DD");
  const sb = db(bot);
  if (!sb) return fail(`${bot} bot isn't set up in Vercel yet`, 500);

  // Trades are shown on the day their last position closed, so also look
  // at the day before for layers that closed before the 22:00 reset.
  const prev = new Date(Date.parse(`${date}T12:00:00Z`) - 86_400_000).toISOString().slice(0, 10);
  const { data: raw, error } = await tradesQuery<Row & { trading_day: string }>(sb, bot,
    "position_id, setup_key, direction, volume, open_time, close_time, open_price, close_price, profit, trading_day",
    (q) => q.gte("trading_day", prev).lte("trading_day", date).order("open_time"));
  if (error) return fail(error.message, 500);
  const lastDay = new Map<string, string>();
  for (const r of raw ?? []) {
    const d = lastDay.get(r.setup_key);
    if (!d || r.trading_day > d) lastDay.set(r.setup_key, r.trading_day);
  }
  const data = (raw ?? []).filter((r) => lastDay.get(r.setup_key) === date);

  const trades = new Map<string, {
    key: string; chart: string | null;
    direction: string; open: string; close: string; profit: number;
    layers: { volume: number; open: number | null; close: number | null; profit: number }[];
  }>();
  for (const r of (data ?? []) as Row[]) {
    const t = trades.get(r.setup_key) ?? {
      key: r.setup_key, chart: null,
      direction: r.direction, open: r.open_time, close: r.close_time, profit: 0, layers: [],
    };
    if (r.open_time < t.open) t.open = r.open_time;
    if (r.close_time > t.close) t.close = r.close_time;
    t.profit += Number(r.profit);
    t.layers.push({
      volume: Number(r.volume),
      open: r.open_price === null ? null : Number(r.open_price),
      close: r.close_price === null ? null : Number(r.close_price),
      profit: Number(r.profit),
    });
    trades.set(r.setup_key, t);
  }
  // Trade photo journal (SQL 018): a link to each trade's picture, valid
  // for an hour. Nothing breaks if the table or bucket isn't there yet.
  if (trades.size) {
    const { data: charts } = await sb.from("trade_charts").select("setup_key, path")
      .eq("bot", bot).in("setup_key", [...trades.keys()]);
    for (const c of (charts ?? []) as { setup_key: string; path: string }[]) {
      const t = trades.get(c.setup_key);
      if (!t || t.chart) continue;
      const { data: signed } = await sb.storage.from("trade-charts").createSignedUrl(c.path, 3600);
      if (signed?.signedUrl) t.chart = signed.signedUrl;
    }
  }
  const list = [...trades.values()]
    .map((t) => ({ ...t, profit: Math.round(t.profit * 100) / 100 }))
    .sort((a, b) => a.close.localeCompare(b.close));
  return NextResponse.json({ bot, date, trades: list }, { headers: { "Cache-Control": "no-store" } });
}
