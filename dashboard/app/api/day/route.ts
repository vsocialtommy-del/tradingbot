import { NextResponse, type NextRequest } from "next/server";

import { db, fail, isBot, needLogin } from "@/lib/db";

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

  const { data, error } = await sb.from("closed_trades")
    .select("position_id, setup_key, direction, volume, open_time, close_time, open_price, close_price, profit")
    .eq("bot", bot).eq("trading_day", date).order("open_time");
  if (error) return fail(error.message, 500);

  const trades = new Map<string, {
    direction: string; open: string; close: string; profit: number;
    layers: { volume: number; open: number | null; close: number | null; profit: number }[];
  }>();
  for (const r of (data ?? []) as Row[]) {
    const t = trades.get(r.setup_key) ?? {
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
  const list = [...trades.values()]
    .map((t) => ({ ...t, profit: Math.round(t.profit * 100) / 100 }))
    .sort((a, b) => a.close.localeCompare(b.close));
  return NextResponse.json({ bot, date, trades: list }, { headers: { "Cache-Control": "no-store" } });
}
