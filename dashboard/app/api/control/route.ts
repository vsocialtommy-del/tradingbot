import { NextResponse, type NextRequest } from "next/server";

import { checkMaxLosses, checkRisk, checkTrendFilter, db, fail, isBot, needLogin } from "@/lib/db";

export const dynamic = "force-dynamic";

// { bot, trading: true|false }                 the on/off switch
// { bot, risk: "1%"|"100"|null, maxLosses: 2|null }  settings (null = bot default)
// { bot, trendFilter: "off"|"own"|"higher" }        trend filter (12 Nov 2026)
export async function POST(req: NextRequest) {
  const denied = await needLogin();
  if (denied) return denied;
  const body = await req.json().catch(() => ({}));
  if (!isBot(body.bot)) return fail("Unknown bot");
  const sb = db(body.bot);
  if (!sb) return fail(`${body.bot} bot isn't set up in Vercel yet`, 500);

  const rows: { key: string; value: unknown; updated_by: string }[] = [];
  if (typeof body.trading === "boolean") {
    rows.push({ key: "kill_switch", value: !body.trading, updated_by: "dashboard" });
  }
  if ("risk" in body || "maxLosses" in body) {
    const risk = checkRisk(body.risk);
    if (risk instanceof Error) return fail(risk.message);
    const losses = checkMaxLosses(body.maxLosses);
    if (losses instanceof Error) return fail(losses.message);
    // "Bot default" is saved as "" (not null): bot_config.value can't be
    // empty, and the bot treats "" the same as no setting.
    rows.push({ key: "risk", value: risk ?? "", updated_by: "dashboard" });
    rows.push({ key: "max_losses", value: losses ?? "", updated_by: "dashboard" });
  }
  if ("trendFilter" in body) {
    const trend = checkTrendFilter(body.trendFilter);
    if (trend instanceof Error) return fail(trend.message);
    rows.push({ key: "trend_filter", value: trend, updated_by: "dashboard" });
  }
  if (!rows.length) return fail("Nothing to change");
  const { error } = await sb.from("bot_config").upsert(rows, { onConflict: "key" });
  if (error) return fail(`Couldn't save: ${error.message}`, 500);
  return NextResponse.json({ ok: true });
}
