import { NextResponse } from "next/server";

import { BOTS, db, needLogin, type Bot } from "@/lib/db";

export const dynamic = "force-dynamic";

async function one(bot: Bot) {
  const sb = db(bot);
  if (!sb) return { bot, configured: false };
  const [status, config, commands] = await Promise.all([
    sb.from("bot_status").select("*").eq("bot", bot).maybeSingle(),
    sb.from("bot_config").select("key, value").in("key", ["kill_switch", "risk", "max_losses", "trend_filter", "account_pubkey"]),
    sb.from("bot_commands").select("id, command, status, created_at, finished_at, output")
      .eq("bot", bot).order("created_at", { ascending: false }).limit(6),
  ]);
  const cfg: Record<string, unknown> = {};
  for (const r of config.data ?? []) cfg[r.key] = r.value;
  const err = status.error ?? config.error ?? commands.error;
  return {
    bot,
    configured: true,
    error: err ? (/does not exist|schema cache|PGRST205/i.test(err.message)
      ? "Dashboard tables missing: run SQL 016 in this bot's Supabase"
      : err.message) : null,
    status: status.data ?? null,
    killSwitch: cfg.kill_switch === true,
    risk: (cfg.risk as string | null) || null,       // "" = bot default
    maxLosses: (cfg.max_losses as number | null) || null,
    trendFilter: (cfg.trend_filter as string | null) || null,   // null = bot default (off)
    loginKey: (cfg.account_pubkey as string | null) || null,    // public half; private stays on the laptop
    commands: commands.data ?? [],
  };
}

export async function GET() {
  const denied = await needLogin();
  if (denied) return denied;
  const bots = await Promise.all(BOTS.map(one));
  return NextResponse.json(
    { bots, serverTime: new Date().toISOString() },
    { headers: { "Cache-Control": "no-store" } },
  );
}
