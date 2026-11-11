import { NextResponse, type NextRequest } from "next/server";

import { db, fail, isBot, needLogin } from "@/lib/db";

export const dynamic = "force-dynamic";

const COMMANDS = ["update", "restart", "start", "stop"];

// Queues a button press for the laptop supervisor (picked up within ~5 s).
export async function POST(req: NextRequest) {
  const denied = await needLogin();
  if (denied) return denied;
  const body = await req.json().catch(() => ({}));
  if (!isBot(body.bot)) return fail("Unknown bot");
  if (!COMMANDS.includes(body.command)) return fail("Unknown command");
  const sb = db(body.bot);
  if (!sb) return fail(`${body.bot} bot isn't set up in Vercel yet`, 500);

  // A double tap shouldn't queue it twice.
  const since = new Date(Date.now() - 10 * 60_000).toISOString();
  const { data: open } = await sb.from("bot_commands").select("id")
    .eq("bot", body.bot).eq("command", body.command).in("status", ["pending", "running"])
    .gte("created_at", since).limit(1);
  if (open?.length) return NextResponse.json({ ok: true, id: open[0].id, already: true });

  const { data, error } = await sb.from("bot_commands")
    .insert({ bot: body.bot, command: body.command }).select("id").single();
  if (error) return fail(`Couldn't send: ${error.message}`, 500);
  return NextResponse.json({ ok: true, id: data.id });
}
