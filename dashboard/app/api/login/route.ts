import { timingSafeEqual } from "crypto";
import { NextResponse, type NextRequest } from "next/server";

import { db } from "@/lib/db";
import { COOKIE, authConfigured, makeToken } from "@/lib/session";

export const dynamic = "force-dynamic";

const MAX_TRIES = 5;
const LOCK_MINUTES = 15;

function samePin(a: string, b: string): boolean {
  const x = Buffer.from(a.padEnd(8, " "));
  const y = Buffer.from(b.padEnd(8, " "));
  return x.length === y.length && timingSafeEqual(x, y) && a.length === b.length;
}

// Wrong tries are counted in the M5 bot's Supabase (dashboard_auth), so
// the 15 minute lock holds across phones and Vercel instances.
export async function POST(req: NextRequest) {
  if (!authConfigured()) {
    return NextResponse.json(
      { error: "Not set up yet: add DASHBOARD_PIN (4 digits) and SESSION_SECRET in Vercel." },
      { status: 500 },
    );
  }
  const sb = db("M5");
  if (!sb) {
    return NextResponse.json(
      { error: "Not set up yet: add the M5 bot's Supabase URL and key in Vercel." },
      { status: 500 },
    );
  }
  let pin = "";
  try {
    pin = String((await req.json())?.pin ?? "");
  } catch {
    /* empty body */
  }

  const { data: row, error } = await sb
    .from("dashboard_auth").select("failed_attempts, locked_until").eq("id", 1).maybeSingle();
  if (error) {
    return NextResponse.json(
      { error: "Can't check the PIN right now (Supabase). Has SQL 016 been run?" },
      { status: 503 },
    );
  }
  const now = Date.now();
  const lockedUntil = row?.locked_until ? Date.parse(row.locked_until) : 0;
  if (lockedUntil > now) {
    return NextResponse.json({ error: "locked", lockedUntil: row!.locked_until }, { status: 429 });
  }

  if (/^\d{4}$/.test(pin) && samePin(pin, process.env.DASHBOARD_PIN!)) {
    await sb.from("dashboard_auth").upsert({ id: 1, failed_attempts: 0, locked_until: null });
    const { token } = await makeToken(now);
    const res = NextResponse.json({ ok: true });
    res.cookies.set(COOKIE, token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      // No maxAge / expires: a session cookie, gone when the app closes.
    });
    return res;
  }

  const tries = (row?.failed_attempts ?? 0) + 1;
  await new Promise((r) => setTimeout(r, 600)); // slow down guessing
  if (tries >= MAX_TRIES) {
    const until = new Date(now + LOCK_MINUTES * 60_000).toISOString();
    await sb.from("dashboard_auth").upsert({ id: 1, failed_attempts: 0, locked_until: until });
    return NextResponse.json({ error: "locked", lockedUntil: until }, { status: 429 });
  }
  await sb.from("dashboard_auth").upsert({ id: 1, failed_attempts: tries, locked_until: null });
  return NextResponse.json({ error: "wrong", triesLeft: MAX_TRIES - tries }, { status: 401 });
}
