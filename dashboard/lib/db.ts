// Server-only Supabase access, one project per bot. The service role
// keys never reach the browser: every read and write goes through the
// API routes, which check the login cookie first.

import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { COOKIE, validToken } from "./session";

export const BOTS = ["M5", "M1"] as const;
export type Bot = (typeof BOTS)[number];

export function isBot(v: unknown): v is Bot {
  return v === "M5" || v === "M1";
}

const clients = new Map<Bot, SupabaseClient>();

export function db(bot: Bot): SupabaseClient | null {
  const url = process.env[`${bot}_SUPABASE_URL`];
  const key = process.env[`${bot}_SUPABASE_SERVICE_ROLE_KEY`];
  if (!url || !key) return null;
  let c = clients.get(bot);
  if (!c) {
    c = createClient(url, key, { auth: { persistSession: false } });
    clients.set(bot, c);
  }
  return c;
}

/** null if logged in, otherwise the 401 to return. */
export async function needLogin(): Promise<NextResponse | null> {
  const ok = await validToken(cookies().get(COOKIE)?.value);
  return ok ? null : NextResponse.json({ error: "Log in again" }, { status: 401 });
}

export function fail(message: string, status = 400): NextResponse {
  return NextResponse.json({ error: message }, { status });
}

/** Same rules as bot/profiles.py parse_risk / parse_max_losses. */
export function checkRisk(raw: unknown): string | null | Error {
  if (raw === null || raw === undefined || raw === "") return null;
  const t = String(raw).trim().replace(/£/g, "").replace(/\s/g, "").toLowerCase();
  if (t.endsWith("%")) {
    const pct = Number(t.slice(0, -1));
    if (!Number.isFinite(pct) || pct <= 0 || pct > 20) {
      return new Error("Risk % must be above 0 and at most 20%");
    }
    return `${pct}%`;
  }
  const amount = Number(t);
  if (!Number.isFinite(amount) || amount <= 0 || amount > 10000) {
    return new Error("Risk must be a % (like 1%) or an amount (like 100)");
  }
  return String(amount);
}

export function checkMaxLosses(raw: unknown): number | null | Error {
  if (raw === null || raw === undefined || raw === "") return null;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1 || n > 50) {
    return new Error("Max losses must be a whole number from 1 to 50");
  }
  return n;
}
