// Signed login cookie. Works in middleware (edge) and API routes (node):
// only Web Crypto is used.
//
// Cookie = "<expiry ms>.<HMAC-SHA256(SESSION_SECRET, v1|expiry|PIN)>" (full).
// View-only (27 Nov 2026, VIEW_PIN): "<expiry ms>.v.<HMAC(v2|expiry|view|VIEW_PIN)>".
// Putting the PIN in the signed text means changing DASHBOARD_PIN in
// Vercel logs every phone out. The cookie only lives while the app is
// open (no expiry date) and is never valid for more than 12 hours.

export const COOKIE = "xb_session";
export const SESSION_HOURS = 12;

const enc = new TextEncoder();

function b64url(buf: ArrayBuffer): string {
  let s = "";
  for (const b of new Uint8Array(buf)) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function authConfigured(): boolean {
  const pin = process.env.DASHBOARD_PIN ?? "";
  const secret = process.env.SESSION_SECRET ?? "";
  return /^\d{4}$/.test(pin) && secret.length >= 16;
}

export type Role = "full" | "view";

/** The watch-only PIN (VIEW_PIN in Vercel), "" if not set or same as the main PIN. */
export function viewPin(): string {
  const v = process.env.VIEW_PIN ?? "";
  return /^\d{4}$/.test(v) && v !== (process.env.DASHBOARD_PIN ?? "") ? v : "";
}

async function sign(exp: number, role: Role = "full"): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(process.env.SESSION_SECRET ?? ""),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const text = role === "full"
    ? `v1|${exp}|${process.env.DASHBOARD_PIN ?? ""}`
    : `v2|${exp}|view|${viewPin()}`;
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(text));
  return b64url(sig);
}

export async function makeToken(now = Date.now(), role: Role = "full"): Promise<{ token: string; exp: number }> {
  const exp = now + SESSION_HOURS * 3600 * 1000;
  const sig = await sign(exp, role);
  return { token: role === "full" ? `${exp}.${sig}` : `${exp}.v.${sig}`, exp };
}

function sameText(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/** "full", "view" (watch only) or null (not logged in). */
export async function tokenRole(token: string | undefined | null): Promise<Role | null> {
  if (!token || !authConfigured()) return null;
  const parts = token.split(".");
  const exp = Number(parts[0]);
  if (!Number.isFinite(exp) || exp < Date.now()) return null;
  if (parts.length === 2) return sameText(parts[1], await sign(exp, "full")) ? "full" : null;
  if (parts.length === 3 && parts[1] === "v" && viewPin()) {
    return sameText(parts[2], await sign(exp, "view")) ? "view" : null;
  }
  return null;
}

export async function validToken(token: string | undefined | null): Promise<boolean> {
  return (await tokenRole(token)) !== null;
}
