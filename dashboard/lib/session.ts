// Signed login cookie. Works in middleware (edge) and API routes (node):
// only Web Crypto is used.
//
// Cookie = "<expiry ms>.<HMAC-SHA256(SESSION_SECRET, v1|expiry|PIN)>".
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

async function sign(exp: number): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(process.env.SESSION_SECRET ?? ""),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign(
    "HMAC",
    key,
    enc.encode(`v1|${exp}|${process.env.DASHBOARD_PIN ?? ""}`),
  );
  return b64url(sig);
}

export async function makeToken(now = Date.now()): Promise<{ token: string; exp: number }> {
  const exp = now + SESSION_HOURS * 3600 * 1000;
  return { token: `${exp}.${await sign(exp)}`, exp };
}

function sameText(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export async function validToken(token: string | undefined | null): Promise<boolean> {
  if (!token || !authConfigured()) return false;
  const dot = token.indexOf(".");
  if (dot < 1) return false;
  const exp = Number(token.slice(0, dot));
  if (!Number.isFinite(exp) || exp < Date.now()) return false;
  return sameText(token.slice(dot + 1), await sign(exp));
}
