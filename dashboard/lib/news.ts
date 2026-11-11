// USD news for the dashboard: Forex Factory's weekly calendar (the same
// feed the bot's news filter uses) and FinancialJuice headlines.
// Server only. Both are cached in memory so the dashboard's polling never
// hammers either site (Forex Factory blocks feeds pulled too often).

export type CalEvent = {
  time: string;            // ISO, UTC
  title: string;
  impact: "High" | "Medium" | "Low" | "Holiday";
  forecast: string | null;
  previous: string | null;
};
export type Headline = { id: string; title: string; summary: string | null; time: string; link: string | null };

const FF_URL = process.env.NEWS_FF_URL || "https://nfs.faireconomy.media/ff_calendar_thisweek.json"; // env only for local tests
const FJ_URL = process.env.NEWS_FJ_URL || "https://www.financialjuice.com/feed.ashx?xy=rss";
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36";

type Cache<T> = { at: number; data: T; error: string | null };
let cal: Cache<CalEvent[]> | null = null;
let heads: Cache<Headline[]> | null = null;

async function getText(url: string, accept: string): Promise<string> {
  const res = await fetch(url, {
    headers: { "User-Agent": UA, Accept: accept },
    cache: "no-store",
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) throw new Error(`${new URL(url).hostname} said ${res.status}`);
  return res.text();
}

export async function usdCalendar(): Promise<Cache<CalEvent[]>> {
  if (cal && Date.now() - cal.at < 15 * 60_000) return cal;
  try {
    const raw = JSON.parse(await getText(FF_URL, "application/json")) as Record<string, string>[];
    const data = raw
      .filter((r) => String(r.country).toUpperCase() === "USD")
      .map((r) => ({
        time: new Date(r.date).toISOString(),
        title: String(r.title ?? ""),
        impact: (["High", "Medium", "Low", "Holiday"].includes(r.impact) ? r.impact : "Low") as CalEvent["impact"],
        forecast: r.forecast ? String(r.forecast) : null,
        previous: r.previous ? String(r.previous) : null,
      }))
      .sort((a, b) => a.time.localeCompare(b.time));
    cal = { at: Date.now(), data, error: null };
  } catch (e) {
    // Keep showing the last good copy; try again in 5 minutes.
    cal = { at: Date.now() - 10 * 60_000, data: cal?.data ?? [], error: (e as Error).message };
  }
  return cal;
}

// Headlines that can move USD / gold. FinancialJuice covers everything
// (EUR, JPY, crypto...), so keep only these. Whole words only.
const USD_TERMS = [
  "USD", "dollar", "Fed", "FOMC", "Powell", "Trump", "White House", "Treasury", "Bessent", "tariffs?",
  "sanctions?", "Iran", "Israel", "Hormuz", "Gaza", "Russia", "Ukraine", "Putin", "China", "Xi",
  "oil", "crude", "OPEC", "gold", "CPI", "PPI", "PCE", "NFP", "payrolls?", "jobless", "unemployment", "GDP",
  "ISM", "retail sales", "yields?", "Senate", "Congress", "shutdown", "debt ceiling", "Waller",
  "Williams", "Goolsbee", "Hammack", "Barr", "Bowman", "Jefferson", "Logan", "Musalem", "Schmid", "Daly",
  "Kashkari", "Bostic", "Harker", "Miran", "war", "strikes?", "attacks?", "missiles?", "ceasefire",
];
// Plain string, not a template: the Next build compiler mangles "\\b" inside templates.
const USD_RE = new RegExp("\\b(" + USD_TERMS.join("|") + ")\\b", "i");
const US_RE = /\b(US|U\.S\.?)\b/; // capital US only (not "us")

function tag(xml: string, name: string): string | null {
  const m = xml.match(new RegExp("<" + name + "[^>]*>([\\s\\S]*?)</" + name + ">", "i"));
  if (!m) return null;
  return m[1].replace(/^<!\[CDATA\[|\]\]>$/g, "").trim();
}

function clean(s: string): string {
  return s
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ").trim();
}

export function parseFeed(xml: string): Headline[] {
  const out: Headline[] = [];
  for (const m of xml.matchAll(/<item>([\s\S]*?)<\/item>/gi)) {
    const item = m[1];
    const title = clean(tag(item, "title") ?? "").replace(/^FinancialJuice:\s*/i, "");
    if (!title) continue;
    const pub = tag(item, "pubDate");
    const t = pub ? new Date(pub) : null;
    if (!t || Number.isNaN(t.getTime())) continue;
    const desc = clean(tag(item, "description") ?? "");
    out.push({
      id: tag(item, "guid") ?? `${t.getTime()}|${title}`,
      title,
      summary: desc && desc !== title ? desc.slice(0, 280) : null,
      time: t.toISOString(),
      link: tag(item, "link"),
    });
  }
  return out;
}

export function usdOnly(list: Headline[]): Headline[] {
  return list.filter((h) => USD_RE.test(h.title) || US_RE.test(h.title));
}

export async function usdHeadlines(): Promise<Cache<Headline[]>> {
  if (heads && Date.now() - heads.at < 60_000) return heads;
  try {
    const all = parseFeed(await getText(FJ_URL, "application/rss+xml, application/xml, text/xml"));
    const since = Date.now() - 12 * 3600_000;
    const data = usdOnly(all).filter((h) => Date.parse(h.time) >= since)
      .sort((a, b) => b.time.localeCompare(a.time)).slice(0, 15);
    heads = { at: Date.now(), data, error: null };
  } catch (e) {
    heads = { at: Date.now() - 30_000, data: heads?.data ?? [], error: (e as Error).message };
  }
  return heads;
}
