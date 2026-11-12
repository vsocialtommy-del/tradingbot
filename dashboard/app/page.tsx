"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ------------------------------------------------------------------ types

type Zone = {
  direction: "BUY" | "SELL"; top: number; bottom: number; entry: number;
  sl: number | null; tp1: number | null; distance: number;
};
type Position = {
  ticket: number; direction: string; volume: number; entry: number;
  sl: number | null; tp: number | null; profit: number;
};
type StatusData = {
  bot?: string; timeframe?: string; version?: string | null; started_at?: string;
  trading_enabled?: boolean; bid?: number; ask?: number; spread?: number;
  settings?: { risk: string; risk_is_pct: boolean; max_losses: number; from_dashboard: boolean; runner: string };
  account?: { login?: number; currency?: string; balance: number; equity: number } | null;
  today?: {
    trading_day: string; realised: number; floating: number; trades: number;
    wins?: number; losses: number; blocked: boolean; blocked_reason: string | null;
  } | null;
  open_positions?: Position[];
  next_buy?: Zone | null; next_sell?: Zone | null;
  trend?: { state: "UPTREND" | "DOWNTREND" | "RANGE"; timeframe: string; rule: string;
    since: string | null; up_above: number | null; down_below: number | null } | null;
};
type Command = { id: number; command: string; status: string; created_at: string; finished_at: string | null; output: string | null };
type BotRow = {
  bot: "M5" | "M1"; configured: boolean; error?: string | null;
  status?: {
    updated_at: string; data: StatusData; supervisor_seen_at: string | null;
    process_running: boolean | null; supervisor_note?: string | null;
  } | null;
  killSwitch?: boolean; risk?: string | null; maxLosses?: number | null; commands?: Command[];
};
type DayCell = { pnl: number; trades: number; wins: number; losses: number };
type Month = { days: Record<string, DayCell>; total: number; trades: number; wins: number; losses: number };
type CalEvent = { time: string; title: string; impact: "High" | "Medium" | "Low" | "Holiday"; forecast: string | null; previous: string | null };
type Headline = { id: string; title: string; summary: string | null; time: string; link: string | null };
type News = { calendar: CalEvent[]; calendarError: string | null; headlines: Headline[]; headlinesError: string | null };

type DayTrade = {
  direction: string; open: string; close: string; profit: number;
  layers: { volume: number; open: number | null; close: number | null; profit: number }[];
};

// ------------------------------------------------------------------ helpers

const TZ = "Europe/London";
const sym = (cur?: string) => (cur === "USD" ? "$" : cur === "EUR" ? "€" : "£");
const money = (v: number, cur?: string, dp = 0) =>
  `${v < 0 ? "−" : ""}${sym(cur)}${Math.abs(v).toLocaleString("en-GB", { minimumFractionDigits: dp, maximumFractionDigits: dp })}`;
const signed = (v: number, cur?: string, dp = 2) => (v > 0 ? "+" : "") + money(v, cur, dp);
const short = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(Math.round(v))}`;
const tone = (v: number) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
const hhmm = (iso: string) => new Date(iso).toLocaleTimeString("en-GB", { timeZone: TZ, hour: "2-digit", minute: "2-digit" });
const dayMon = (iso: string) => new Date(iso).toLocaleDateString("en-GB", { timeZone: TZ, day: "numeric", month: "short" });
const londonDate = () => new Date().toLocaleDateString("en-CA", { timeZone: TZ });
const monthName = (ym: string) => {
  const [y, m] = ym.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 15)).toLocaleDateString("en-GB", { month: "long", year: "numeric", timeZone: "UTC" }).toUpperCase();
};
const shiftMonth = (ym: string, by: number) => {
  const [y, m] = ym.split("-").map(Number);
  const d = new Date(Date.UTC(y, m - 1 + by, 1));
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
};
const pts = (v: number) => (v >= 10 ? v.toFixed(0) : v.toFixed(1));

type State = { label: string; cls: "ok" | "warn" | "bad" | "" ; running: boolean };

function botState(b: BotRow, now: number): State {
  if (!b.configured) return { label: "Not set up in Vercel", cls: "", running: false };
  if (b.error) return { label: "Setup needed", cls: "bad", running: false };
  const s = b.status;
  if (!s) return { label: "Waiting for first update", cls: "", running: false };
  const upd = now - Date.parse(s.updated_at);
  const sup = s.supervisor_seen_at ? now - Date.parse(s.supervisor_seen_at) : Infinity;
  const fresh = upd < 60_000;
  if (!fresh && sup > 90_000) return { label: "Laptop offline", cls: "", running: false };
  if (s.process_running === false && sup < 90_000) {
    return { label: s.supervisor_note ? "Restarting" : "Stopped", cls: "bad", running: false };
  }
  if (!fresh) return { label: "Starting…", cls: "warn", running: true };
  if (b.killSwitch) return { label: "Off · no new trades", cls: "warn", running: true };
  if (s.data.today?.blocked) return { label: "Done today", cls: "ok", running: true };
  return { label: "Running", cls: "ok", running: true };
}

// The bot's highest allowed value (50) is shown as "No limit".
const NO_LIMIT = "50";
const lossesText = (v: string | number | null | undefined) =>
  v === null || v === undefined || v === "" ? "–" : Number(v) >= 50 ? "No limit" : String(v);

function TrendBadge({ t }: { t: StatusData["trend"] }) {
  if (!t) return null;
  const cls = t.state === "UPTREND" ? "up" : t.state === "DOWNTREND" ? "down" : "range";
  const txt = t.state === "UPTREND" ? "▲ UP" : t.state === "DOWNTREND" ? "▼ DOWN" : "◆ RANGE";
  return <span className={`trend ${cls}`}>{txt} · {t.timeframe}</span>;
}

const riskText = (r?: string) => (!r ? "" : r.endsWith("%") ? r : `£${r}`);

// ------------------------------------------------------------------ page

export default function Dashboard() {
  const [bots, setBots] = useState<BotRow[] | null>(null);
  const [skew, setSkew] = useState(0);
  const [now, setNow] = useState<number>(0); // set after load (no server/phone clock mismatch)
  const [netErr, setNetErr] = useState(false);
  const [open, setOpen] = useState<"M5" | "M1" | null>(null);
  const [calOpen, setCalOpen] = useState<"M5" | "M1" | null>(null);
  const [months, setMonths] = useState<Record<string, Month>>({});
  const [toast, setToast] = useState<string | null>(null);
  const [news, setNews] = useState<News | null>(null);
  const [confirm, setConfirm] = useState<{ text: string; ok: string; danger?: boolean; run: () => Promise<void> } | null>(null);
  const [pendingToggle, setPendingToggle] = useState<Record<string, boolean>>({});

  const say = useCallback((t: string) => {
    setToast(t);
    setTimeout(() => setToast((x) => (x === t ? null : x)), 3500);
  }, []);

  const api = useCallback(async (url: string, init?: RequestInit) => {
    const res = await fetch(url, { cache: "no-store", ...init });
    if (res.status === 401) {
      window.location.replace("/login");
      throw new Error("login");
    }
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(j.error ?? `Error ${res.status}`);
    return j;
  }, []);

  const loadStatus = useCallback(async () => {
    try {
      const j = await api("/api/status");
      setBots(j.bots);
      setSkew(Date.parse(j.serverTime) - Date.now());
      setNetErr(false);
    } catch (e) {
      if ((e as Error).message !== "login") setNetErr(true);
    }
  }, [api]);

  const loadMonth = useCallback(async (bot: string, ym: string) => {
    try {
      const j = await api(`/api/calendar?bot=${bot}&month=${ym}`);
      setMonths((m) => ({ ...m, [`${bot}|${ym}`]: j }));
    } catch { /* shown as empty */ }
  }, [api]);

  const thisMonth = useMemo(() => {
    const td = bots?.find((b) => b.status?.data?.today?.trading_day)?.status?.data?.today?.trading_day;
    return (td ?? londonDate()).slice(0, 7);
  }, [bots]);

  useEffect(() => {
    // Locks when the app is closed: the PIN screen sets a flag in
    // sessionStorage, which the phone wipes when the app is closed.
    let open = true;
    try { open = sessionStorage.getItem("xb_open") === "1"; } catch { /* storage blocked: rely on the cookie */ }
    if (!open) {
      void fetch("/api/logout", { method: "POST" }).finally(() => window.location.replace("/login"));
      return;
    }
    setNow(Date.now());
    void loadStatus();
    let st = setInterval(loadStatus, 5000);
    const clock = setInterval(() => setNow(Date.now()), 1000);
    const vis = () => {
      clearInterval(st);
      if (document.visibilityState === "visible") {
        void loadStatus();
        st = setInterval(loadStatus, 5000);
      }
    };
    document.addEventListener("visibilitychange", vis);
    return () => { clearInterval(st); clearInterval(clock); document.removeEventListener("visibilitychange", vis); };
  }, [loadStatus]);

  useEffect(() => {
    const go = () => { void loadMonth("M5", thisMonth); void loadMonth("M1", thisMonth); };
    go();
    const t = setInterval(go, 60_000);
    return () => clearInterval(t);
  }, [thisMonth, loadMonth]);

  useEffect(() => {
    const go = () => { api("/api/news").then(setNews).catch(() => { /* keep last */ }); };
    go();
    const t = setInterval(go, 60_000);
    return () => clearInterval(t);
  }, [api]);

  const serverNow = now + skew;

  const sendCommand = (bot: string, command: string) => async () => {
    try {
      const j = await api("/api/command", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bot, command }),
      });
      say(j.already ? "Already on its way" : "Sent. The laptop picks it up within a few seconds.");
      void loadStatus();
    } catch (e) { say((e as Error).message); }
  };

  const setTrading = (bot: string, on: boolean) => async () => {
    setPendingToggle((p) => ({ ...p, [bot]: on }));
    try {
      await api("/api/control", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bot, trading: on }),
      });
      say(on ? `${bot} on. It can trade again within 30s.` : `${bot} off. No new trades within 30s; zones keep updating.`);
      await loadStatus();
    } catch (e) { say((e as Error).message); }
    setPendingToggle((p) => { const n = { ...p }; delete n[bot]; return n; });
  };

  const askToggle = (b: BotRow) => {
    const on = !(pendingToggle[b.bot] ?? !b.killSwitch);
    setConfirm(on
      ? { text: `Turn the ${b.bot} bot back on? It can open trades again straight away.`, ok: "Turn on", run: setTrading(b.bot, true) }
      : { text: `Turn the ${b.bot} bot off? It won't open new trades. Zones keep updating and an open trade is still managed.`, ok: "Turn off", danger: true, run: setTrading(b.bot, false) });
  };

  const header = (
    <header className="top">
      <img src="/icons/logo-96.png" alt="" />
      <div className="grow">
        <h1 className="brand">XAU/TISM BOT</h1>
        <p>2 BOTS · {netErr ? "NO CONNECTION" : "LIVE"} · {now ? new Date(now).toLocaleTimeString("en-GB", { timeZone: TZ, hour: "2-digit", minute: "2-digit" }) : "--:--"} UK</p>
      </div>
      <button className="top-btn" disabled={!bots?.some((b) => b.configured)}
        onClick={() => setConfirm({
          text: "Pull the latest code from GitHub and restart BOTH bots on it? Open trades stay in MT5.",
          ok: "Update",
          run: sendCommand(bots?.find((b) => b.configured)?.bot ?? "M5", "update"),
        })}>UPDATE</button>
    </header>
  );

  if (!bots) {
    return <main className="wrap">{header}<div className="box">{netErr ? "Can't reach the dashboard server." : "Loading…"}</div></main>;
  }

  const openBot = bots.find((b) => b.bot === open) ?? null;

  return (
    <main className="wrap">
      {header}
      {netErr && <div className="banner">Connection lost. Showing the last numbers; retrying every 5 seconds.</div>}

      <div className="row">
        {bots.map((b) => (
          <BotCard key={b.bot} b={b} now={serverNow}
            trading={pendingToggle[b.bot] ?? !b.killSwitch}
            onOpen={() => setOpen(b.bot)} onToggle={() => askToggle(b)} />
        ))}
      </div>

      <h2 className="sec">{monthName(thisMonth)}</h2>
      <div className="row">
        {bots.map((b) => (
          <MiniCal key={b.bot} bot={b.bot} ym={thisMonth} month={months[`${b.bot}|${thisMonth}`]}
            today={b.status?.data?.today?.trading_day ?? londonDate()}
            currency={b.status?.data?.account?.currency}
            onOpen={() => setCalOpen(b.bot)} />
        ))}
      </div>

      <UsdNews news={news} now={serverNow} />

      {openBot && (
        <BotPopup b={openBot} now={serverNow} trading={pendingToggle[openBot.bot] ?? !openBot.killSwitch}
          onClose={() => setOpen(null)} onToggle={() => askToggle(openBot)}
          onCommand={(cmd, text, danger) => setConfirm({ text, ok: "Yes", danger, run: sendCommand(openBot.bot, cmd) })}
          api={api} say={say} reload={loadStatus} />
      )}

      {calOpen && (
        <BigCal bot={calOpen} start={thisMonth} months={months} load={loadMonth} api={api}
          currency={bots.find((b) => b.bot === calOpen)?.status?.data?.account?.currency}
          today={bots.find((b) => b.bot === calOpen)?.status?.data?.today?.trading_day ?? londonDate()}
          onClose={() => setCalOpen(null)} />
      )}

      {confirm && (
        <>
          <div className="ov top" onClick={() => setConfirm(null)} />
          <div className="confirm">
            <p>{confirm.text}</p>
            <div className="btns" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <button className="btn" onClick={() => setConfirm(null)}>Cancel</button>
              <button className={`btn ${confirm.danger ? "danger" : "primary"}`}
                onClick={async () => { const c = confirm; setConfirm(null); await c.run(); }}>{confirm.ok}</button>
            </div>
          </div>
        </>
      )}
      {toast && <div className="toast">{toast}</div>}
    </main>
  );
}

// ------------------------------------------------------------------ bot card

function BotCard({ b, now, trading, onOpen, onToggle }: {
  b: BotRow; now: number; trading: boolean; onOpen: () => void; onToggle: () => void;
}) {
  const st = botState(b, now);
  const d = b.status?.data ?? {};
  const cur = d.account?.currency;
  const today = d.today;
  return (
    <div className="card tap" onClick={onOpen} role="button">
      <div className="hd">
        <div style={{ minWidth: 0 }}>
          <div className="name">{b.bot} BOT</div>
          <div className="sub">{[riskText(d.settings?.risk), d.account?.login ? `#${d.account.login}` : null].filter(Boolean).join(" · ") || " "}</div>
        </div>
        <button className={`sw ${trading ? "on" : ""}`} aria-label={trading ? "Turn off" : "Turn on"}
          disabled={!b.configured || !!b.error}
          onClick={(e) => { e.stopPropagation(); onToggle(); }} />
      </div>
      <div className="st"><span className={`dot ${st.cls}`} />{st.label}</div>
      <TrendBadge t={d.trend} />
      <div className="kv"><label>Balance</label><b>{d.account ? money(d.account.balance, cur) : "–"}</b></div>
      <div className="kv"><label>Today</label><b className={today ? tone(today.realised) : ""}>{today ? signed(today.realised, cur) : "–"}</b></div>
      <div className="kv"><label>W / L today</label><b>{today ? `${today.wins ?? 0}/${today.losses}` : "–"}</b></div>
      <div className="zn">
        <span className="b">BUY</span> {d.next_buy ? <>{d.next_buy.entry.toFixed(2)} · <em>{pts(d.next_buy.distance)} pts</em></> : "none"}<br />
        <span className="s">SELL</span> {d.next_sell ? <>{d.next_sell.entry.toFixed(2)} · <em>{pts(d.next_sell.distance)} pts</em></> : "none"}
      </div>
      <div className="tap-hint">TAP FOR MORE</div>
    </div>
  );
}

// ------------------------------------------------------------------ popup

function ZoneBox({ z, dir }: { z: Zone | null | undefined; dir: "BUY" | "SELL" }) {
  return (
    <div className={`zone ${dir === "BUY" ? "buy" : "sell"}`}>
      <div className="t">{dir}</div>
      {z ? (
        <>
          <div className="p">{z.bottom.toFixed(2)} – {z.top.toFixed(2)}</div>
          <div className="m">
            Entry {z.entry.toFixed(2)}{z.sl !== null ? ` · SL ${z.sl.toFixed(2)}` : ""}<br />
            {z.tp1 !== null ? `TP1 ${z.tp1.toFixed(2)} · ` : ""}{pts(z.distance)} pts away
          </div>
        </>
      ) : (
        <>
          <div className="p">None waiting</div>
          <div className="m">No {dir.toLowerCase()} zone {dir === "BUY" ? "below" : "above"} price</div>
        </>
      )}
    </div>
  );
}

function BotPopup({ b, now, trading, onClose, onToggle, onCommand, api, say, reload }: {
  b: BotRow; now: number; trading: boolean; onClose: () => void; onToggle: () => void;
  onCommand: (cmd: string, text: string, danger?: boolean) => void;
  api: (u: string, i?: RequestInit) => Promise<any>; say: (t: string) => void; reload: () => Promise<void>;
}) {
  const st = botState(b, now);
  const d = b.status?.data ?? {};
  const cur = d.account?.currency;
  const today = d.today;
  const [editing, setEditing] = useState(false);
  const [showOut, setShowOut] = useState<number | null>(null);
  const updAgo = b.status ? Math.max(0, Math.round((now - Date.parse(b.status.updated_at)) / 1000)) : null;

  return (
    <>
      <div className="ov" onClick={onClose} />
      <div className="pop fit">
        <button className="x" onClick={onClose} aria-label="Close">✕</button>
        <div className="hd" style={{ paddingRight: 30 }}>
          <div style={{ minWidth: 0 }}>
            <div className="name" style={{ fontSize: 17 }}>{b.bot} BOT</div>
            <div className="sub">
              {[d.account?.login ? `#${d.account.login}` : null, d.timeframe ? `${d.timeframe} zones` : null, d.settings?.runner].filter(Boolean).join(" · ")}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, fontWeight: 600, color: "var(--gold2)" }}>
            {trading ? "ON" : "OFF"}
            <button className={`sw ${trading ? "on" : ""}`} onClick={onToggle} disabled={!b.configured || !!b.error} aria-label="Switch" />
          </div>
        </div>
        <div className="st" style={{ fontSize: 10.5 }}>
          <span className={`dot ${st.cls}`} />{st.label}
          {updAgo !== null && updAgo < 3600 ? ` · updated ${updAgo}s ago` : ""}
          {d.bid ? ` · price ${d.bid.toFixed(2)}` : ""}
        </div>
        {b.error && <div className="banner">{b.error}</div>}
        {!b.configured && <div className="banner">Add {b.bot}_SUPABASE_URL and {b.bot}_SUPABASE_SERVICE_ROLE_KEY in Vercel.</div>}
        {b.status?.supervisor_note && <div className="banner">{b.bot} bot {b.status.supervisor_note}</div>}
        {today?.blocked && today.blocked_reason && <div className="box" style={{ marginTop: 8 }}>Stopped for the day: {today.blocked_reason}</div>}

        <div className="stats">
          <div className="stat"><label>Balance</label><b>{d.account ? money(d.account.balance, cur) : "–"}</b></div>
          <div className="stat"><label>Today</label><b className={today ? tone(today.realised) : ""}>{today ? signed(today.realised, cur) : "–"}</b></div>
          <div className="stat"><label>W / L today</label><b>{today ? `${today.wins ?? 0}/${today.losses}` : "–"}</b></div>
        </div>

        <h3>TREND</h3>
        <div className="box">
          {d.trend ? (
            <>
              <TrendBadge t={d.trend} />{" "}
              {d.trend.state === "RANGE" ? "No break yet, so buys and sells are both allowed." :
                d.trend.state === "DOWNTREND" ? "Buys blocked (unless inside a fresh higher-timeframe zone)." :
                "Sells blocked (unless inside a fresh higher-timeframe zone)."}
              <br />
              {d.trend.since && <>Since {dayMon(d.trend.since)} {hhmm(d.trend.since)} UK. </>}
              {d.trend.up_above !== null && <>Turns <b className="pos">up</b> on an {d.trend.timeframe} close above <b>{d.trend.up_above.toFixed(2)}</b>. </>}
              {d.trend.down_below !== null && <>Turns <b className="neg">down</b> on an {d.trend.timeframe} close below <b>{d.trend.down_below.toFixed(2)}</b>.</>}
            </>
          ) : "Waiting for the bot's first candles"}
        </div>

        <h3>OPEN TRADE</h3>
        <div className="box">
          {d.open_positions?.length ? d.open_positions.map((p) => (
            <div key={p.ticket} className="trade-line">
              <span>{p.direction} {p.volume} @ {p.entry.toFixed(2)}{p.sl ? ` · SL ${p.sl.toFixed(2)}` : ""}{p.tp ? ` · TP ${p.tp.toFixed(2)}` : ""}</span>
              <b className={tone(p.profit)}>{signed(p.profit, cur)}</b>
            </div>
          )) : <>No open trade{d.spread !== undefined ? ` · spread ${d.spread.toFixed(2)}` : ""}</>}
        </div>

        <h3>CLOSEST ZONES</h3>
        <div className="two">
          <ZoneBox z={d.next_buy} dir="BUY" />
          <ZoneBox z={d.next_sell} dir="SELL" />
        </div>

        <h3>SETTINGS</h3>
        <div className="two">
          <button className="field" onClick={() => setEditing(true)} disabled={!b.configured || !!b.error}>
            <label>Risk per trade</label>
            <div>{riskText(d.settings?.risk) || riskText(b.risk ?? undefined) || "–"}</div>
            <div className="edit">{b.risk ? "SET HERE · TAP TO CHANGE" : "BOT DEFAULT · TAP TO CHANGE"}</div>
          </button>
          <button className="field" onClick={() => setEditing(true)} disabled={!b.configured || !!b.error}>
            <label>Max losses / day</label>
            <div>{lossesText(d.settings?.max_losses ?? b.maxLosses)} {today ? <small>· {today.losses} used today</small> : null}</div>
            <div className="edit">{b.maxLosses ? "SET HERE · TAP TO CHANGE" : "BOT DEFAULT · TAP TO CHANGE"}</div>
          </button>
        </div>

        <div className="btns" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <button className="btn" disabled={!b.configured}
            onClick={() => onCommand("restart", `Restart the ${b.bot} bot? Open trades stay in MT5.`)}>Restart</button>
          {st.running
            ? <button className="btn danger" disabled={!b.configured}
                onClick={() => onCommand("stop", `Stop the ${b.bot} bot completely? Open trades and orders stay in MT5 but nothing manages them until you start it again.`, true)}>Stop</button>
            : <button className="btn" disabled={!b.configured}
                onClick={() => onCommand("start", `Start the ${b.bot} bot?`)}>Start</button>}
        </div>

        <h3>LAST COMMANDS</h3>
        <div className="box">
          {b.commands?.length ? b.commands.map((c) => (
            <div key={c.id} onClick={() => setShowOut(showOut === c.id ? null : c.id)} style={{ cursor: c.output ? "pointer" : "default" }}>
              <div className="cmd">
                <span>{c.command[0].toUpperCase() + c.command.slice(1)} · <span className={c.status}>{c.status === "pending" ? "waiting for laptop" : c.status}</span></span>
                <span>{dayMon(c.created_at)} {hhmm(c.created_at)}</span>
              </div>
              {showOut === c.id && c.output && <div className="cmd-out">{c.output}</div>}
            </div>
          )) : "None yet"}
        </div>
        {d.version && <p className="note">Code {d.version}{d.started_at ? ` · running since ${dayMon(d.started_at)} ${hhmm(d.started_at)}` : ""}</p>}
      </div>
      {editing && <SettingsEditor b={b} onClose={() => setEditing(false)} api={api} say={say} reload={reload} />}
    </>
  );
}

// ------------------------------------------------------------------ settings

function SettingsEditor({ b, onClose, api, say, reload }: {
  b: BotRow; onClose: () => void;
  api: (u: string, i?: RequestInit) => Promise<any>; say: (t: string) => void; reload: () => Promise<void>;
}) {
  const [risk, setRisk] = useState<string>(b.risk ?? "");
  const [losses, setLosses] = useState<string>(b.maxLosses ? String(b.maxLosses) : "");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const RISKS = ["0.5%", "1%", "2%", "50", "100", "150"];
  const LOSSES = ["1", "2", "3", "4", "5", "6", "8", "10", NO_LIMIT];

  const save = async () => {
    setBusy(true); setErr(null);
    try {
      await api("/api/control", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bot: b.bot, risk: risk || null, maxLosses: losses || null }),
      });
      say(`${b.bot} settings saved. The bot picks them up within 30s.`);
      await reload();
      onClose();
    } catch (e) { setErr((e as Error).message); }
    setBusy(false);
  };

  return (
    <>
      <div className="ov top" onClick={onClose} />
      <div className="confirm" style={{ top: "12%" }}>
        <div className="name" style={{ fontSize: 14 }}>{b.bot} BOT SETTINGS</div>
        <h3>RISK PER TRADE</h3>
        <div className="chips">
          <button className={`chip ${risk === "" ? "on" : ""}`} onClick={() => setRisk("")}>Bot default</button>
          {RISKS.map((r) => (
            <button key={r} className={`chip ${risk === r ? "on" : ""}`} onClick={() => setRisk(r)}>{riskText(r)}</button>
          ))}
        </div>
        <input className="input" placeholder="Or type: 0.75% or 120" value={risk} onChange={(e) => setRisk(e.target.value.replace(/£/g, ""))} />
        <h3>MAX LOSSES A DAY</h3>
        <div className="chips">
          <button className={`chip ${losses === "" ? "on" : ""}`} onClick={() => setLosses("")}>Bot default</button>
          {LOSSES.map((l) => (
            <button key={l} className={`chip ${losses === l ? "on" : ""}`} onClick={() => setLosses(l)}>{lossesText(l)}</button>
          ))}
        </div>
        <p className="note">
          % = share of balance per trade, a number = fixed £ per trade. The bot also stops for the day once it&apos;s down
          (max losses + 1) trades&apos; worth. No limit = it keeps trading all day. Bot default = what&apos;s in its .env file. Changes apply within 30 seconds, from the next trade.
        </p>
        {err && <div className="err">{err}</div>}
        <div className="btns" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" onClick={save} disabled={busy}>{busy ? "Saving…" : "Save"}</button>
        </div>
      </div>
    </>
  );
}

// ------------------------------------------------------------------ calendars

function cells(ym: string): (string | null)[] {
  const [y, m] = ym.split("-").map(Number);
  const first = new Date(Date.UTC(y, m - 1, 1)).getUTCDay(); // 0 = Sunday
  const lead = (first + 6) % 7; // Monday first
  const n = new Date(Date.UTC(y, m, 0)).getUTCDate();
  const out: (string | null)[] = Array(lead).fill(null);
  for (let d = 1; d <= n; d++) out.push(`${ym}-${String(d).padStart(2, "0")}`);
  return out;
}
const WEEK = ["M", "T", "W", "T", "F", "S", "S"];

function MiniCal({ bot, ym, month, today, currency, onOpen }: {
  bot: string; ym: string; month?: Month; today: string; currency?: string; onOpen: () => void;
}) {
  return (
    <div className="card tap" onClick={onOpen} role="button">
      <div className="ct"><b>{bot} BOT</b><span className={tone(month?.total ?? 0)}>{month ? signed(month.total, currency, 0) : "–"}</span></div>
      <div className="cm">{month ? `${month.trades} TRADES · ` : ""}TAP TO ENLARGE</div>
      <div className="mw">{WEEK.map((w, i) => <div key={i}>{w}</div>)}</div>
      <div className="mg">
        {cells(ym).map((day, i) => {
          if (!day) return <div key={i} className="c e" />;
          const c = month?.days[day];
          const cls = c ? (c.pnl > 0 ? "up" : c.pnl < 0 ? "dn" : "flat") : "";
          return (
            <div key={i} className={`c ${cls} ${day === today ? "td" : ""}`}>
              <i>{Number(day.slice(8))}</i>{c ? <b>{short(c.pnl)}</b> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function BigCal({ bot, start, months, load, api, currency, today, onClose }: {
  bot: string; start: string; months: Record<string, Month>; load: (b: string, ym: string) => Promise<void>;
  api: (u: string, i?: RequestInit) => Promise<any>; currency?: string; today: string; onClose: () => void;
}) {
  const [ym, setYm] = useState(start);
  const [day, setDay] = useState<string | null>(null);
  const [trades, setTrades] = useState<DayTrade[] | null>(null);
  const month = months[`${bot}|${ym}`];
  const asked = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!asked.current.has(ym)) { asked.current.add(ym); void load(bot, ym); }
  }, [bot, ym, load]);

  useEffect(() => {
    if (!day) return;
    setTrades(null);
    api(`/api/day?bot=${bot}&date=${day}`).then((j) => setTrades(j.trades)).catch(() => setTrades([]));
  }, [bot, day, api]);

  return (
    <>
      <div className="ov" onClick={onClose} />
      <div className="pop fit">
        <button className="x" onClick={onClose} aria-label="Close">✕</button>
        {!day ? (
          <>
            <div className="name" style={{ fontSize: 16, marginBottom: 8 }}>{bot} BOT CALENDAR</div>
            <div className="cal-nav">
              <button onClick={() => setYm(shiftMonth(ym, -1))} aria-label="Previous month">‹</button>
              <b>{monthName(ym)}</b>
              <button onClick={() => setYm(shiftMonth(ym, 1))} aria-label="Next month">›</button>
            </div>
            <div className="bw">{["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"].map((w) => <div key={w}>{w}</div>)}</div>
            <div className="bg">
              {cells(ym).map((dd, i) => {
                if (!dd) return <div key={i} className="d e" />;
                const c = month?.days[dd];
                const cls = c ? (c.pnl > 0 ? "up" : c.pnl < 0 ? "dn" : "") : "";
                return (
                  <button key={i} className={`d ${cls} ${dd === today ? "td" : ""}`} onClick={() => c && setDay(dd)} disabled={!c} style={{ opacity: 1 }}>
                    <i>{Number(dd.slice(8))}</i>
                    {c ? <><b>{signed(c.pnl, currency, 0)}</b><small>{c.wins}/{c.losses}</small></> : null}
                  </button>
                );
              })}
            </div>
            <div className="month-sum">
              <div className="stat"><label>Month</label><b className={tone(month?.total ?? 0)}>{month ? signed(month.total, currency) : "–"}</b></div>
              <div className="stat"><label>Trades</label><b>{month?.trades ?? "–"}</b></div>
              <div className="stat"><label>W / L</label><b>{month ? `${month.wins}/${month.losses}` : "–"}</b></div>
            </div>
            <p className="note">Days run 22:00 to 22:00 UK (the bot&apos;s trading day). Tap a day to see its trades.</p>
          </>
        ) : (
          <>
            <div className="cal-nav" style={{ marginRight: 30 }}>
              <button onClick={() => setDay(null)}>‹ Back</button>
              <b>{new Date(`${day}T12:00:00Z`).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" }).toUpperCase()}</b>
              <span />
            </div>
            {month?.days[day] && (
              <div className="month-sum" style={{ marginTop: 0, marginBottom: 10 }}>
                <div className="stat"><label>Day</label><b className={tone(month.days[day].pnl)}>{signed(month.days[day].pnl, currency)}</b></div>
                <div className="stat"><label>Trades</label><b>{month.days[day].trades}</b></div>
                <div className="stat"><label>W / L</label><b>{month.days[day].wins}/{month.days[day].losses}</b></div>
              </div>
            )}
            {trades === null ? <div className="box">Loading…</div> : trades.length === 0 ? <div className="box">No trades</div> : (
              trades.map((t, i) => (
                <div key={i} className="zone" style={{ marginBottom: 8 }}>
                  <div className="trade-line">
                    <span className="t" style={{ color: t.direction === "BUY" ? "var(--up)" : "var(--down)" }}>{t.direction}</span>
                    <b className={tone(t.profit)}>{signed(t.profit, currency)}</b>
                  </div>
                  <div className="meta">{hhmm(t.open)} → {hhmm(t.close)} UK · {t.layers.length} {t.layers.length === 1 ? "position" : "positions"}</div>
                  {t.layers.map((l, j) => (
                    <div key={j} className="meta trade-line">
                      <span>{l.volume} lots · {l.open?.toFixed(2) ?? "?"} → {l.close?.toFixed(2) ?? "?"}</span>
                      <span className={tone(l.profit)}>{signed(l.profit, currency)}</span>
                    </div>
                  ))}
                </div>
              ))
            )}
          </>
        )}
      </div>
    </>
  );
}

// ------------------------------------------------------------------ USD news

function Folder({ impact }: { impact: CalEvent["impact"] }) {
  return <span className={`folder ${impact.toLowerCase()}`} title={`${impact} impact`} />;
}

const ago = (iso: string, now: number) => {
  const m = Math.max(0, Math.round((now - Date.parse(iso)) / 60_000));
  return m < 1 ? "just now" : m < 60 ? `${m} min ago` : `${Math.floor(m / 60)} hr ago`;
};
const until = (iso: string, now: number) => {
  const m = Math.max(0, Math.round((Date.parse(iso) - now) / 60_000));
  return m < 60 ? `in ${m} min` : `in ${Math.floor(m / 60)}h ${m % 60}m`;
};

function UsdNews({ news, now }: { news: News | null; now: number }) {
  const today = londonDate();
  const events = (news?.calendar ?? []).filter(
    (e) => new Date(e.time).toLocaleDateString("en-CA", { timeZone: TZ }) >= today,
  );
  const days = new Map<string, CalEvent[]>();
  for (const e of events) {
    const d = new Date(e.time).toLocaleDateString("en-CA", { timeZone: TZ });
    days.set(d, [...(days.get(d) ?? []), e]);
  }
  const next = events.find((e) => Date.parse(e.time) > now && e.impact !== "Holiday");
  const fresh = (h: Headline) => now - Date.parse(h.time) < 15 * 60_000;

  return (
    <>
      <h2 className="sec">USD NEWS · THIS WEEK</h2>
      <div className="card news">
        {!news ? <div className="meta">Loading…</div> : events.length === 0 ? (
          <div className="meta">{news.calendarError ? `Can't load the calendar right now (${news.calendarError})` : "No more USD events this week"}</div>
        ) : [...days.entries()].map(([d, list]) => (
          <div key={d}>
            <div className={`news-day ${d === today ? "today" : ""}`}>
              {new Date(`${d}T12:00:00Z`).toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" }).toUpperCase()}
              {d === today ? " · TODAY" : ""}
            </div>
            {list.map((e, i) => {
              const past = Date.parse(e.time) <= now;
              const isNext = next === e;
              return (
                <div key={i} className={`news-row ${past ? "past" : ""} ${isNext ? "next" : ""}`}>
                  <span className="news-time">{hhmm(e.time)}</span>
                  <Folder impact={e.impact} />
                  <span className="news-title">
                    {e.title}
                    {isNext && <em> · {until(e.time, now)}</em>}
                  </span>
                  {(e.forecast || e.previous) && (
                    <span className="news-fp">
                      {e.forecast ? <>F {e.forecast}</> : null}{e.forecast && e.previous ? <br /> : null}{e.previous ? <>P {e.previous}</> : null}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ))}
        <p className="note" style={{ marginTop: 8 }}>Forex Factory, USD only, UK times. Red = high, orange = medium, yellow = low impact.</p>
      </div>

      <h2 className="sec">USD HEADLINES</h2>
      <div className="news-heads">
        {!news ? <div className="box">Loading…</div> : news.headlines.length === 0 ? (
          <div className="box">{news.headlinesError ? `Can't load headlines right now (${news.headlinesError})` : "No USD headlines in the last 12 hours"}</div>
        ) : news.headlines.map((h) => (
          <a key={h.id} className={`head ${fresh(h) ? "fresh" : ""}`} href={h.link ?? undefined} target="_blank" rel="noreferrer">
            <div className="head-title">{fresh(h) && <span className="badge">NEW</span>}{h.title}</div>
            <div className="meta">FinancialJuice · {ago(h.time, now)} · {hhmm(h.time)} UK</div>
            {h.summary && <div className="head-sum">{h.summary}</div>}
          </a>
        ))}
      </div>
    </>
  );
}
