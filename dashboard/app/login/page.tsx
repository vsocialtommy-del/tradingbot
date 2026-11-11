"use client";

import { useEffect, useState } from "react";

export default function Login() {
  const [pin, setPin] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [lockedUntil, setLockedUntil] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  const locked = lockedUntil !== null && lockedUntil > now;

  async function submit(code: string) {
    setBusy(true);
    setMsg(null);
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pin: code }),
      });
      const j = await res.json().catch(() => ({}));
      if (res.ok) {
        try { sessionStorage.setItem("xb_open", "1"); } catch { /* storage blocked */ }
        window.location.replace("/");
        return;
      }
      if (j.error === "locked") {
        setLockedUntil(Date.parse(j.lockedUntil));
      } else if (j.error === "wrong") {
        setMsg(`Wrong PIN. ${j.triesLeft} ${j.triesLeft === 1 ? "try" : "tries"} left before a 15 minute lock.`);
      } else {
        setMsg(j.error ?? "Something went wrong");
      }
    } catch {
      setMsg("No connection");
    }
    setPin("");
    setBusy(false);
  }

  function press(d: string) {
    if (busy || locked) return;
    if (d === "del") return setPin((p) => p.slice(0, -1));
    const next = (pin + d).slice(0, 4);
    setPin(next);
    if (next.length === 4) void submit(next);
  }

  const left = locked ? Math.ceil((lockedUntil! - now) / 1000) : 0;

  return (
    <main className="login">
      <img src="/icons/logo-192.png" alt="" className="login-logo" />
      <h1 className="brand">XAU/TISM BOT</h1>
      <p className="login-sub">{locked ? "Locked" : "Enter your PIN"}</p>
      <div className="pin-dots">
        {[0, 1, 2, 3].map((i) => (
          <span key={i} className={i < pin.length ? "on" : ""} />
        ))}
      </div>
      <p className="login-msg">
        {locked
          ? `Too many wrong tries. Try again in ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`
          : msg ?? " "}
      </p>
      <div className="pad">
        {["1", "2", "3", "4", "5", "6", "7", "8", "9", "", "0", "del"].map((d, i) =>
          d === "" ? <span key={i} /> : (
            <button key={i} onClick={() => press(d)} disabled={busy || locked} aria-label={d === "del" ? "Delete" : d}>
              {d === "del" ? "⌫" : d}
            </button>
          ),
        )}
      </div>
    </main>
  );
}
