# 🎯 Tommy's Trading Bot — Personal Todo List

This is your step-by-step list. What you physically do.
The tools (Claude Code, Cowork, Claude in Chrome) do the work — you direct them.

Companion file: `strategy-spec-v4-final.md`

---

## ✅ STEP 1: Get the spec ready (5 mins)

- [ ] Save `strategy-spec-v4-final.md` to laptop
- [ ] Make sure you can find it easily

---

## ✅ STEP 2: Set up accounts (30 mins)

You do this manually in browser:

- [ ] Sign up at vantagemarkets.com → create demo account → save login details
- [ ] Create new GitHub repo: `gold-trading-bot` (private)
- [ ] Create new Supabase project: `gold-bot`
- [ ] Sign up at finnhub.io → get free API key (for news filter)
- [ ] Create new Vercel project (will deploy dashboard later)
- [ ] Save all credentials in a notes app

🔵 Claude in Chrome can help: open Chrome with extension → "Sign me up for a Vantage demo account" → it fills the forms

---

## ✅ STEP 3: Install software on laptop (20 mins)

- [ ] Install Python 3.11+ from python.org
- [ ] Install MetaTrader 5 desktop app from Vantage's download link
- [ ] Open MT5, log in with demo credentials, check XAUUSD chart loads

---

## ✅ STEP 4: Open Claude Code in new project folder (5 mins)

- [ ] Make new folder on laptop: `gold-trading-bot`
- [ ] Open Claude Code in that folder
- [ ] Drag `strategy-spec-v4-final.md` into the chat
- [ ] Type: "Read this spec carefully. We're building Phase A. Start with Section 10 — create the project folder structure and set up Python environment."

---

## ✅ STEP 5: Let Claude Code build Phase A (1-2 hours)

- [ ] Watch what Claude Code does
- [ ] When it asks questions, answer them
- [ ] When it finishes, review the PR link it gives you
- [ ] Merge the PR on GitHub
- [ ] Tell Claude Code: "Merged. Now run the test to confirm MT5 library imports successfully."

---

## ✅ STEP 6: Set up Supabase tables (15 mins)

- [ ] Tell Claude Code: "Write the SQL migrations for the 6 tables in Section 9.2 of the spec."
- [ ] Copy the SQL Claude Code generates
- [ ] Open Supabase dashboard → SQL Editor → paste → run
- [ ] Tell Claude Code: "Tables created. Now write the Python supabase_logger.py wrapper."
- [ ] Merge PR

🔵 Claude in Chrome can do this for you: "Open my Supabase project, go to SQL Editor, paste this SQL and run it"

---

## ✅ STEP 7: Test MT5 connection (30 mins)

- [ ] Tell Claude Code: "Build the MT5 connector per Section 9 of the spec. Then write a test script that connects, prints my balance, and fetches 100 candles of XAUUSD."
- [ ] Add your MT5 credentials to the `.env` file Claude Code created
- [ ] Run the test (Claude Code will give you the command)
- [ ] If it fails → paste the error back to Claude Code → it debugs
- [ ] Repeat until balance prints successfully

---

## ✅ STEP 8: Build core strategy modules (Week 2-3)

For each module in the spec (Sections 3-6), repeat this pattern:

- [ ] Tell Claude Code: "Build [module name] per Section X of the spec."
- [ ] Review the PR link
- [ ] Merge
- [ ] Tell Claude Code: "Run the unit tests for this module."
- [ ] If tests pass → next module. If they fail → paste error → debug.

**Module order:**

1. `structure.py` (HH/HL tracking)
2. `pattern_detection.py` (W/M/N)
3. `zone_marking.py`
4. `zone_refinement.py`
5. `strong_point.py`
6. `imbalance.py`

---

## ✅ STEP 9: Validate strategy against your eyes (1 hour)

- [ ] Take 5-10 screenshots of recent XAUUSD charts where you can see clear setups
- [ ] Tell Claude Code: "Run the bot's pattern detector on the same time period these screenshots cover. Show me what zones it found."
- [ ] Compare side-by-side: does the bot find the same zones you'd mark?
- [ ] If yes → continue. If no → tell Claude Code what's wrong, iterate.

---

## ✅ STEP 10: Build execution layer (Week 3-4)

- [ ] Tell Claude Code: "Build the execution layer — order manager, position tracker, TP1 manager, risk checks. Per Section 4-7 of the spec."
- [ ] Review PRs, merge each one
- [ ] Tell Claude Code: "Build main.py to tie everything together into a runnable bot."
- [ ] Run the bot on demo for 1 hour, watch what it does in MT5 mobile app

---

## ✅ STEP 11: Backtest (Week 4-5)

- [ ] Download XAUUSD historical data from Dukascopy

🔵 Claude in Chrome can do this: "Go to Dukascopy historical data, configure XAUUSD M1 from 2021-2025, download CSV"

- [ ] Tell Claude Code: "Build the backtest framework per Section 11. Run it on this historical data."
- [ ] Review backtest results
- [ ] Tell Claude Code: "Win rate is X%, profit factor is Y. Adjust parameters [Z] to see if we can improve."
- [ ] Iterate until backtest passes spec criteria

---

## ✅ STEP 12: Build dashboard (Week 5-6)

- [ ] Tell Claude Code: "Build the Next.js dashboard per Section 9.3 of the spec. Include kill switch, live trades, performance summary, config editor."
- [ ] Review, merge, deploy to Vercel (auto-deploys from GitHub, same as eltree.app)
- [ ] Tell Claude Code: "Build Telegram alerts for trade events using a Vercel API route."
- [ ] Tell Claude Code: "Set up Vercel cron job for news scraping every 15 mins."
- [ ] Test alerts work on your phone

---

## ✅ STEP 13: Run demo for 30+ days (Month 2-3)

**Daily routine (15 mins):**

- [ ] Open dashboard on phone, check overnight trades
- [ ] Note any weird behaviour

**Weekly routine (1 hour):**

- [ ] Open Cowork on desktop
- [ ] Tell Cowork: "Pull this week's trades from Supabase. Summarise performance vs backtest. Flag any anomalies. Write a markdown report."
- [ ] Read the report
- [ ] If issues found → open Claude Code → "This week showed [issue]. Investigate and propose fix."

---

## ✅ STEP 14: Go live micro (Month 3-4)

- [ ] Set up Hetzner VPS (Claude Code walks you through)
- [ ] Deploy bot to VPS
- [ ] Fund Vantage live account with $500
- [ ] Switch bot config from demo → live
- [ ] Monitor like a hawk for first 2 weeks

---

## ✅ STEP 15: Scale + vSocial (Month 4+)

- [ ] Once 60+ days of profitable live → enable risk-based sizing (v1.1)
- [ ] Set up vSocial signal account
- [ ] Promote to copiers gradually

---

## 🔧 Tool Quick Reference

| Tool | When to use it |
|---|---|
| Claude Code | All actual coding — building modules, fixing bugs, running scripts |
| Cowork | Weekly performance reviews, reading reports, file organisation |
| 🔵 Claude in Chrome | Form-filling on Vantage/Supabase, browsing Dukascopy for data, navigating dashboards |
| MT5 mobile app | Watching live trades on phone |
| Dashboard (Next.js on Vercel) | Custom monitoring, kill switch, config editor |
| GitHub | Reviewing and merging PRs from Claude Code |
| Supabase dashboard | Spot-checking data, running ad-hoc queries |
| Hetzner VPS | Where the bot lives — only touch for setup + rare debugging |

---

## 🚦 The Pattern That Repeats

For 90% of the build, your loop is:

1. Tell Claude Code what to build (reference the spec section)
2. Wait for it to finish + give you a PR link
3. Review the PR on GitHub
4. Merge it
5. Tell Claude Code what's next

That's the rhythm. Keep doing it until the bot is built.

---

## ⚠️ Things to Watch For

- **Don't rush past failed tests.** If a test fails, fix it before moving on.
- **Don't skip the spec doc.** Always reference the relevant section when prompting Claude Code.
- **Don't go live before backtest passes.** All criteria in spec Section 11.2 must be met.
- **Don't skip demo phase.** Minimum 30 days, preferably 60.
- **Don't enable vSocial until you have 60+ days of profitable live trading.**

---

End of Tommy's Todo List — Companion to strategy-spec-v4-final.md
