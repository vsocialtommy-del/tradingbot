"""Laptop supervisor for the phone dashboard (Nov 2026).

Runs both bots on the laptop and carries out the dashboard's buttons::

    python -m bot.supervisor            (start_bots.bat does this)

* Starts the M5 bot (``.env``) and the M1 bot (``.env.m1``) as child
  processes, with no window each. Their output goes to ``logs/``.
* Restarts a bot that crashes (waiting a bit longer each time).
* Every few seconds, looks in each bot's own Supabase project for
  dashboard commands (``bot_commands``) and runs them:

  - ``update``   git pull (plus pip install if requirements.txt changed),
                 then restart BOTH bots, since they share the code
  - ``restart``  stop that bot, start it again
  - ``stop``     stop that bot and keep it stopped
  - ``start``    start that bot

* Writes a heartbeat to ``bot_status`` (``supervisor_seen_at`` and
  ``process_running``) so the dashboard can tell "laptop offline" from
  "bot crashed".

Stopping a bot is the same as pressing Ctrl+C in its window: it finishes
the current loop and disconnects. Open trades and pending orders stay in
MT5 exactly as they are.

Commands older than 10 minutes are ignored, so a button pressed while the
laptop was off doesn't fire hours later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]

# The bot exits with this when another copy is already running
# (bot/main.py ALREADY_RUNNING_EXIT).
ALREADY_RUNNING_EXIT = 75
# The supervisor exits with this to ask start_bots.bat to start a fresh
# copy of it (after an update changed the supervisor's own code).
RELAUNCH_EXIT = 3

BOTS = {"m5": ("M5", ".env"), "m1": ("M1", ".env.m1")}
COMMANDS = ("update", "restart", "start", "stop")
STALE_AFTER = timedelta(minutes=10)
STOP_TIMEOUT = 45.0
STATE_FILE = ROOT / "logs" / "supervisor_state.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _missing_table(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(s in text for s in (
        "does not exist", "pgrst205", "could not find the table", "schema cache", "404",
    ))


# --------------------------------------------------------------------------- #
# Child processes
# --------------------------------------------------------------------------- #


@dataclass
class BotProcess:
    """One bot as a child process."""

    key: str                      # "m5" / "m1"
    label: str                    # "M5" / "M1"
    popen: Callable[..., Any] = subprocess.Popen
    clock: Callable[[], float] = time.monotonic
    wanted: bool = True           # False after Stop from the dashboard
    proc: Any = None
    started_at: float = 0.0
    crashes: int = 0
    next_try: float = 0.0
    note: str | None = None       # last thing worth telling the dashboard

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def command(self) -> list[str]:
        return [sys.executable, "-m", "bot.main", "--bot", self.key]

    def start(self) -> None:
        if self.running:
            return
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        out_path = logs / f"{self.key}.console.log"
        try:
            if out_path.exists() and out_path.stat().st_size > 20 * 1024 * 1024:
                out_path.replace(logs / f"{self.key}.console.old.log")
        except OSError:
            pass
        out = open(out_path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115 (handed to the child)
        env = dict(os.environ)
        env.setdefault("LOGURU_LEVEL", "INFO")  # the window lag came from DEBUG spam
        env["PYTHONUNBUFFERED"] = "1"
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.proc = self.popen(
            self.command(), cwd=str(ROOT), env=env,
            stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        out.close()  # the child has its own handle now
        self.started_at = self.clock()
        self.note = None
        logger.info(f"{self.label} bot started (pid {self.proc.pid})")

    def stop(self, timeout: float = STOP_TIMEOUT) -> bool:
        """Ask the bot to stop like Ctrl+C would; force it after ``timeout``.
        Returns True if it stopped by itself."""
        if not self.running:
            self.proc = None
            return True
        logger.info(f"stopping the {self.label} bot (pid {self.proc.pid})")
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.send_signal(signal.SIGINT)
        except Exception:  # noqa: BLE001
            logger.exception(f"{self.label}: couldn't send the stop signal")
        try:
            self.proc.wait(timeout=timeout)
            clean = True
        except subprocess.TimeoutExpired:
            logger.warning(f"{self.label} bot didn't stop in {timeout:.0f}s; forcing it")
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            clean = False
        self.proc = None
        logger.info(f"{self.label} bot stopped")
        return clean

    def check(self) -> None:
        """Restart a bot that ended when it wasn't asked to."""
        if not self.wanted:
            return
        if self.proc is not None and self.proc.poll() is None:
            if self.clock() - self.started_at > 600:
                self.crashes = 0  # ran 10 minutes: reset the back-off
            return
        now = self.clock()
        if self.proc is not None:
            code = self.proc.poll()
            self.proc = None
            if code == ALREADY_RUNNING_EXIT:
                self.note = (
                    "already running in another window; close it and the "
                    "supervisor takes over within a minute"
                )
                logger.warning(f"{self.label} bot is {self.note}")
                self.next_try = now + 60
                return
            self.crashes += 1
            wait = min(10 * 2 ** (self.crashes - 1), 300)
            self.note = f"stopped by itself (exit code {code}); restarting in {wait}s"
            logger.error(f"{self.label} bot {self.note}. See logs/{self.key}.console.log")
            self.next_try = now + wait
            return
        if now >= self.next_try:
            self.start()


# --------------------------------------------------------------------------- #
# Supabase side (one project per bot)
# --------------------------------------------------------------------------- #


class Channel:
    """Commands + heartbeat for one bot in its own Supabase project."""

    def __init__(self, client: Any, label: str) -> None:
        self.client = client
        self.label = label
        self._off_until = 0.0
        self._warned = False

    def _failed(self, what: str, exc: Exception) -> None:
        if _missing_table(exc):
            if not self._warned:
                logger.warning(
                    f"{self.label}: dashboard tables not in Supabase yet (run "
                    "migrations/016_dashboard.sql). Bots run as normal; retrying "
                    "every 5 minutes."
                )
                self._warned = True
            self._off_until = time.monotonic() + 300
        else:
            logger.warning(f"{self.label}: {what} failed ({exc})")

    @property
    def available(self) -> bool:
        return time.monotonic() >= self._off_until

    def pending(self) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            res = (
                self.client.table("bot_commands").select("*")
                .eq("bot", self.label).eq("status", "pending")
                .order("created_at").limit(10).execute()
            )
            self._warned = False
            return list(res.data or [])
        except Exception as exc:  # noqa: BLE001
            self._failed("reading commands", exc)
            return []

    def claim(self, cmd_id: int) -> bool:
        """pending -> running; False if something else already took it."""
        try:
            res = (
                self.client.table("bot_commands")
                .update({"status": "running", "started_at": _now().isoformat()})
                .eq("id", cmd_id).eq("status", "pending").execute()
            )
            return bool(res.data)
        except Exception as exc:  # noqa: BLE001
            self._failed("claiming a command", exc)
            return False

    def finish(self, cmd_id: int, ok: bool, output: str) -> None:
        try:
            self.client.table("bot_commands").update({
                "status": "done" if ok else "failed",
                "finished_at": _now().isoformat(),
                "output": output[-4000:],
            }).eq("id", cmd_id).execute()
        except Exception as exc:  # noqa: BLE001
            self._failed("saving a command result", exc)

    def heartbeat(self, running: bool, note: str | None) -> None:
        if not self.available:
            return
        row = {
            "supervisor_seen_at": _now().isoformat(),
            "process_running": running,
            "supervisor_note": note,
        }
        try:
            res = self.client.table("bot_status").update(row).eq("bot", self.label).execute()
            if not res.data:
                # First time: the bot hasn't written its row yet.
                self.client.table("bot_status").insert({
                    "bot": self.label, "updated_at": "1970-01-01T00:00:00+00:00",
                    "data": {}, **row,
                }).execute()
            self._warned = False
        except Exception as exc:  # noqa: BLE001
            self._failed("heartbeat", exc)


def make_channel(key: str) -> Channel | None:
    """Supabase client for one bot from its env file (not os.environ, so
    the two bots' keys never mix)."""
    from dotenv import dotenv_values

    label, env_file = BOTS[key]
    vals = dotenv_values(ROOT / env_file)
    url, secret = vals.get("SUPABASE_URL"), vals.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not secret:
        logger.warning(f"{label}: no Supabase details in {env_file}; dashboard buttons off for it")
        return None
    from supabase import create_client

    return Channel(create_client(url, secret), label)


# --------------------------------------------------------------------------- #
# Update (git pull)
# --------------------------------------------------------------------------- #


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def run_cmd(args: list[str], timeout: float) -> tuple[bool, str]:
    try:
        out = subprocess.run(
            args, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout, check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{' '.join(args)}: {exc}"
    text = (out.stdout or "") + (out.stderr or "")
    return out.returncode == 0, text.strip()


def git_update(runner: Callable[[list[str], float], tuple[bool, str]] = run_cmd
               ) -> tuple[bool, str, bool]:
    """git pull, then pip install if requirements.txt changed.
    Returns (ok, output, supervisor_code_changed)."""
    req_before = _digest(ROOT / "requirements.txt")
    sup_before = _digest(Path(__file__))
    ok, out = runner(["git", "pull", "--ff-only"], 180)
    lines = [f"git pull: {out or 'ok'}"]
    if not ok:
        return False, "\n".join(lines), False
    if _digest(ROOT / "requirements.txt") != req_before:
        ok, pip_out = runner(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 900,
        )
        lines.append("requirements.txt changed, pip install: " + ("ok" if ok else pip_out[-1500:]))
        if not ok:
            return False, "\n".join(lines), False
    return True, "\n".join(lines), _digest(Path(__file__)) != sup_before


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #


@dataclass
class Supervisor:
    bots: dict[str, BotProcess]
    channels: dict[str, Channel | None]
    updater: Callable[[], tuple[bool, str, bool]] = git_update
    poll_seconds: float = 5.0
    heartbeat_seconds: float = 15.0
    relaunch: bool = False
    _last_beat: float = field(default=0.0, repr=False)

    def save_state(self) -> None:
        try:
            STATE_FILE.parent.mkdir(exist_ok=True)
            STATE_FILE.write_text(json.dumps({k: b.wanted for k, b in self.bots.items()}))
        except OSError:
            pass

    def handle(self, key: str, cmd: dict[str, Any]) -> tuple[bool, str]:
        bot = self.bots[key]
        name = cmd.get("command")
        if name == "start":
            bot.wanted = True
            bot.crashes = 0
            bot.next_try = 0.0
            if bot.running:
                return True, f"{bot.label} bot was already running"
            bot.start()
            self.save_state()
            return True, f"{bot.label} bot started"
        if name == "stop":
            bot.wanted = False
            self.save_state()
            if not bot.running:
                return True, f"{bot.label} bot wasn't running; it stays stopped"
            clean = bot.stop()
            return True, f"{bot.label} bot stopped" + ("" if clean else " (had to force it)")
        if name == "restart":
            bot.wanted = True
            clean = bot.stop()
            bot.crashes = 0
            bot.start()
            self.save_state()
            return True, f"{bot.label} bot restarted" + ("" if clean else " (had to force the stop)")
        if name == "update":
            ok, out, sup_changed = self.updater()
            if not ok:
                return False, out + "\nNothing restarted; both bots still running the old code."
            restarted = []
            for other in self.bots.values():
                if other.wanted:
                    other.stop()
                    if not sup_changed:
                        other.start()
                    restarted.append(other.label)
            if sup_changed:
                self.relaunch = True
                return True, out + "\nThe supervisor itself was updated; starting a fresh copy of it, then the bots."
            names = " and ".join(restarted) or "no"
            return True, out + f"\nRestarted {names} bot{'s' if len(restarted) != 1 else ''} on the new code."
        return False, f"unknown command {name!r}"

    def poll_commands(self) -> None:
        for key, ch in self.channels.items():
            if ch is None:
                continue
            for cmd in ch.pending():
                cid = cmd.get("id")
                created = _parse_ts(cmd.get("created_at"))
                if not ch.claim(cid):
                    continue
                if created is not None and _now() - created > STALE_AFTER:
                    ch.finish(cid, False, "Ignored: pressed over 10 minutes ago "
                              "(the laptop was probably off). Press it again.")
                    continue
                logger.info(f"dashboard: {cmd.get('command')} {ch.label}")
                try:
                    ok, out = self.handle(key, cmd)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("command failed")
                    ok, out = False, f"error: {exc}"
                ch.finish(cid, ok, out)
                logger.info(f"dashboard: {cmd.get('command')} {ch.label}: {'done' if ok else 'FAILED'}")
                if self.relaunch:
                    return

    def heartbeat(self, force: bool = False) -> None:
        mono = time.monotonic()
        if not force and mono - self._last_beat < self.heartbeat_seconds:
            return
        self._last_beat = mono
        for key, ch in self.channels.items():
            if ch is not None:
                bot = self.bots[key]
                ch.heartbeat(bot.running, bot.note)

    def tick(self) -> None:
        for bot in self.bots.values():
            bot.check()
        self.poll_commands()
        self.heartbeat()

    def shutdown(self) -> None:
        for bot in self.bots.values():
            bot.stop()
        for ch in self.channels.values():
            if ch is not None:
                ch.heartbeat(False, None)

    def run(self) -> int:
        try:
            while not self.relaunch:
                try:
                    self.tick()
                except Exception:  # noqa: BLE001
                    logger.exception("supervisor tick failed; carrying on")
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            logger.info("Ctrl+C: stopping both bots")
        finally:
            self.shutdown()
        return RELAUNCH_EXIT if self.relaunch else 0


def load_wanted(resume: bool) -> dict[str, bool]:
    wanted = {k: True for k in BOTS}
    if resume:
        try:
            saved = json.loads(STATE_FILE.read_text())
            wanted.update({k: bool(v) for k, v in saved.items() if k in BOTS})
        except (OSError, ValueError):
            pass
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bot.supervisor")
    parser.add_argument("--resume", action="store_true",
                        help="keep bots stopped that were stopped from the dashboard")
    parser.add_argument("--only", choices=sorted(BOTS), action="append",
                        help="run only this bot (can repeat)")
    args = parser.parse_args()

    os.chdir(ROOT)
    logger.add("logs/supervisor.log", level="INFO", rotation="5 MB", retention=3,
               encoding="utf-8")
    wanted = load_wanted(args.resume)
    keys = args.only or list(BOTS)
    bots = {k: BotProcess(k, BOTS[k][0], wanted=wanted[k]) for k in keys}
    channels = {}
    for k in keys:
        try:
            channels[k] = make_channel(k)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{BOTS[k][0]}: Supabase not reachable ({exc}); buttons off for it")
            channels[k] = None
    sup = Supervisor(bots=bots, channels=channels)
    sup.save_state()
    logger.info(
        "supervisor running: " + ", ".join(
            f"{b.label} {'on' if b.wanted else 'kept stopped'}" for b in bots.values()
        ) + ". Ctrl+C here stops both bots. Logs are in the logs folder."
    )
    sys.exit(sup.run())


if __name__ == "__main__":
    main()
