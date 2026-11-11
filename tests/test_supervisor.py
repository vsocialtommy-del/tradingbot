"""Laptop supervisor (dashboard buttons, crash restarts, heartbeat)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from bot import supervisor as sv
from bot.supervisor import (
    ALREADY_RUNNING_EXIT,
    BotProcess,
    Channel,
    Supervisor,
    git_update,
)

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeProc:
    _next_pid = 100

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.args, self.kwargs = args, kwargs
        self.code: int | None = None
        self.signals: list[Any] = []
        self.ignore_stop = False
        self.killed = False

    def poll(self) -> int | None:
        return self.code

    def send_signal(self, sig: Any) -> None:
        self.signals.append(sig)
        if not self.ignore_stop:
            self.code = 0

    def wait(self, timeout: float | None = None) -> int:
        if self.code is None:
            raise subprocess.TimeoutExpired("bot", timeout or 0)
        return self.code

    def kill(self) -> None:
        self.killed = True
        self.code = -9


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class FakeQuery:
    def __init__(self, db: FakeDB, table: str) -> None:
        self.db, self.table = db, table
        self.filters: list[tuple[str, Any]] = []
        self.op = "select"
        self.payload: dict[str, Any] | None = None

    def select(self, *_a: Any) -> FakeQuery:
        return self

    def order(self, *_a: Any, **_k: Any) -> FakeQuery:
        return self

    def limit(self, *_a: Any) -> FakeQuery:
        return self

    def eq(self, col: str, val: Any) -> FakeQuery:
        self.filters.append((col, val))
        return self

    def update(self, payload: dict[str, Any]) -> FakeQuery:
        self.op, self.payload = "update", payload
        return self

    def insert(self, payload: dict[str, Any]) -> FakeQuery:
        self.op, self.payload = "insert", payload
        return self

    def execute(self) -> SimpleNamespace:
        if self.db.fail:
            raise self.db.fail
        rows = self.db.tables.setdefault(self.table, [])
        if self.op == "insert":
            rows.append(dict(self.payload or {}))
            return SimpleNamespace(data=[self.payload])
        hit = [r for r in rows if all(r.get(c) == v for c, v in self.filters)]
        if self.op == "update":
            for r in hit:
                r.update(self.payload or {})
        return SimpleNamespace(data=hit)


class FakeDB:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.fail: Exception | None = None

    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)


def _bot(key: str = "m5", **kw: Any) -> BotProcess:
    return BotProcess(key, key.upper(), popen=FakeProc, clock=kw.pop("clock", Clock()), **kw)


def _cmd(db: FakeDB, cid: int, bot: str, command: str, age: timedelta = timedelta(seconds=5)) -> None:
    db.tables.setdefault("bot_commands", []).append({
        "id": cid, "bot": bot, "command": command, "status": "pending",
        "created_at": (datetime.now(timezone.utc) - age).isoformat(),
    })


@pytest.fixture(autouse=True)
def _tmp_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sv, "ROOT", tmp_path)
    monkeypatch.setattr(sv, "STATE_FILE", tmp_path / "logs" / "state.json")


# --------------------------------------------------------------------------- #
# BotProcess
# --------------------------------------------------------------------------- #


class TestBotProcess:
    def test_start_runs_the_right_bot_quietly(self) -> None:
        b = _bot("m1")
        b.start()
        assert b.running
        args = b.proc.args[0]
        assert args[-3:] == ["bot.main", "--bot", "m1"]
        assert b.proc.kwargs["env"]["LOGURU_LEVEL"] == "INFO"

    def test_stop_is_graceful(self) -> None:
        b = _bot()
        b.start()
        proc = b.proc
        assert b.stop() is True
        assert proc.signals and not proc.killed and not b.running

    def test_stop_forces_after_timeout(self) -> None:
        b = _bot()
        b.start()
        b.proc.ignore_stop = True
        proc = b.proc
        assert b.stop(timeout=0.01) is False
        assert proc.killed

    def test_crash_restarts_with_backoff(self) -> None:
        clock = Clock()
        b = _bot(clock=clock)
        b.start()
        first = b.proc
        first.code = 1                     # crashed
        b.check()
        assert b.proc is None and "exit code 1" in b.note
        clock.t += 5
        b.check()
        assert b.proc is None              # still waiting (10 s)
        clock.t += 6
        b.check()
        assert b.running and b.proc is not first
        b.proc.code = 1                    # crashes again: waits longer
        b.check()
        clock.t += 11
        b.check()
        assert b.proc is None
        clock.t += 10
        b.check()
        assert b.running

    def test_already_running_elsewhere_waits_a_minute(self) -> None:
        clock = Clock()
        b = _bot(clock=clock)
        b.start()
        b.proc.code = ALREADY_RUNNING_EXIT
        b.check()
        assert "another window" in b.note and b.crashes == 0
        clock.t += 30
        b.check()
        assert b.proc is None
        clock.t += 31
        b.check()
        assert b.running

    def test_stopped_from_dashboard_stays_stopped(self) -> None:
        b = _bot(wanted=False)
        b.check()
        assert b.proc is None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def _sup(updater=None) -> tuple[Supervisor, FakeDB, FakeDB]:
    m5db, m1db = FakeDB(), FakeDB()
    bots = {"m5": _bot("m5"), "m1": _bot("m1")}
    for b in bots.values():
        b.start()
    sup = Supervisor(
        bots=bots,
        channels={"m5": Channel(m5db, "M5"), "m1": Channel(m1db, "M1")},
        updater=updater or (lambda: (True, "git pull: Already up to date.", False)),
    )
    return sup, m5db, m1db


class TestCommands:
    def test_stop_then_start(self) -> None:
        sup, m5db, _ = _sup()
        _cmd(m5db, 1, "M5", "stop")
        sup.poll_commands()
        row = m5db.tables["bot_commands"][0]
        assert row["status"] == "done" and "stopped" in row["output"]
        assert not sup.bots["m5"].running and sup.bots["m1"].running
        sup.bots["m5"].check()
        assert not sup.bots["m5"].running      # not brought back by the crash check
        _cmd(m5db, 2, "M5", "start")
        sup.poll_commands()
        assert sup.bots["m5"].running

    def test_restart_only_that_bot(self) -> None:
        sup, _, m1db = _sup()
        m5_proc, m1_proc = sup.bots["m5"].proc, sup.bots["m1"].proc
        _cmd(m1db, 1, "M1", "restart")
        sup.poll_commands()
        assert sup.bots["m1"].proc is not m1_proc and sup.bots["m1"].running
        assert sup.bots["m5"].proc is m5_proc

    def test_update_restarts_both(self) -> None:
        sup, m5db, _ = _sup()
        before = {k: b.proc for k, b in sup.bots.items()}
        _cmd(m5db, 1, "M5", "update")
        sup.poll_commands()
        row = m5db.tables["bot_commands"][0]
        assert row["status"] == "done" and "M5 and M1" in row["output"]
        assert all(b.running and b.proc is not before[k] for k, b in sup.bots.items())

    def test_failed_pull_restarts_nothing(self) -> None:
        sup, m5db, _ = _sup(updater=lambda: (False, "git pull: conflict", False))
        before = {k: b.proc for k, b in sup.bots.items()}
        _cmd(m5db, 1, "M5", "update")
        sup.poll_commands()
        row = m5db.tables["bot_commands"][0]
        assert row["status"] == "failed" and "old code" in row["output"]
        assert all(b.proc is before[k] for k, b in sup.bots.items())

    def test_update_of_supervisor_asks_for_a_relaunch(self) -> None:
        sup, m5db, _ = _sup(updater=lambda: (True, "git pull: ok", True))
        _cmd(m5db, 1, "M5", "update")
        sup.poll_commands()
        assert sup.relaunch
        assert not any(b.running for b in sup.bots.values())

    def test_old_commands_are_ignored(self) -> None:
        sup, m5db, _ = _sup()
        proc = sup.bots["m5"].proc
        _cmd(m5db, 1, "M5", "stop", age=timedelta(hours=3))
        sup.poll_commands()
        row = m5db.tables["bot_commands"][0]
        assert row["status"] == "failed" and "10 minutes" in row["output"]
        assert sup.bots["m5"].proc is proc and proc.poll() is None

    def test_other_bots_commands_not_taken(self) -> None:
        sup, m5db, _ = _sup()
        _cmd(m5db, 1, "M1", "stop")      # wrong project for M1: left alone
        sup.poll_commands()
        assert m5db.tables["bot_commands"][0]["status"] == "pending"

    def test_missing_tables_dont_break_anything(self) -> None:
        sup, m5db, m1db = _sup()
        m5db.fail = m1db.fail = RuntimeError("PGRST205 Could not find the table")
        sup.poll_commands()
        sup.heartbeat(force=True)
        assert all(b.running for b in sup.bots.values())


class TestHeartbeat:
    def test_heartbeat_creates_then_updates_row(self) -> None:
        sup, m5db, _ = _sup()
        sup.heartbeat(force=True)
        rows = m5db.tables["bot_status"]
        assert len(rows) == 1 and rows[0]["process_running"] is True
        assert rows[0]["updated_at"].startswith("1970")   # bot hasn't written yet
        sup.bots["m5"].stop()
        sup.bots["m5"].note = "stopped by itself"
        sup.heartbeat(force=True)
        assert len(rows) == 1 and rows[0]["process_running"] is False
        assert rows[0]["supervisor_note"] == "stopped by itself"


class TestGitUpdate:
    def test_pip_only_when_requirements_change(self, tmp_path) -> None:
        req = tmp_path / "requirements.txt"
        req.write_text("a\n")
        calls: list[list[str]] = []

        def runner(args, timeout):
            calls.append(args)
            if args[0] == "git":
                req.write_text("a\nb\n")     # the pull changed it
            return True, "ok"

        ok, out, sup_changed = git_update(runner)
        assert ok and not sup_changed
        assert any("pip" in a for a in calls[1])
        assert "pip install: ok" in out

        calls.clear()
        ok, _out, _ = git_update(lambda a, t: (calls.append(a) or True, "Already up to date."))
        assert ok and len(calls) == 1

    def test_failed_pull_stops_there(self) -> None:
        ok, out, _ = git_update(lambda a, t: (False, "fatal: Not possible to fast-forward"))
        assert not ok and "fast-forward" in out


class TestStateFile:
    def test_resume_keeps_stopped_bots_stopped(self) -> None:
        sup, m5db, _ = _sup()
        _cmd(m5db, 1, "M5", "stop")
        sup.poll_commands()
        assert sv.load_wanted(resume=True) == {"m5": False, "m1": True}
        assert sv.load_wanted(resume=False) == {"m5": True, "m1": True}


class TestSingleInstanceLock:
    def test_second_copy_refused(self, tmp_path) -> None:
        from bot.main import _single_instance_lock

        first = _single_instance_lock("m5", folder=str(tmp_path))
        assert first is not None
        import subprocess as sp
        import sys
        code = (
            "import sys; from bot.main import _single_instance_lock as L; "
            f"sys.exit(0 if L('m5', folder={str(tmp_path)!r}) is None else 1)"
        )
        assert sp.run([sys.executable, "-c", code], check=False).returncode == 0
        first.close()
        assert sp.run([sys.executable, "-c", code.replace("is None", "is not None")], check=False).returncode == 0
