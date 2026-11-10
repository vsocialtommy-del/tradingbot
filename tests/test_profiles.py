"""M5 / M1 bot profiles, % of balance risk and the ``--bot`` option."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import bot.main as main_mod
from bot.main import Bot, BotLoopConfig
from bot.profiles import ENV_FILES, build_profile
from bot.risk.xautism_risk import RiskConfig, effective_risk
from bot.strategy.pipeline import StrategyPipelineConfig


class TestProfiles:
    def test_m5_is_the_current_bot_unchanged(self) -> None:
        cfg, pipe = build_profile("m5")
        base = BotLoopConfig()
        assert cfg.bot_name == "M5"
        assert cfg.ohlc_timeframe == "M5"
        assert cfg.risk == base.risk                       # £100, 2 losses, -£300
        assert cfg.direction_gate == base.direction_gate   # M15 + fresh H1
        assert cfg.xautism_entries == base.xautism_entries  # 75/25, 4 pt trail
        assert pipe == StrategyPipelineConfig()

    def test_m1_settings(self) -> None:
        cfg, pipe = build_profile("m1")
        assert cfg.bot_name == "M1" and cfg.ohlc_timeframe == "M1"
        assert pipe.zone_min_size_points == 1.0
        assert cfg.direction_gate.trend_timeframe == "5min"
        assert cfg.direction_gate.htf_timeframe == "15min"
        assert cfg.xautism_entries.runner_enabled
        assert cfg.xautism_entries.runner_keep_frac == 0.5
        assert cfg.xautism_entries.runner_trail_points == 2.0
        assert cfg.risk.risk_pct == 0.10
        assert cfg.risk.max_daily_loss_pct == 0.30
        assert cfg.risk.max_losses_per_day == 2
        # everything else is the same as the M5 bot
        base = BotLoopConfig()
        assert cfg.levels == base.levels
        assert cfg.enable_news_filter and cfg.enable_news_close
        assert cfg.max_simultaneous_setups == 1

    def test_unknown_bot(self) -> None:
        with pytest.raises(ValueError):
            build_profile("m15")

    def test_env_files(self) -> None:
        assert ENV_FILES == {"m5": ".env", "m1": ".env.m1"}


class TestPercentRisk:
    def test_fixed_pound_config_unchanged(self) -> None:
        cfg = RiskConfig()
        assert effective_risk(cfg, 12345.0) is cfg

    def test_percent_of_balance(self) -> None:
        cfg = RiskConfig(risk_pct=0.10, max_daily_loss_pct=0.30)
        eff = effective_risk(cfg, 1000.0)
        assert eff.risk_per_trade == pytest.approx(100.0)
        assert eff.max_daily_loss == pytest.approx(300.0)
        assert effective_risk(cfg, 2500.0).risk_per_trade == pytest.approx(250.0)

    def test_bot_uses_live_balance(self, mocker: MockerFixture) -> None:
        cfg, _ = build_profile("m1")
        mt5 = mocker.MagicMock()
        mt5.get_balance.return_value = 1500.0
        b = Bot.__new__(Bot)
        b.config, b.mt5 = cfg, mt5
        r = b._risk_now()
        assert r.risk_per_trade == pytest.approx(150.0)
        assert r.max_daily_loss == pytest.approx(450.0)

    def test_balance_failure_falls_back(self, mocker: MockerFixture) -> None:
        cfg, _ = build_profile("m1")
        mt5 = mocker.MagicMock()
        mt5.get_balance.side_effect = RuntimeError("no MT5")
        b = Bot.__new__(Bot)
        b.config, b.mt5 = cfg, mt5
        assert b._risk_now() is cfg.risk

    def test_m5_bot_never_asks_for_balance(self, mocker: MockerFixture) -> None:
        cfg, _ = build_profile("m5")
        mt5 = mocker.MagicMock()
        b = Bot.__new__(Bot)
        b.config, b.mt5 = cfg, mt5
        assert b._risk_now() is cfg.risk
        mt5.get_balance.assert_not_called()


class TestMainEntry:
    def _run(self, mocker: MockerFixture, argv: list[str]) -> MagicMock:
        mocker.patch.object(sys, "argv", ["bot.main", *argv])
        mocker.patch.object(main_mod.MT5Connector, "from_env", return_value=MagicMock())
        mocker.patch.object(main_mod.SupabaseLogger, "from_env", return_value=MagicMock())
        bot_cls = mocker.patch.object(main_mod, "Bot")
        mocker.patch.object(main_mod.signal, "signal")
        main_mod.main()
        return bot_cls

    def test_default_is_m5_with_env(self, mocker: MockerFixture) -> None:
        dot = mocker.patch.object(main_mod, "load_dotenv")
        bot_cls = self._run(mocker, [])
        dot.assert_called_once_with()
        assert bot_cls.call_args.kwargs["config"].bot_name == "M5"

    def test_m1_loads_env_m1(self, mocker: MockerFixture, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env.m1").write_text("MT5_LOGIN=1\n")
        dot = mocker.patch.object(main_mod, "load_dotenv")
        bot_cls = self._run(mocker, ["--bot", "m1"])
        assert str(dot.call_args.args[0]) == ".env.m1"
        assert dot.call_args.kwargs == {"override": True}
        kw = bot_cls.call_args.kwargs
        assert kw["config"].ohlc_timeframe == "M1"
        assert kw["strategy_pipeline_config"].zone_min_size_points == 1.0

    def test_m1_without_env_file_stops_with_a_clear_message(
        self, mocker: MockerFixture, tmp_path, monkeypatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mocker.patch.object(sys, "argv", ["bot.main", "--bot", "m1"])
        with pytest.raises(SystemExit, match=r"\.env\.m1 not found"):
            main_mod.main()
