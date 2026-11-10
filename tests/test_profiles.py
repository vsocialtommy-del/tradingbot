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
        assert cfg.xautism_entries.m5_close_exit is True    # M5 keeps its close exit
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
        assert cfg.xautism_entries.m5_close_exit is False   # 30% SL only
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


class TestM1RiskSwitch:
    @pytest.mark.parametrize("text,pct,daily_pct", [("10%", 0.10, 0.30), ("5%", 0.05, 0.15), (" 5 % ", 0.05, 0.15)])
    def test_percent(self, text, pct, daily_pct) -> None:
        cfg, _ = build_profile("m1", text)
        assert cfg.risk.risk_pct == pytest.approx(pct)
        assert cfg.risk.max_daily_loss_pct == pytest.approx(daily_pct)
        assert cfg.risk.max_losses_per_day == 2

    @pytest.mark.parametrize("text", ["100", "£100"])
    def test_fixed_100_like_m5(self, text) -> None:
        cfg, _ = build_profile("m1", text)
        m5, _ = build_profile("m5")
        assert cfg.risk.risk_pct is None and cfg.risk.max_daily_loss_pct is None
        assert cfg.risk.risk_per_trade == 100.0
        assert cfg.risk.max_daily_loss == 300.0 == m5.risk.max_daily_loss
        assert cfg.risk.max_losses_per_day == 2
        # still the M1 bot otherwise
        assert cfg.ohlc_timeframe == "M1" and cfg.xautism_entries.runner_keep_frac == 0.5

    def test_default_is_10_percent(self) -> None:
        assert build_profile("m1")[0].risk.risk_pct == pytest.approx(0.10)

    @pytest.mark.parametrize("text", ["ten", "0%", "50%", "-5", ""])
    def test_bad_values(self, text) -> None:
        if text == "":
            assert build_profile("m1", text)[0].risk.risk_pct == pytest.approx(0.10)
            return
        with pytest.raises(ValueError, match="isn't valid"):
            build_profile("m1", text)

    def test_m5_refuses_the_switch(self) -> None:
        with pytest.raises(ValueError, match="M1 bot only"):
            build_profile("m5", "5%")


class TestRiskFromEnvAndCommand:
    def _run(self, mocker, argv, tmp_path, monkeypatch, env_text="MT5_LOGIN=1\n"):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env.m1").write_text(env_text)
        mocker.patch.object(sys, "argv", ["bot.main", *argv])
        mocker.patch.object(main_mod.MT5Connector, "from_env", return_value=MagicMock())
        mocker.patch.object(main_mod.SupabaseLogger, "from_env", return_value=MagicMock())
        mocker.patch.object(main_mod.signal, "signal")
        bot_cls = mocker.patch.object(main_mod, "Bot")
        main_mod.main()
        return bot_cls.call_args.kwargs["config"].risk

    def test_risk_from_env_file(self, mocker, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("RISK", raising=False)
        r = self._run(mocker, ["--bot", "m1"], tmp_path, monkeypatch, "RISK=5%\n")
        assert r.risk_pct == pytest.approx(0.05)
        monkeypatch.delenv("RISK", raising=False)

    def test_command_overrides_env_file(self, mocker, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("RISK", raising=False)
        r = self._run(mocker, ["--bot", "m1", "--risk", "100"], tmp_path, monkeypatch, "RISK=5%\n")
        assert r.risk_pct is None and r.risk_per_trade == 100.0
        monkeypatch.delenv("RISK", raising=False)

    def test_bad_value_stops_with_message(self, mocker, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("RISK", raising=False)
        with pytest.raises(SystemExit, match="isn't valid"):
            self._run(mocker, ["--bot", "m1", "--risk", "lots"], tmp_path, monkeypatch)

    def test_m5_with_risk_refused(self, mocker) -> None:
        mocker.patch.object(sys, "argv", ["bot.main", "--risk", "5%"])
        mocker.patch.object(main_mod, "load_dotenv")
        with pytest.raises(SystemExit, match="M1 bot only"):
            main_mod.main()


class TestM1MaxLosses:
    def test_default_two(self) -> None:
        cfg, _ = build_profile("m1", "1%")
        assert cfg.risk.max_losses_per_day == 2
        assert cfg.risk.max_daily_loss_pct == pytest.approx(0.03)

    @pytest.mark.parametrize("text,n", [("10", 10), (" 5 ", 5), (3, 3), ("1", 1)])
    def test_switch(self, text, n) -> None:
        cfg, _ = build_profile("m1", "1%", text)
        assert cfg.risk.max_losses_per_day == n
        # daily cap is one trade more than the losses allowed
        assert cfg.risk.max_daily_loss_pct == pytest.approx(0.01 * (n + 1))

    def test_fixed_amount_cap_scales(self) -> None:
        cfg, _ = build_profile("m1", "100", "10")
        assert cfg.risk.max_losses_per_day == 10
        assert cfg.risk.max_daily_loss == pytest.approx(1100.0)

    def test_empty_means_default(self) -> None:
        assert build_profile("m1", "1%", "")[0].risk.max_losses_per_day == 2

    @pytest.mark.parametrize("text", ["0", "51", "two", "2.5", "-1"])
    def test_bad_values(self, text) -> None:
        with pytest.raises(ValueError, match="MAX_LOSSES"):
            build_profile("m1", "1%", text)

    def test_m5_refuses(self) -> None:
        with pytest.raises(ValueError, match="M1 bot only"):
            build_profile("m5", None, "10")

    def test_from_env_file(self, mocker, tmp_path, monkeypatch) -> None:
        for k in ("RISK", "MAX_LOSSES"):
            monkeypatch.delenv(k, raising=False)
        r = TestRiskFromEnvAndCommand()._run(
            mocker, ["--bot", "m1"], tmp_path, monkeypatch, "RISK=1%\nMAX_LOSSES=10\n",
        )
        assert r.max_losses_per_day == 10
        for k in ("RISK", "MAX_LOSSES"):
            monkeypatch.delenv(k, raising=False)

    def test_command_overrides_env(self, mocker, tmp_path, monkeypatch) -> None:
        for k in ("RISK", "MAX_LOSSES"):
            monkeypatch.delenv(k, raising=False)
        r = TestRiskFromEnvAndCommand()._run(
            mocker, ["--bot", "m1", "--max-losses", "4"], tmp_path, monkeypatch,
            "MAX_LOSSES=10\n",
        )
        assert r.max_losses_per_day == 4
        for k in ("RISK", "MAX_LOSSES"):
            monkeypatch.delenv(k, raising=False)

    def test_bad_env_stops_with_message(self, mocker, tmp_path, monkeypatch) -> None:
        for k in ("RISK", "MAX_LOSSES"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(SystemExit, match="MAX_LOSSES"):
            TestRiskFromEnvAndCommand()._run(
                mocker, ["--bot", "m1"], tmp_path, monkeypatch, "MAX_LOSSES=lots\n",
            )
        monkeypatch.delenv("MAX_LOSSES", raising=False)

    def test_m5_command_refused(self, mocker) -> None:
        mocker.patch.object(sys, "argv", ["bot.main", "--max-losses", "10"])
        mocker.patch.object(main_mod, "load_dotenv")
        with pytest.raises(SystemExit, match="M1 bot only"):
            main_mod.main()

    def test_m5_ignores_max_losses_env(self, mocker, monkeypatch) -> None:
        monkeypatch.setenv("MAX_LOSSES", "10")
        mocker.patch.object(sys, "argv", ["bot.main"])
        mocker.patch.object(main_mod, "load_dotenv")
        mocker.patch.object(main_mod.MT5Connector, "from_env", return_value=MagicMock())
        mocker.patch.object(main_mod.SupabaseLogger, "from_env", return_value=MagicMock())
        mocker.patch.object(main_mod.signal, "signal")
        bot_cls = mocker.patch.object(main_mod, "Bot")
        main_mod.main()
        assert bot_cls.call_args.kwargs["config"].risk.max_losses_per_day == 2

