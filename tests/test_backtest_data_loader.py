"""Tests for ``bot.backtest.data_loader``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bot.backtest.data_loader import (
    load_dukascopy_csv,
    validate_ohlc,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


DUKA_HEADER = "Gmt time,Open,High,Low,Close,Volume"
GOOD_DUKA = "\n".join([
    DUKA_HEADER,
    "01.05.2026 08:00:00.000,1900.00,1901.50,1899.50,1901.00,123",
    "01.05.2026 08:05:00.000,1901.00,1902.00,1900.50,1901.75,134",
    "01.05.2026 08:10:00.000,1901.75,1902.25,1901.00,1902.00,89",
])


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


class TestLoadDukascopyCSV:
    def test_loads_dukascopy_format(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "good.csv", GOOD_DUKA)
        df = load_dukascopy_csv(p)

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 3
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"
        assert df.index[0] == pd.Timestamp("2026-05-01T08:00:00", tz="UTC")
        assert df["close"].iloc[-1] == 1902.00

    def test_accepts_lowercase_timestamp_header(self, tmp_path: Path) -> None:
        # Some sources use ``timestamp`` directly.
        content = "\n".join([
            "timestamp,open,high,low,close",
            "2026-05-01T08:00:00Z,1900,1901,1899,1900.5",
            "2026-05-01T08:05:00Z,1900.5,1901.5,1900,1901",
        ])
        p = _write(tmp_path, "modern.csv", content)
        df = load_dukascopy_csv(p)
        assert len(df) == 2

    def test_volume_optional(self, tmp_path: Path) -> None:
        content = "\n".join([
            "Gmt time,Open,High,Low,Close",
            "01.05.2026 08:00:00.000,1900,1901,1899,1900.5",
        ])
        p = _write(tmp_path, "novol.csv", content)
        df = load_dukascopy_csv(p)
        assert "volume" not in df.columns

    def test_dukascopy_utc_header_with_utc_suffix_in_values(
        self, tmp_path: Path,
    ) -> None:
        # Dukascopy's 2024+ exports use "UTC" as the column header AND
        # embed " UTC" in each timestamp value.
        content = "\n".join([
            "UTC,Open,High,Low,Close,Volume",
            "09.11.2025 23:00:00.000 UTC,1900.00,1901.50,1899.50,1901.00,123",
            "09.11.2025 23:05:00.000 UTC,1901.00,1902.00,1900.50,1901.75,134",
        ])
        p = _write(tmp_path, "duka_utc.csv", content)
        df = load_dukascopy_csv(p)
        assert len(df) == 2
        assert df.index[0] == pd.Timestamp("2025-11-09T23:00:00", tz="UTC")
        assert df["close"].iloc[1] == 1901.75

    def test_dukascopy_utc_header_lowercase_value_suffix(
        self, tmp_path: Path,
    ) -> None:
        # Defensive: lowercase " utc" suffix should also strip cleanly.
        content = "\n".join([
            "utc,Open,High,Low,Close",
            "09.11.2025 23:00:00.000 utc,1900,1901,1899,1900.5",
        ])
        p = _write(tmp_path, "duka_utc_lower.csv", content)
        df = load_dukascopy_csv(p)
        assert len(df) == 1
        assert df.index.tz is not None

    def test_dukascopy_gmt_value_suffix_stripped(
        self, tmp_path: Path,
    ) -> None:
        # Some older clients emit " GMT" instead of " UTC". Same fix.
        content = "\n".join([
            "Gmt time,Open,High,Low,Close",
            "09.11.2025 23:00:00.000 GMT,1900,1901,1899,1900.5",
        ])
        p = _write(tmp_path, "duka_gmt_suffix.csv", content)
        df = load_dukascopy_csv(p)
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp("2025-11-09T23:00:00", tz="UTC")

    def test_dukascopy_local_time_header(self, tmp_path: Path) -> None:
        # Some Dukascopy export configurations label the column
        # "Local time" even when the timestamps are UTC. We accept the
        # header; the user is responsible for confirming the actual tz.
        content = "\n".join([
            "Local time,Open,High,Low,Close",
            "09.11.2025 23:00:00.000,1900,1901,1899,1900.5",
        ])
        p = _write(tmp_path, "duka_local.csv", content)
        df = load_dukascopy_csv(p)
        assert len(df) == 1
        assert df.index[0] == pd.Timestamp("2025-11-09T23:00:00", tz="UTC")

    def test_mt5_format_yyyy_dot_mm_dot_dd_no_seconds(
        self, tmp_path: Path,
    ) -> None:
        # MT5's default CSV export: YYYY.MM.DD HH:MM (no seconds).
        # Year-first detection must pick this up despite the dot
        # separator (which my old detection only handled for `-`/`/`).
        content = "\n".join([
            "timestamp,open,high,low,close,volume",
            "2026.04.30 23:55,4005.65,4006.20,4005.10,4005.80,134",
            "2026.05.01 00:00,4005.80,4006.50,4005.50,4006.20,98",
        ])
        p = _write(tmp_path, "mt5.csv", content)
        df = load_dukascopy_csv(p)
        assert len(df) == 2
        assert df.index[0] == pd.Timestamp("2026-04-30T23:55:00", tz="UTC")
        # Missing seconds default to :00 (pandas standard).
        assert df.index[0].second == 0
        assert df["close"].iloc[1] == 4006.20

    def test_mt5_format_with_seconds(self, tmp_path: Path) -> None:
        # Some MT5 configurations export seconds.
        content = "\n".join([
            "timestamp,open,high,low,close",
            "2026.04.30 23:55:00,4005.65,4006.20,4005.10,4005.80",
            "2026.04.30 23:55:30,4005.80,4006.00,4005.70,4005.90",
        ])
        p = _write(tmp_path, "mt5_sec.csv", content)
        df = load_dukascopy_csv(p)
        assert len(df) == 2
        assert df.index[0] == pd.Timestamp("2026-04-30T23:55:00", tz="UTC")
        assert df.index[1] == pd.Timestamp("2026-04-30T23:55:30", tz="UTC")

    def test_three_formats_produce_equivalent_frames(
        self, tmp_path: Path,
    ) -> None:
        """The same instant in MT5 / Dukascopy / ISO formats should
        produce identical rows after loading."""
        rows_mt5 = "\n".join([
            "timestamp,open,high,low,close",
            "2026.04.30 23:55,1900.0,1901.0,1899.0,1900.5",
            "2026.05.01 00:00,1900.5,1901.5,1900.0,1901.0",
        ])
        rows_duka = "\n".join([
            "Gmt time,Open,High,Low,Close",
            "30.04.2026 23:55:00.000 UTC,1900.0,1901.0,1899.0,1900.5",
            "01.05.2026 00:00:00.000 UTC,1900.5,1901.5,1900.0,1901.0",
        ])
        rows_iso = "\n".join([
            "timestamp,open,high,low,close",
            "2026-04-30T23:55:00Z,1900.0,1901.0,1899.0,1900.5",
            "2026-05-01T00:00:00Z,1900.5,1901.5,1900.0,1901.0",
        ])
        df_mt5 = load_dukascopy_csv(_write(tmp_path, "mt5.csv", rows_mt5))
        df_duka = load_dukascopy_csv(_write(tmp_path, "duka.csv", rows_duka))
        df_iso = load_dukascopy_csv(_write(tmp_path, "iso.csv", rows_iso))

        # Index, columns, and values all match.
        pd.testing.assert_index_equal(df_mt5.index, df_duka.index)
        pd.testing.assert_index_equal(df_mt5.index, df_iso.index)
        for col in ("open", "high", "low", "close"):
            pd.testing.assert_series_equal(
                df_mt5[col], df_duka[col], check_names=False,
            )
            pd.testing.assert_series_equal(
                df_mt5[col], df_iso[col], check_names=False,
            )

    def test_sorts_by_timestamp(self, tmp_path: Path) -> None:
        # Out-of-order rows should be sorted ascending.
        content = "\n".join([
            DUKA_HEADER,
            "01.05.2026 08:10:00.000,1901,1902,1900.5,1901.75,89",
            "01.05.2026 08:00:00.000,1900,1901.5,1899.5,1901,123",
            "01.05.2026 08:05:00.000,1901,1902,1900.5,1901.75,134",
        ])
        p = _write(tmp_path, "unsorted.csv", content)
        df = load_dukascopy_csv(p)
        assert list(df.index) == sorted(df.index)

    def test_drops_duplicate_timestamps_first_wins(
        self, tmp_path: Path,
    ) -> None:
        # Two rows at the same timestamp; first one's close wins.
        content = "\n".join([
            DUKA_HEADER,
            "01.05.2026 08:00:00.000,1900,1901.5,1899.5,1900.50,1",
            "01.05.2026 08:00:00.000,1900,1901.5,1899.5,1900.99,1",  # dup
            "01.05.2026 08:05:00.000,1901,1902,1900.5,1901.75,1",
        ])
        p = _write(tmp_path, "dup.csv", content)
        df = load_dukascopy_csv(p)
        assert len(df) == 2
        assert df["close"].iloc[0] == 1900.50  # first row wins

    def test_naive_timestamps_treated_as_utc(self, tmp_path: Path) -> None:
        content = "\n".join([
            "timestamp,open,high,low,close",
            "2026-05-01 08:00:00,1900,1901,1899,1900.5",
        ])
        p = _write(tmp_path, "naive.csv", content)
        df = load_dukascopy_csv(p)
        assert str(df.index.tz) == "UTC"

    def test_already_tz_aware_converted_to_utc(self, tmp_path: Path) -> None:
        # Timestamps with non-UTC offset get converted (not relabelled).
        content = "\n".join([
            "timestamp,open,high,low,close",
            "2026-05-01T10:00:00+02:00,1900,1901,1899,1900.5",  # = 08:00 UTC
        ])
        p = _write(tmp_path, "tz.csv", content)
        df = load_dukascopy_csv(p)
        assert df.index[0] == pd.Timestamp("2026-05-01T08:00:00", tz="UTC")


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #


class TestErrorPaths:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dukascopy_csv(tmp_path / "nope.csv")

    def test_missing_required_columns(self, tmp_path: Path) -> None:
        # No 'close' column.
        content = "\n".join([
            "Gmt time,Open,High,Low",
            "01.05.2026 08:00:00.000,1900,1901,1899",
        ])
        p = _write(tmp_path, "bad.csv", content)
        with pytest.raises(ValueError, match="missing required"):
            load_dukascopy_csv(p)

    def test_missing_timestamp_column(self, tmp_path: Path) -> None:
        content = "\n".join([
            "open,high,low,close",
            "1900,1901,1899,1900.5",
        ])
        p = _write(tmp_path, "nots.csv", content)
        with pytest.raises(ValueError, match="missing timestamp"):
            load_dukascopy_csv(p)

    def test_unparseable_timestamp(self, tmp_path: Path) -> None:
        content = "\n".join([
            DUKA_HEADER,
            "garbage_not_a_date,1900,1901,1899,1900.5,1",
        ])
        p = _write(tmp_path, "badts.csv", content)
        with pytest.raises(ValueError, match="Unparseable timestamp"):
            load_dukascopy_csv(p)

    def test_high_lt_low_rejected(self, tmp_path: Path) -> None:
        content = "\n".join([
            DUKA_HEADER,
            "01.05.2026 08:00:00.000,1900,1899,1901,1900.5,1",  # high<low
        ])
        p = _write(tmp_path, "broken.csv", content)
        with pytest.raises(ValueError, match="high < low"):
            load_dukascopy_csv(p)

    def test_close_outside_range_rejected(self, tmp_path: Path) -> None:
        content = "\n".join([
            DUKA_HEADER,
            "01.05.2026 08:00:00.000,1900,1901,1899,1905,1",  # close > high
        ])
        p = _write(tmp_path, "outrange.csv", content)
        with pytest.raises(ValueError, match="outside"):
            load_dukascopy_csv(p)

    def test_nan_price_rejected(self, tmp_path: Path) -> None:
        content = "\n".join([
            DUKA_HEADER,
            "01.05.2026 08:00:00.000,1900,1901,1899,,1",  # close blank
            "01.05.2026 08:05:00.000,1901,1902,1900.5,1901.75,1",
        ])
        p = _write(tmp_path, "nanp.csv", content)
        with pytest.raises(ValueError, match="NaN"):
            load_dukascopy_csv(p)


# --------------------------------------------------------------------------- #
# validate_ohlc on prebuilt DataFrames
# --------------------------------------------------------------------------- #


class TestValidateOHLC:
    def _make(self, **kwargs) -> pd.DataFrame:
        idx = pd.DatetimeIndex(
            ["2026-05-01T08:00:00", "2026-05-01T08:05:00"], tz="UTC",
        )
        base = {"open": [1900.0, 1901.0], "high": [1901.0, 1902.0],
                "low": [1899.0, 1900.0], "close": [1900.5, 1901.5]}
        base.update(kwargs)
        return pd.DataFrame(base, index=idx)

    def test_valid_passes(self) -> None:
        validate_ohlc(self._make())

    def test_naive_index_rejected(self) -> None:
        df = self._make()
        df.index = df.index.tz_localize(None)
        with pytest.raises(ValueError, match="tz-aware"):
            validate_ohlc(df)

    def test_missing_column_rejected(self) -> None:
        df = self._make().drop(columns=["close"])
        with pytest.raises(ValueError, match="missing required"):
            validate_ohlc(df)
