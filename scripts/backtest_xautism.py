"""Run the XAUTISM backtest on MT5-exported bars.

    python scripts/backtest_xautism.py M5.csv [M1.csv] [--start 2026-07-08] [--end 2026-10-22] [--out trades.csv]

M5.csv / M1.csv are MT5 "Export bars" files (tab separated). Several M5
files can be joined with commas; overlapping bars are de-duplicated.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from loguru import logger

from bot.backtest.xautism import XautismBacktest, load_mt5_csv, summarise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("m5", help="M5 CSV(s), comma separated")
    ap.add_argument("m1", nargs="?", help="M1 CSV (optional, for stacked M1 zones + intrabar path)")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default="xautism_trades.csv")
    a = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"].startswith("bot.backtest"))

    frames = [load_mt5_csv(p) for p in a.m5.split(",")]
    m5 = pd.concat(frames).sort_index()
    m5 = m5[~m5.index.duplicated(keep="first")]
    m1 = load_mt5_csv(a.m1) if a.m1 else None

    res = XautismBacktest(m5, m1).run(
        pd.Timestamp(a.start, tz="UTC") if a.start else None,
        pd.Timestamp(a.end, tz="UTC") if a.end else None,
    )
    df = res.frame()
    df.to_csv(a.out, index=False)
    print(summarise(df))
    print("skips:", res.skips)
    print("days stopped by risk gates:", res.blocked_days)
    print(f"trades written to {a.out}")


if __name__ == "__main__":
    main()
