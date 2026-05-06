"""MT5 connection smoke test.

Run on the Hetzner VPS (or any host with MT5 desktop installed and
logged into the Vantage demo account):

    python scripts/test_mt5_connection.py

What it does:
  1. Loads .env (must contain MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH).
  2. Connects via MT5Connector.from_env().
  3. Prints account info (login, server, currency, balance, leverage).
  4. Fetches the last 100 1H candles for XAUUSD and prints first / last bar.
  5. Disconnects cleanly even if any step fails.

Exit codes:
  0 — success
  1 — any failure (full traceback printed via loguru)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from bot.execution.mt5_connector import MT5Connector  # noqa: E402


def main() -> int:
    logger.info("MT5 connection smoke test starting")

    connector = MT5Connector.from_env()
    try:
        connector.connect()

        info = connector.get_account_info()
        logger.info("=== Account ===")
        logger.info(f"  Login:    {info['login']}")
        logger.info(f"  Server:   {info['server']}")
        logger.info(f"  Currency: {info['currency']}")
        logger.info(f"  Balance:  {info['balance']:.2f}")
        logger.info(f"  Equity:   {info['equity']:.2f}")
        logger.info(f"  Leverage: 1:{info['leverage']}")

        symbol = "XAUUSD"
        df = connector.get_ohlc(symbol, "H1", 100)
        logger.info(f"=== OHLC: {symbol} H1 (last {len(df)} bars) ===")
        logger.info(f"  First bar @ {df.index[0]}:\n{df.iloc[0].to_dict()}")
        logger.info(f"  Last bar  @ {df.index[-1]}:\n{df.iloc[-1].to_dict()}")

        tick = connector.get_current_price(symbol)
        logger.info(
            f"=== Live tick: bid={tick['bid']} ask={tick['ask']} "
            f"spread_pts={(tick['ask'] - tick['bid']) * 100:.1f} time={tick['time']}"
        )

    except Exception:
        logger.exception("Smoke test failed")
        return 1
    finally:
        connector.disconnect()

    logger.success("Smoke test complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
