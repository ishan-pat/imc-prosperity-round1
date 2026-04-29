"""
Round 4 backtest. Reuses the Round-3 simulation engine, just swaps the data
directory + default trader path.

Round 4 historical data is days 1, 2, 3 only (no day 0).

IMPORTANT: the trader hard-codes LIVE_TTE_DAY_START = 4 (live submission
TTE).  The historical replay days had TTE = 8, 7, 6.  We monkey-patch
LIVE_TTE_DAY_START → 8 just for this backtest run so the trader's σ-implied
voucher fair values match the market data being replayed.  Without this,
the trader systematically mis-prices vouchers vs the historical book.

Usage (from repo root):
    python3 -m backtesting.round4_backtest
    python3 -m backtesting.round4_backtest --passive-share 0.3
"""
import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtesting import round3_backtest as engine  # noqa: E402

# Override engine paths to point at round 4 data.
engine.DATA_DIR = REPO_ROOT / "data" / "round4"
engine.PRICE_FILES = [
    (engine.DATA_DIR / "prices_round_4_day_1.csv", 1),
    (engine.DATA_DIR / "prices_round_4_day_2.csv", 2),
    (engine.DATA_DIR / "prices_round_4_day_3.csv", 3),
]
engine.TRADE_FILES = {
    1: engine.DATA_DIR / "trades_round_4_day_1.csv",
    2: engine.DATA_DIR / "trades_round_4_day_2.csv",
    3: engine.DATA_DIR / "trades_round_4_day_3.csv",
}


def load_trader_for_backtest(path: str):
    """Load trader.py and patch constants for historical replay:
      - LIVE_TTE_DAY_START = 8 to match the round-4 day 1 historical TTE
        (the trader's day-rollover logic decrements by 1 each day).
      - DAY_LENGTH_TS / EOD windows back to historical 1M-tick days.  The
        live trader uses 100K-tick days, but the historical CSVs span
        timestamps 0…999,900 per day.  Without this patch the trader would
        EOD-flatten at ts=95K and stop trading the rest of the day."""
    spec = importlib.util.spec_from_file_location("trader_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.LIVE_TTE_DAY_START = 8
    mod.DAY_LENGTH_TS = 1_000_000
    mod.EOD_FLATTEN_VOUCHERS_START_TS = 950_000
    mod.EOD_FLATTEN_LINEAR_START_TS = 950_000
    return mod.Trader()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trader",
        default=str(REPO_ROOT / "submissions" / "round4" / "trader.py"),
    )
    parser.add_argument("--passive-share", type=float, default=0.5)
    args = parser.parse_args()
    trader = load_trader_for_backtest(args.trader)
    engine.run_backtest(trader, passive_share=args.passive_share)
