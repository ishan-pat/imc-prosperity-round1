"""Leave-one-day-out cross-validation for round 5 traders.

Runs a trader on each day in isolation (fresh state, no cross-day position
carry) and reports per-fold PnL/DD/Sharpe. Use this instead of a single
held-out day when deciding whether a strategy generalizes.

Usage:
    python3 -m backtesting.round5_loo_cv
    python3 -m backtesting.round5_loo_cv --trader submissions/round5/trader_v2.py
    python3 -m backtesting.round5_loo_cv --passive-share 0.1
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtesting.round5_attribution import load_trader, run_attribution  # noqa: E402

DAYS = [2, 3, 4]


def loo(trader_path: str, passive_share: float):
    rows = []
    for day in DAYS:
        trader = load_trader(trader_path)
        m = run_attribution(trader, [day], passive_share=passive_share)
        rows.append((day, m["total_pnl"], m["max_drawdown"],
                     m["max_drawdown_duration_ticks"], m["sharpe_tick"],
                     m["per_day_product"][day]))
    print(f"\n=== LOO-CV ({Path(trader_path).name}, passive_share={passive_share}) ===")
    print(f"{'Day':>4} {'PnL':>12} {'Max DD':>10} {'DD Ticks':>10} {'Sharpe':>8}")
    print("-" * 50)
    for day, pnl, dd, dur, sh, _ in rows:
        print(f"{day:>4} {pnl:>+12,.0f} {dd:>10,.0f} {dur:>10,} {sh:>+8.3f}")
    print("-" * 50)
    avg_pnl = sum(r[1] for r in rows) / len(rows)
    avg_dd = sum(r[2] for r in rows) / len(rows)
    avg_sh = sum(r[4] for r in rows) / len(rows)
    print(f"{'Mean':>4} {avg_pnl:>+12,.0f} {avg_dd:>10,.0f} {'':>10} {avg_sh:>+8.3f}")

    # cross-day product stability
    products = sorted({p for _, _, _, _, _, pp in rows for p in pp})
    print(f"\nPer-product PnL by held-out day (positive on all 3 = stable signal):")
    print(f"{'Product':<32} " + " ".join(f"{'Day '+str(d):>10}" for d in DAYS) +
          f" {'Min':>10} {'Mean':>10}")
    rec = []
    for prod in products:
        cells = [pp.get(prod, 0.0) for _, _, _, _, _, pp in rows]
        if all(abs(c) < 0.5 for c in cells):
            continue
        rec.append((prod, cells, min(cells), sum(cells) / len(cells)))
    rec.sort(key=lambda r: -r[3])
    for prod, cells, mn, mean in rec:
        cell_str = " ".join(f"{c:>10,.0f}" for c in cells)
        flag = " ✓" if mn > 0 else " ✗" if mean < 0 else ""
        print(f"{prod:<32} {cell_str} {mn:>+10,.0f} {mean:>+10,.0f}{flag}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trader",
                        default=str(REPO_ROOT / "submissions" / "round5" / "trader_v1.py"))
    parser.add_argument("--passive-share", type=float, default=0.5)
    args = parser.parse_args()
    loo(args.trader, args.passive_share)
