"""
Round 5 backtest with per-day × per-product attribution and risk metrics.

Wraps round5_backtest.{load_ticks, load_trades, simulate_aggressive,
simulate_passive} so the fill model stays identical. Emits:
  - per-(day, product) PnL  (incremental, equity at end of day - equity at start)
  - tick-level equity curve (for drawdown / Sharpe)
  - max drawdown, max drawdown duration, Sharpe (tick-level)

Usage (from repo root):
    python3 -m backtesting.round5_attribution --trader submissions/round5/trader_v1.py
    python3 -m backtesting.round5_attribution --days 2,3
    python3 -m backtesting.round5_attribution --days 4 --json out.json
"""
import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datamodel import OrderDepth, TradingState, Observation  # noqa: E402
from backtesting.round5_backtest import (  # noqa: E402
    DATA_DIR, PRICE_FILES, TRADE_FILES,
    load_ticks, load_trades, simulate_aggressive, simulate_passive,
)


def load_trader(path: str):
    spec = importlib.util.spec_from_file_location("trader_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trader()


def run_attribution(trader, days, passive_share: float = 0.5):
    """Run backtest, return dict with per-day×product PnL and risk metrics."""
    positions = defaultdict(int)
    total_cash = defaultdict(float)
    trader_data_str = ""

    # equity curve sampled at every tick (for drawdown / Sharpe)
    equity_curve = []        # list of (day, ts, equity)
    per_day_product = {}     # {day: {product: pnl_increment}}
    n_trades = defaultdict(int)
    max_abs_position = defaultdict(int)

    prev_total_equity = 0.0
    prev_per_product_equity = defaultdict(float)

    files = [(p, d) for (p, d) in PRICE_FILES if d in days]
    if not files:
        raise SystemExit(f"No price files for days {days}")

    for price_file, day in files:
        ticks = load_ticks(price_file)
        trades = load_trades(TRADE_FILES[day])
        per_day_product[day] = {}

        sorted_ts = sorted(ticks.keys())
        for ts in sorted_ts:
            tick_data = ticks[ts]
            order_depths = {}
            for prod, data in tick_data.items():
                od = OrderDepth()
                od.buy_orders = dict(data["bids"])
                od.sell_orders = dict(data["asks"])
                order_depths[prod] = od

            state = TradingState(
                traderData=trader_data_str,
                timestamp=ts,
                listings={},
                order_depths=order_depths,
                own_trades={p: [] for p in order_depths},
                market_trades={p: [] for p in order_depths},
                position=dict(positions),
                observations=Observation({}, {}),
            )
            result, _, trader_data_str = trader.run(state)

            all_orders = [o for orders in result.values() for o in orders]
            for o in all_orders:
                n_trades[o.symbol] += 1

            residuals, agg_f, agg_c = simulate_aggressive(
                all_orders, tick_data, positions, {}
            )
            pass_f, pass_c = simulate_passive(
                residuals, trades.get(ts, []), positions, dict(agg_f), passive_share
            )
            for prod in set(list(agg_f) + list(pass_f)):
                positions[prod] += agg_f.get(prod, 0) + pass_f.get(prod, 0)
            for prod in set(list(agg_c) + list(pass_c)):
                c = agg_c.get(prod, 0.0) + pass_c.get(prod, 0.0)
                total_cash[prod] += c
            for prod, p in positions.items():
                max_abs_position[prod] = max(max_abs_position[prod], abs(p))

            # tick-level equity
            equity = 0.0
            for prod, pos in positions.items():
                mid = tick_data.get(prod, {}).get("mid")
                if mid is None:
                    # fall back to last known mid via cash-only marking
                    continue
                equity += total_cash.get(prod, 0.0) + pos * mid
            # cash-only contribution from products absent this tick
            for prod in total_cash:
                if prod not in tick_data:
                    equity += total_cash[prod]
            equity_curve.append((day, ts, equity))

        # End of day: per-product attribution increment
        final_tick = ticks[sorted_ts[-1]]
        day_total = 0.0
        for prod in sorted(set(list(positions.keys()) + list(total_cash.keys()))):
            mid = final_tick.get(prod, {}).get("mid") or 0
            pos = positions.get(prod, 0)
            equity = total_cash.get(prod, 0.0) + pos * mid
            increment = equity - prev_per_product_equity[prod]
            per_day_product[day][prod] = increment
            prev_per_product_equity[prod] = equity
            day_total += equity
        prev_total_equity = day_total

    # ── risk metrics ──────────────────────────────────────────────────────
    equities = [e for (_, _, e) in equity_curve]
    if not equities:
        return {"total_pnl": 0.0}

    peak = equities[0]
    max_dd = 0.0
    max_dd_duration = 0
    cur_dd_start = 0
    for i, e in enumerate(equities):
        if e > peak:
            peak = e
            cur_dd_start = i
        dd = peak - e
        if dd > max_dd:
            max_dd = dd
        max_dd_duration = max(max_dd_duration, i - cur_dd_start)

    diffs = [equities[i] - equities[i - 1] for i in range(1, len(equities))]
    if diffs:
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        sharpe = (mean / std) * math.sqrt(len(diffs)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_pnl": prev_total_equity,
        "per_day_product": per_day_product,
        "max_drawdown": max_dd,
        "max_drawdown_duration_ticks": max_dd_duration,
        "sharpe_tick": sharpe,
        "n_ticks": len(equities),
        "max_abs_position": dict(max_abs_position),
        "n_trades": dict(n_trades),
        "final_positions": dict(positions),
    }


def fmt_table(metrics, days):
    lines = []
    lines.append(f"=== Trader run on days {sorted(days)} ===")
    lines.append(f"Total PnL: {metrics['total_pnl']:>+12,.0f}")
    lines.append(f"Max drawdown: {metrics['max_drawdown']:>+12,.0f}")
    lines.append(f"Max DD duration: {metrics['max_drawdown_duration_ticks']:>10,} ticks")
    lines.append(f"Sharpe (tick): {metrics['sharpe_tick']:>+12.3f}")
    lines.append("")
    lines.append("Per-day × per-product PnL:")
    products = set()
    for day, pp in metrics["per_day_product"].items():
        products.update(pp.keys())
    products = sorted(products)
    header = f"{'Product':<32} " + " ".join(f"{'Day '+str(d):>10}" for d in sorted(metrics['per_day_product'])) + f" {'Total':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    rows = []
    for prod in products:
        cells = [metrics["per_day_product"][d].get(prod, 0.0) for d in sorted(metrics["per_day_product"])]
        total = sum(cells)
        if any(abs(c) > 0.5 for c in cells) or abs(total) > 0.5:
            rows.append((prod, cells, total))
    rows.sort(key=lambda r: -abs(r[2]))
    for prod, cells, total in rows:
        cell_str = " ".join(f"{c:>10,.0f}" for c in cells)
        lines.append(f"{prod:<32} {cell_str} {total:>+12,.0f}")
    lines.append("-" * len(header))
    day_totals = {d: sum(metrics["per_day_product"][d].values()) for d in sorted(metrics["per_day_product"])}
    grand = sum(day_totals.values())
    lines.append(f"{'TOTAL':<32} " + " ".join(f"{day_totals[d]:>+10,.0f}" for d in sorted(day_totals)) + f" {grand:>+12,.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trader",
                        default=str(REPO_ROOT / "submissions" / "round5" / "trader_v1.py"))
    parser.add_argument("--days", default="2,3,4",
                        help="Comma-separated day numbers to run (e.g., 2,3 or 4).")
    parser.add_argument("--passive-share", type=float, default=0.5)
    parser.add_argument("--json", default=None,
                        help="Write metrics to this path as JSON.")
    args = parser.parse_args()

    days = [int(d) for d in args.days.split(",")]
    trader = load_trader(args.trader)
    metrics = run_attribution(trader, days, passive_share=args.passive_share)
    print(fmt_table(metrics, days))
    if args.json:
        # final_positions ints; per_day_product keys -> str for JSON
        out = {
            **{k: v for k, v in metrics.items() if k not in ("per_day_product",)},
            "per_day_product": {str(d): pp for d, pp in metrics["per_day_product"].items()},
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
