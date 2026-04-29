"""
Round 5 backtest with trade-tape passive fill model.

Same model as round2_backtest:
  1. Aggressive fills against the order book at each tick.
  2. Passive fills from trade tape, split 50/50 between buy/sell side
     (Round 5 trades have empty buyer/seller fields, so we can't infer side).
  3. Position limits hardcoded to 10 per product (Round 5 brief).
  4. Per-day incremental PnL (Δequity, not MtM-of-carried-position).

Usage (from repo root):
    python3 -m backtesting.round5_backtest
    python3 -m backtesting.round5_backtest --trader submissions/round5/trader_v1.py
    python3 -m backtesting.round5_backtest --passive-share 0.3
    python3 -m backtesting.round5_backtest --product PANEL_2X4   # focus on one product
"""
import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datamodel import OrderDepth, TradingState, Observation, Order  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "round5"
PRICE_FILES = [
    (DATA_DIR / "prices_round_5_day_2.csv", 2),
    (DATA_DIR / "prices_round_5_day_3.csv", 3),
    (DATA_DIR / "prices_round_5_day_4.csv", 4),
]
TRADE_FILES = {
    2: DATA_DIR / "trades_round_5_day_2.csv",
    3: DATA_DIR / "trades_round_5_day_3.csv",
    4: DATA_DIR / "trades_round_5_day_4.csv",
}

# Round 5 brief: position limit = 10 per product, all 50 products
POSITION_LIMIT_DEFAULT = 10


def load_trader(path: str):
    spec = importlib.util.spec_from_file_location("trader_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trader()


def load_ticks(filename: Path) -> dict:
    """{timestamp: {product: {mid, bids, asks}}}"""
    ticks: dict = {}
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts = int(row["timestamp"])
            prod = row["product"]
            ticks.setdefault(ts, {})
            try:
                mid = float(row["mid_price"]) if row["mid_price"] else None
            except ValueError:
                mid = None
            bids, asks = {}, {}
            for level in ["1", "2", "3"]:
                bp = row.get(f"bid_price_{level}", "")
                bv = row.get(f"bid_volume_{level}", "")
                ap = row.get(f"ask_price_{level}", "")
                av = row.get(f"ask_volume_{level}", "")
                if bp and bv:
                    try:
                        bids[int(float(bp))] = int(bv)
                    except ValueError:
                        pass
                if ap and av:
                    try:
                        asks[int(float(ap))] = -int(av)
                    except ValueError:
                        pass
            ticks[ts][prod] = {"mid": mid, "bids": bids, "asks": asks}
    return ticks


def load_trades(filename: Path) -> dict:
    trades: dict = defaultdict(list)
    if not filename.exists():
        return trades
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                ts = int(row["timestamp"])
                price = float(row["price"])
                qty = int(row["quantity"])
            except (ValueError, KeyError):
                continue
            trades[ts].append({"symbol": row["symbol"], "price": price, "qty": qty})
    return trades


def _clamp_to_limit(qty: int, current_pos: int, limit: int = POSITION_LIMIT_DEFAULT) -> int:
    if qty > 0:
        return max(0, min(qty, limit - current_pos))
    return min(0, max(qty, -(limit + current_pos)))


def simulate_aggressive(orders, book_at_ts, positions_in, agg_filled_scratch):
    residuals = []
    filled = defaultdict(int)
    cash = defaultdict(float)
    for order in orders:
        prod = order.symbol
        price = order.price
        cur = positions_in.get(prod, 0) + agg_filled_scratch.get(prod, 0) + filled[prod]
        qty = _clamp_to_limit(order.quantity, cur)
        if qty == 0:
            continue
        book = book_at_ts.get(prod, {})
        if qty > 0:
            remaining = qty
            for ask_p in sorted(book.get("asks", {}).keys()):
                if ask_p > price or remaining <= 0:
                    break
                avail = -book["asks"][ask_p]
                fill = min(remaining, avail)
                cash[prod] -= ask_p * fill
                filled[prod] += fill
                remaining -= fill
            if remaining > 0:
                residuals.append(Order(prod, price, remaining))
        else:
            remaining = -qty
            for bid_p in sorted(book.get("bids", {}).keys(), reverse=True):
                if bid_p < price or remaining <= 0:
                    break
                avail = book["bids"][bid_p]
                fill = min(remaining, avail)
                cash[prod] += bid_p * fill
                filled[prod] -= fill
                remaining -= fill
            if remaining > 0:
                residuals.append(Order(prod, price, -remaining))
    return residuals, filled, cash


def simulate_passive(residuals, trades_at_ts, positions_in, already_filled, passive_share):
    filled = defaultdict(int)
    cash = defaultdict(float)
    buys = defaultdict(list)
    sells = defaultdict(list)
    for r in residuals:
        if r.quantity > 0:
            buys[r.symbol].append([r.price, r.quantity])
        else:
            sells[r.symbol].append([r.price, -r.quantity])
    for prod in buys:
        buys[prod].sort(key=lambda x: -x[0])
    for prod in sells:
        sells[prod].sort(key=lambda x: x[0])
    for trade in trades_at_ts:
        prod = trade["symbol"]
        tp = trade["price"]
        tq = trade["qty"]
        buy_pool = 0.5 * tq * passive_share
        sell_pool = 0.5 * tq * passive_share
        pool = buy_pool
        for entry in buys.get(prod, []):
            if pool < 1 or entry[1] <= 0 or entry[0] < tp:
                continue
            cur = positions_in.get(prod, 0) + already_filled.get(prod, 0) + filled[prod]
            max_buy = POSITION_LIMIT_DEFAULT - cur
            fill_qty = int(min(entry[1], pool, max_buy))
            if fill_qty <= 0:
                continue
            cash[prod] -= entry[0] * fill_qty
            filled[prod] += fill_qty
            entry[1] -= fill_qty
            pool -= fill_qty
        pool = sell_pool
        for entry in sells.get(prod, []):
            if pool < 1 or entry[1] <= 0 or entry[0] > tp:
                continue
            cur = positions_in.get(prod, 0) + already_filled.get(prod, 0) + filled[prod]
            max_sell = POSITION_LIMIT_DEFAULT + cur
            fill_qty = int(min(entry[1], pool, max_sell))
            if fill_qty <= 0:
                continue
            cash[prod] += entry[0] * fill_qty
            filled[prod] -= fill_qty
            entry[1] -= fill_qty
            pool -= fill_qty
    return filled, cash


def run_backtest(trader, passive_share: float = 0.5,
                 verbose: bool = True, focus_product: str = None) -> dict:
    positions = defaultdict(int)
    total_cash = defaultdict(float)
    trader_data_str = ""

    if verbose:
        print(f"{'Day':>4} {'Product':<32} {'Pos':>5} {'Δcash':>13} {'MtM':>13} {'Equity':>13}")
        print("-" * 92)

    prev_total_equity = 0.0
    max_abs_position = defaultdict(int)
    n_trades = defaultdict(int)
    per_product_pnl = defaultdict(float)

    for price_file, day in PRICE_FILES:
        ticks = load_ticks(price_file)
        trades = load_trades(TRADE_FILES[day])
        day_cash = defaultdict(float)

        for ts in sorted(ticks.keys()):
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

            residuals, agg_f, agg_c = simulate_aggressive(all_orders, tick_data, positions, {})
            pass_f, pass_c = simulate_passive(
                residuals, trades.get(ts, []), positions, dict(agg_f), passive_share
            )
            for prod in set(list(agg_f) + list(pass_f)):
                positions[prod] += agg_f.get(prod, 0) + pass_f.get(prod, 0)
            for prod in set(list(agg_c) + list(pass_c)):
                c = agg_c.get(prod, 0.0) + pass_c.get(prod, 0.0)
                day_cash[prod] += c
                total_cash[prod] += c
            for prod, p in positions.items():
                max_abs_position[prod] = max(max_abs_position[prod], abs(p))

        # End-of-day MtM
        final_tick = ticks[max(ticks.keys())]
        day_equity = 0.0
        for prod in sorted(set(list(positions.keys()) + list(total_cash.keys()))):
            mid = final_tick.get(prod, {}).get("mid") or 0
            pos = positions.get(prod, 0)
            total_c = total_cash.get(prod, 0.0)
            equity = total_c + pos * mid
            day_equity += equity
            per_product_pnl[prod] = equity
            if verbose and (focus_product is None or prod == focus_product or pos != 0
                            or day_cash.get(prod, 0) != 0):
                print(f"{day:>4} {prod:<32} {pos:>5} {day_cash.get(prod, 0):>13.0f} "
                      f"{pos*mid:>13.0f} {equity:>13.0f}")
        day_pnl = day_equity - prev_total_equity
        if verbose:
            print(f"{'':>4} {'DAY INCR PNL':<32} {'':>5} {'':>13} {'':>13} {day_pnl:>13.0f}")
            print()
        prev_total_equity = day_equity

    if verbose:
        print("=" * 92)
        print(f"OVERALL PnL: {prev_total_equity:,.0f} XIRECs   "
              f"(passive_share={passive_share})")
        print()
        print(f"{'Product':<32} {'Final Eq':>12} {'Max |Pos|':>10} {'Order ticks':>12}")
        for prod in sorted(per_product_pnl.keys(),
                           key=lambda p: -abs(per_product_pnl[p])):
            if per_product_pnl[prod] != 0 or max_abs_position[prod] > 0:
                print(f"  {prod:<30} {per_product_pnl[prod]:>12,.0f} "
                      f"{max_abs_position[prod]:>10} {n_trades[prod]:>12}")

    return {
        "total_pnl": prev_total_equity,
        "per_product": dict(per_product_pnl),
        "max_abs_position": dict(max_abs_position),
        "n_trades": dict(n_trades),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trader",
                        default=str(REPO_ROOT / "submissions" / "round5" / "trader_v1.py"))
    parser.add_argument("--passive-share", type=float, default=0.5)
    parser.add_argument("--product", default=None,
                        help="Focus output on a single product (still simulates all).")
    args = parser.parse_args()
    trader = load_trader(args.trader)
    run_backtest(trader, passive_share=args.passive_share, focus_product=args.product)
