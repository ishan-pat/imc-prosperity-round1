"""
Round 3 backtest. Mirrors round2_backtest.py but with the Round 3 product set
(HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_<K>) and matching position limits.

Usage (from repo root):
    python3 -m backtesting.round3_backtest
    python3 -m backtesting.round3_backtest --trader submissions/round3/trader.py
    python3 -m backtesting.round3_backtest --passive-share 0.3
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

DATA_DIR = REPO_ROOT / "data" / "round3"
PRICE_FILES = [
    (DATA_DIR / "prices_round_3_day_0.csv", 0),
    (DATA_DIR / "prices_round_3_day_1.csv", 1),
    (DATA_DIR / "prices_round_3_day_2.csv", 2),
]
TRADE_FILES = {
    0: DATA_DIR / "trades_round_3_day_0.csv",
    1: DATA_DIR / "trades_round_3_day_1.csv",
    2: DATA_DIR / "trades_round_3_day_2.csv",
}

VOUCHER_STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in VOUCHER_STRIKES},
}


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
            for level in ("1", "2", "3"):
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
    out: dict = defaultdict(list)
    if not filename.exists():
        return out
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                ts = int(row["timestamp"])
                price = float(row["price"])
                qty = int(row["quantity"])
            except (ValueError, KeyError):
                continue
            out[ts].append({"symbol": row["symbol"], "price": price, "qty": qty})
    return out


def _clamp_to_limit(prod: str, desired_qty: int, current_pos: int) -> int:
    limit = POSITION_LIMITS.get(prod, 10**9)
    if desired_qty > 0:
        return max(0, min(desired_qty, limit - current_pos))
    return min(0, max(desired_qty, -(limit + current_pos)))


def simulate_aggressive(orders, book_at_ts, positions_in, agg_filled_scratch):
    residuals = []
    filled = defaultdict(int)
    cash = defaultdict(float)

    for order in orders:
        prod = order.symbol
        price = order.price
        cur = positions_in.get(prod, 0) + agg_filled_scratch.get(prod, 0) + filled[prod]
        qty = _clamp_to_limit(prod, order.quantity, cur)
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
            if pool < 1 or entry[1] <= 0:
                continue
            if entry[0] < tp:
                continue
            cur = positions_in.get(prod, 0) + already_filled.get(prod, 0) + filled[prod]
            max_buy = POSITION_LIMITS.get(prod, 10**9) - cur
            fill_qty = int(min(entry[1], pool, max_buy))
            if fill_qty <= 0:
                continue
            cash[prod] -= entry[0] * fill_qty
            filled[prod] += fill_qty
            entry[1] -= fill_qty
            pool -= fill_qty

        pool = sell_pool
        for entry in sells.get(prod, []):
            if pool < 1 or entry[1] <= 0:
                continue
            if entry[0] > tp:
                continue
            cur = positions_in.get(prod, 0) + already_filled.get(prod, 0) + filled[prod]
            max_sell = POSITION_LIMITS.get(prod, 10**9) + cur
            fill_qty = int(min(entry[1], pool, max_sell))
            if fill_qty <= 0:
                continue
            cash[prod] += entry[0] * fill_qty
            filled[prod] -= fill_qty
            entry[1] -= fill_qty
            pool -= fill_qty

    return filled, cash


def run_backtest(trader, passive_share: float = 0.5, verbose: bool = True,
                 days: list = None) -> float:
    positions = defaultdict(int)
    total_cash = defaultdict(float)
    trader_data_str = ""

    if verbose:
        print(f"{'Day':>4} {'Product':<24} {'Pos':>5} {'Δcash':>13} {'MtM':>13} {'Equity':>13}")
        print("-" * 80)

    prev_total_equity = 0.0
    max_abs_position = defaultdict(int)
    ticks_at_limit = defaultdict(int)
    total_ticks = 0

    files_to_run = PRICE_FILES if days is None else [(p, d) for p, d in PRICE_FILES if d in days]

    for price_file, day in files_to_run:
        ticks = load_ticks(price_file)
        trades = load_trades(TRADE_FILES[day])
        day_cash = defaultdict(float)

        for ts in sorted(ticks.keys()):
            total_ticks += 1
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

            all_orders = []
            for orders in result.values():
                all_orders.extend(orders)

            residuals, agg_filled, agg_cash = simulate_aggressive(
                all_orders, tick_data, positions, {}
            )
            pass_filled, pass_cash = simulate_passive(
                residuals, trades.get(ts, []), positions, dict(agg_filled), passive_share
            )

            for prod in set(list(agg_filled) + list(pass_filled)):
                positions[prod] += agg_filled.get(prod, 0) + pass_filled.get(prod, 0)
            for prod in set(list(agg_cash) + list(pass_cash)):
                c = agg_cash.get(prod, 0.0) + pass_cash.get(prod, 0.0)
                day_cash[prod] += c
                total_cash[prod] += c

            for prod, p in positions.items():
                max_abs_position[prod] = max(max_abs_position[prod], abs(p))
                if abs(p) >= POSITION_LIMITS.get(prod, 10**9):
                    ticks_at_limit[prod] += 1

        final_tick = ticks[max(ticks.keys())]
        day_equity = 0.0
        for prod in sorted(set(list(positions.keys()) + list(total_cash.keys()))):
            mid = final_tick.get(prod, {}).get("mid") or 0
            pos = positions.get(prod, 0)
            total_c = total_cash.get(prod, 0.0)
            equity = total_c + pos * mid
            day_equity += equity
            if verbose:
                print(f"{day:>4} {prod:<24} {pos:>5} {day_cash.get(prod, 0):>13.0f} "
                      f"{pos*mid:>13.0f} {equity:>13.0f}")
        day_pnl = day_equity - prev_total_equity
        if verbose:
            print(f"{'':>4} {'DAY INCR PNL':<24} {'':>5} {'':>13} {'':>13} {day_pnl:>13.0f}")
            print()
        prev_total_equity = day_equity

    if verbose:
        print("=" * 80)
        print(f"OVERALL PnL (cumulative equity, MtM at end of last day): "
              f"{prev_total_equity:,.0f} XIRECs")
        print(f"Passive fill share parameter: {passive_share}")
        print()
        print(f"Max |position|:  " + ", ".join(
            f"{p}={v}" for p, v in sorted(max_abs_position.items())
        ))
        print(f"Ticks at limit:  " + ", ".join(
            f"{p}={v}/{total_ticks} ({100*v/total_ticks:.1f}%)"
            for p, v in sorted(ticks_at_limit.items()) if v > 0
        ))
    return prev_total_equity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trader",
        default=str(REPO_ROOT / "submissions" / "round3" / "trader.py"),
    )
    parser.add_argument("--passive-share", type=float, default=0.5)
    args = parser.parse_args()
    trader = load_trader(args.trader)
    run_backtest(trader, passive_share=args.passive_share)
