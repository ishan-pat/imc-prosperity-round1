"""
Backtest the Round 1 submitted trader on historical CSV data.
Usage (from repo root): python3 -m backtesting.round1_backtest
Reports per-day and total P&L for each product.
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "round1"))

from datamodel import OrderDepth, TradingState, Observation  # noqa: E402
from final import Trader  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "round1"
PRICE_FILES = [
    (str(DATA_DIR / "prices_round_1_day_-2.csv"), -2),
    (str(DATA_DIR / "prices_round_1_day_-1.csv"), -1),
    (str(DATA_DIR / "prices_round_1_day_0.csv"),   0),
]

def load_ticks(filename: str) -> dict:
    """Returns {timestamp: {product: {mid, bids, asks}}}"""
    ticks = {}
    with open(filename) as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts = int(row["timestamp"])
            prod = row["product"]
            if ts not in ticks:
                ticks[ts] = {}
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
                        asks[int(float(ap))] = -int(av)  # negative per IMC spec
                    except ValueError:
                        pass
            ticks[ts][prod] = {"mid": mid, "bids": bids, "asks": asks}
    return ticks


def simulate_fill(our_orders: list, ticks_at_ts: dict) -> dict:
    """
    Immediately fill our orders against the current order book (conservative):
    - BUY order at price P: fills against any ask_price <= P
    - SELL order at price P: fills against any bid_price >= P
    Returns filled_qty {product: net_qty} and cash_flow {product: cash}.
    """
    filled_qty = {}
    cash_flow = {}

    for order in our_orders:
        prod = order.symbol
        qty = order.quantity  # positive = buy, negative = sell
        price = order.price
        book = ticks_at_ts.get(prod, {})

        if qty > 0:  # buy order: match against asks
            asks = book.get("asks", {})
            remaining = qty
            cash = 0.0
            for ask_p in sorted(asks.keys()):
                if ask_p > price:
                    break
                avail = -asks[ask_p]  # convert negative to positive
                fill = min(remaining, avail)
                cash -= ask_p * fill
                remaining -= fill
                if remaining <= 0:
                    break
            filled = qty - remaining
            filled_qty[prod] = filled_qty.get(prod, 0) + filled
            cash_flow[prod] = cash_flow.get(prod, 0.0) + cash

        else:  # sell order: match against bids
            bids = book.get("bids", {})
            remaining = -qty  # make positive
            cash = 0.0
            for bid_p in sorted(bids.keys(), reverse=True):
                if bid_p < price:
                    break
                avail = bids[bid_p]
                fill = min(remaining, avail)
                cash += bid_p * fill
                remaining -= fill
                if remaining <= 0:
                    break
            filled = (-qty) - remaining
            filled_qty[prod] = filled_qty.get(prod, 0) - filled
            cash_flow[prod] = cash_flow.get(prod, 0.0) + cash

    return filled_qty, cash_flow


def run_backtest():
    trader = Trader()
    trader_data_str = ""

    total_cash = {}
    positions = {}

    print(f"{'Day':>5} {'Product':<28} {'Pos':>5} {'Cash':>12} {'MtM':>12} {'PnL':>12}")
    print("-" * 80)

    overall_pnl = 0.0

    for filename, day in PRICE_FILES:
        ticks = load_ticks(filename)
        day_cash = {}

        for ts in sorted(ticks.keys()):
            tick_data = ticks[ts]

            order_depths = {}
            for prod, data in tick_data.items():
                od = OrderDepth()
                od.buy_orders = data["bids"]
                od.sell_orders = data["asks"]
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
            for prod, orders in result.items():
                all_orders.extend(orders)

            filled_qty, cash = simulate_fill(all_orders, tick_data)
            for prod, qty in filled_qty.items():
                positions[prod] = positions.get(prod, 0) + qty
            for prod, c in cash.items():
                day_cash[prod] = day_cash.get(prod, 0.0) + c
                total_cash[prod] = total_cash.get(prod, 0.0) + c

        final_tick = ticks[max(ticks.keys())]
        day_pnl = 0.0
        for prod in set(list(positions.keys()) + list(day_cash.keys())):
            mid = final_tick.get(prod, {}).get("mid") or 0
            pos = positions.get(prod, 0)
            cash_p = day_cash.get(prod, 0.0)
            pnl = cash_p + pos * mid
            day_pnl += pnl
            print(f"{day:>5} {prod:<28} {pos:>5} {cash_p:>12.0f} {pos*mid:>12.0f} {pnl:>12.0f}")

        overall_pnl += day_pnl
        print(f"{'':>5} {'DAY TOTAL':<28} {'':>5} {'':>12} {'':>12} {day_pnl:>12.0f}")
        print()

    print("=" * 80)
    print(f"OVERALL P&L (mark-to-market): {overall_pnl:,.0f} XIREC")
    print("Note: conservative estimate — passive fills NOT simulated")


if __name__ == "__main__":
    run_backtest()
