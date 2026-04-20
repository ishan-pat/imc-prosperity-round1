"""
Round 2 OSMIUM deep-dive EDA.

Answers four diagnostic questions to guide Phase 2B refinements:
  Q2  Is OSMIUM still mean-reverting around 10,000? (lag-1 AC of returns)
  Q3  What's the realized volatility per day?
  Q4  Is trade flow asymmetric on any day? (classify each trade as buy/sell-init)
  Q5  When we hit +/-80 OSMIUM position, is it adverse or favorable?
      (re-runs the baseline backtest, logs position, looks ahead N ticks)

Usage (from repo root):
    python3 -m notebooks.round2_osmium_analysis
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtesting.round2_backtest import (  # noqa: E402
    PRICE_FILES, TRADE_FILES,
    load_ticks, load_trades, load_trader,
    simulate_aggressive, simulate_passive,
)
from datamodel import OrderDepth, TradingState, Observation  # noqa: E402

PRODUCT = "ASH_COATED_OSMIUM"


# ──────────────────────────────────────────────────────────────────────────
# helpers

def sorted_mids(ticks: dict, product: str):
    """Return list of (ts, mid) for product, ts-sorted.
    Filters out mid==0 (empty book → CSV emits 0) and values far from 10000."""
    out = []
    for ts in sorted(ticks.keys()):
        if product not in ticks[ts]:
            continue
        m = ticks[ts][product]["mid"]
        if m is None or m <= 100 or m >= 20000:
            continue
        out.append((ts, m))
    return out


def book_by_ts(ticks: dict, product: str):
    """{ts: (best_bid, best_ask)} for product."""
    out = {}
    for ts in sorted(ticks.keys()):
        pd = ticks[ts].get(product, {})
        bb = max(pd.get("bids", {}).keys()) if pd.get("bids") else None
        ba = min(pd.get("asks", {}).keys()) if pd.get("asks") else None
        out[ts] = (bb, ba)
    return out


def lag1_autocorr(x):
    if len(x) < 3:
        return None
    mean = statistics.fmean(x)
    num = sum((x[i] - mean) * (x[i - 1] - mean) for i in range(1, len(x)))
    den = sum((xi - mean) ** 2 for xi in x)
    return num / den if den > 0 else None


def rolling_lag1_autocorr(x, window=500):
    """List of lag-1 AC over a rolling window. Returns empty if x is too short."""
    out = []
    for end in range(window, len(x) + 1):
        out.append(lag1_autocorr(x[end - window:end]))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Q2 — mean reversion

def q2_mean_reversion():
    print("=" * 88)
    print("Q2 — OSMIUM mean-reversion check (Round 1 reported lag-1 AC ≈ -0.49)")
    print("=" * 88)
    print(f"{'Day':>4} {'N':>6} {'MeanMid':>10} {'StdMid':>8} {'Lag1AC(Δ)':>12} "
          f"{'MinRollAC':>10} {'MaxRollAC':>10}  {'Verdict':<18}")
    print("-" * 88)
    for price_file, day in PRICE_FILES:
        ticks = load_ticks(price_file)
        mids = [m for _, m in sorted_mids(ticks, PRODUCT)]
        returns = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        ac = lag1_autocorr(returns)
        roll = [a for a in rolling_lag1_autocorr(returns, window=500) if a is not None]
        mean = statistics.fmean(mids)
        std = statistics.stdev(mids) if len(mids) > 1 else 0
        min_roll = min(roll) if roll else float("nan")
        max_roll = max(roll) if roll else float("nan")
        if ac is not None and ac < -0.2:
            verdict = "mean-reverting"
        elif ac is not None and ac > 0.2:
            verdict = "trending (!)"
        elif ac is not None:
            verdict = "weakly reverting"
        else:
            verdict = "insufficient data"
        print(f"{day:>4} {len(mids):>6} {mean:>10.2f} {std:>8.2f} "
              f"{(ac if ac is not None else float('nan')):>12.3f} "
              f"{min_roll:>10.3f} {max_roll:>10.3f}  {verdict:<18}")


# ──────────────────────────────────────────────────────────────────────────
# Q3 — realized vol

def q3_realized_vol():
    print("=" * 88)
    print("Q3 — OSMIUM realized volatility (per-tick price moves)")
    print("=" * 88)
    print(f"{'Day':>4} {'σ(Δmid)':>10} {'σ/mid (bps)':>13} {'max|Δmid|':>10} "
          f"{'p95|Δmid|':>10} {'Zeros':>8}")
    print("-" * 88)
    for price_file, day in PRICE_FILES:
        ticks = load_ticks(price_file)
        mids = [m for _, m in sorted_mids(ticks, PRODUCT)]
        returns = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        if not returns:
            continue
        sigma = statistics.stdev(returns)
        mean_mid = statistics.fmean(mids)
        abs_rets = sorted(abs(r) for r in returns)
        p95 = abs_rets[int(0.95 * len(abs_rets))]
        max_abs = abs_rets[-1]
        n_zero = sum(1 for r in returns if r == 0)
        zero_pct = 100 * n_zero / len(returns)
        print(f"{day:>4} {sigma:>10.3f} {sigma / mean_mid * 10000:>13.2f} "
              f"{max_abs:>10.1f} {p95:>10.1f} {zero_pct:>7.1f}%")


# ──────────────────────────────────────────────────────────────────────────
# Q4 — trade flow asymmetry

def q4_trade_flow():
    print("=" * 88)
    print("Q4 — OSMIUM trade-flow asymmetry (classified via trade price vs. book)")
    print("=" * 88)
    print(f"{'Day':>4} {'BuyInit':>9} {'SellInit':>9} {'Unc':>7} "
          f"{'Buy/Sell':>10} {'NetFlow':>9}")
    print("-" * 88)
    rolls_summary = []
    for price_file, day in PRICE_FILES:
        ticks = load_ticks(price_file)
        trades = load_trades(TRADE_FILES[day])
        book = book_by_ts(ticks, PRODUCT)
        total_buy = total_sell = total_unc = 0
        flow_per_ts = []  # (ts, net_flow_in_window)
        for ts in sorted(trades.keys()):
            trs = trades[ts]
            bb, ba = book.get(ts, (None, None))
            for tr in trs:
                if tr["symbol"] != PRODUCT:
                    continue
                if ba is not None and tr["price"] >= ba:
                    total_buy += tr["qty"]
                    flow_per_ts.append((ts, +tr["qty"]))
                elif bb is not None and tr["price"] <= bb:
                    total_sell += tr["qty"]
                    flow_per_ts.append((ts, -tr["qty"]))
                else:
                    total_unc += tr["qty"]
                    flow_per_ts.append((ts, 0))
        bs = total_buy / total_sell if total_sell else float("inf")
        print(f"{day:>4} {total_buy:>9} {total_sell:>9} {total_unc:>7} "
              f"{bs:>10.2f} {total_buy - total_sell:>+9}")

        # Rolling net flow window — find the worst sustained imbalance
        if flow_per_ts:
            # Rolling sum over 50 consecutive trades
            W = 50
            rolling = []
            for i in range(len(flow_per_ts)):
                lo = max(0, i - W + 1)
                rolling.append(sum(f[1] for f in flow_per_ts[lo:i + 1]))
            max_buy_burst = max(rolling)
            max_sell_burst = min(rolling)
            rolls_summary.append((day, max_buy_burst, max_sell_burst))

    print()
    print(f"{'Day':>4} {'Max 50-trade buy burst':>24} {'Max 50-trade sell burst':>25}")
    for day, mb, ms in rolls_summary:
        print(f"{day:>4} {mb:>+24} {ms:>+25}")


# ──────────────────────────────────────────────────────────────────────────
# Q5 — at-limit adverse vs favorable

def q5_at_limit_analysis(lookahead: int = 500, passive_share: float = 0.5):
    print("=" * 88)
    print(f"Q5 — OSMIUM at-limit episodes "
          f"(position = ±80, lookahead = {lookahead} ticks)")
    print("=" * 88)

    trader_path = REPO_ROOT / "submissions" / "round2" / "trader_v1.py"
    trader = load_trader(str(trader_path))
    positions: dict = defaultdict(int)
    trader_data_str = ""

    print(f"{'Day':>4} {'Long@+80':>10} {'Fav':>6} {'Adv':>6} {'Flat':>6} "
          f"{'Mean μ':>9}  │  {'Short@-80':>10} {'Fav':>6} {'Adv':>6} {'Flat':>6} "
          f"{'Mean μ':>9}")
    print("-" * 88)

    for price_file, day in PRICE_FILES:
        ticks = load_ticks(price_file)
        trades = load_trades(TRADE_FILES[day])
        pos_ts = []  # (ts, osmium_pos, mid)

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
            residuals, agg_f, _ = simulate_aggressive(
                all_orders, tick_data, positions, {}
            )
            pass_f, _ = simulate_passive(
                residuals, trades.get(ts, []), positions, dict(agg_f), passive_share
            )
            for prod in set(list(agg_f) + list(pass_f)):
                positions[prod] += agg_f.get(prod, 0) + pass_f.get(prod, 0)
            osmium_mid = tick_data.get(PRODUCT, {}).get("mid")
            pos_ts.append((ts, positions.get(PRODUCT, 0), osmium_mid))

        # classify at-limit episodes
        long_stats = {"n": 0, "fav": 0, "adv": 0, "flat": 0, "moves": []}
        short_stats = {"n": 0, "fav": 0, "adv": 0, "flat": 0, "moves": []}
        for i, (ts, pos, mid) in enumerate(pos_ts):
            if mid is None or abs(pos) != 80:
                continue
            fut_idx = min(i + lookahead, len(pos_ts) - 1)
            fut_mid = pos_ts[fut_idx][2]
            if fut_mid is None:
                continue
            move = fut_mid - mid
            bucket = long_stats if pos == 80 else short_stats
            bucket["n"] += 1
            if pos == 80:  # favorable if mid rises
                if move > 0:
                    bucket["fav"] += 1
                elif move < 0:
                    bucket["adv"] += 1
                else:
                    bucket["flat"] += 1
                bucket["moves"].append(move)
            else:  # short: favorable if mid falls
                if move < 0:
                    bucket["fav"] += 1
                elif move > 0:
                    bucket["adv"] += 1
                else:
                    bucket["flat"] += 1
                bucket["moves"].append(-move)  # flip sign so "μ" is PnL-favorable direction

        def fmt(bucket):
            if bucket["n"] == 0:
                return f"{0:>10} {'-':>6} {'-':>6} {'-':>6} {'-':>9}"
            mu = statistics.fmean(bucket["moves"])
            return (f"{bucket['n']:>10} {bucket['fav']:>6} {bucket['adv']:>6} "
                    f"{bucket['flat']:>6} {mu:>+9.2f}")

        print(f"{day:>4} {fmt(long_stats)}  │  {fmt(short_stats)}")

    print()
    print("  Interpretation:")
    print("    Fav μ > 0  →  being at the limit precedes a favorable move (good)")
    print("    Fav μ < 0  →  we get stuck at the limit adversely (this is the failure mode)")


# ──────────────────────────────────────────────────────────────────────────
# main

if __name__ == "__main__":
    q2_mean_reversion()
    print()
    q3_realized_vol()
    print()
    q4_trade_flow()
    print()
    q5_at_limit_analysis()
