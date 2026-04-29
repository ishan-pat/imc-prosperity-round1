"""
Round 4 trader — extends Round 3 with COUNTERPARTY-CONDITIONAL logic.

CHANGES vs Round 3 (provenance: notebooks/round4_eda.py)
  1.  TTE_DAY_START = 4 (Round 3 was 5).
  2.  Active voucher universe shifted from {5000…5500} → {4000, 5300…5500,
      6000, 6500}.  Round-4 trade activity in (5000, 5100, 5200, 4500) is
      < 50 trades total — no liquidity to fade mispricings.
  3.  SVI parameters refit on Round 4 data (day 3, TTE=6, closest to live
      TTE=4).  See writeups/round4_calibration.json.
  4.  NEW: counterparty module.  All 7 Marks classified (n=4,281 historical
      trade-legs across days 1-3) by:
        - aggression direction (Lee-Ready)
        - realised PnL trail vs forward mid at +30 ticks
        - per-product activity
      Tier dict hard-coded; live mark_stats updated each tick from
      state.market_trades for corroboration & drift detection.
  5.  Voucher trading is asymmetric for high-K (≥5300):
        - Mark 22 sells those vouchers (97% of high-K sell-side flow);
          we accept their flow with reduced edge_buy threshold.
        - Mark 01 buys them (87% of high-K buy-side flow); they're mildly
          informed, so we widen edge_sell to avoid being lifted at fair.
      VEV_4000 stays symmetric — Mark 14↔Mark 38 flow is balanced 50/50.

WHAT IS UNCHANGED
  - AS calibration σ/k for HYDROGEL & VELVETFRUIT (Round-4 stats are
    identical to Round 3 within rounding: σ_log/day = 0.0217 both rounds).
  - γ tuned values from Round 3 sweep (backtesting/round3_sweep.py).
  - Cointegration HYDR↔VEV: still dropped (β flips sign across days).
  - EOD flatten windows.

REFERENCES
  - Black-Scholes (1973), Avellaneda-Stoikov (2008), Gatheral-Jacquier (2014):
    same as Round 3 trader.
  - NEW Glosten-Milgrom (1985): tier-conditional spread asymmetry.
  - NEW Lee-Ready (1991): aggression classifier (offline only, in EDA).
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Trade, TradingState

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

HYDROGEL = "HYDROGEL_PACK"
UNDERLYING = "VELVETFRUIT_EXTRACT"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHER_SYMS: Dict[int, str] = {k: f"VEV_{k}" for k in STRIKES}
ALL_VOUCHERS = tuple(VOUCHER_SYMS.values())

# Round 4 active universe — drives _trade_vouchers loop.
# Provenance: notebooks/round4_eda.py "voucher activity" table (n_trades > 100
# across days 1-3).  Round 3's {5000-5500} were inactive in Round 4 data.
ACTIVE_VOUCHER_STRIKES = (4000, 5300, 5400, 5500, 6000, 6500)

# High-K threshold for asymmetric voucher quoting (Mark 22 sells, Mark 01 buys).
HIGH_K_VOUCHER_THRESHOLD = 5300

LIMITS: Dict[str, int] = {
    HYDROGEL: 200,
    UNDERLYING: 200,
    **{sym: 300 for sym in ALL_VOUCHERS},
}

# LIVE round-4 day length is 100K timestamps (verified from submission PnL chart
# X-axis ending at 99.9K).  Round 3 historical data had 1M-timestamp days; the
# backtest harness monkey-patches DAY_LENGTH_TS back to 1_000_000 to keep the
# replay coherent (otherwise the trader would think every historical day ends
# 10x earlier and EOD-flatten prematurely).
DAY_LENGTH_TS = 100_000
LIVE_TTE_DAY_START = 4   # Round 4 starts at TTE = 4 days (was 5 in Round 3)

# ───────────────────────────────────────────────────────────────────────────
# Raw-SVI parameters from offline fit (notebooks/round4_eda.py).
# Picked Round-4 day 3 (TTE=6, closest to the live TTE=4 we'll start with;
# RMSE_w = 0.000091 — best fit of the three days).  Note the bound-hit
# σ=1.000 is normal for a near-flat smile (Round-4 IVs span 0.011-0.041).
SVI_PARAMS = {
    "a":     -0.261,
    "b":      0.263,
    "rho":   -0.081,
    "m":     -0.079,
    "sigma":  1.000,
}

SIGMA_FLOOR = 0.005
SIGMA_CEIL = 0.06

# Per-strike historical median IV from notebooks/round4_eda.py, averaged
# across Round-4 days 1-3.  We use these DIRECTLY as the σ_K seed instead of
# evaluating SVI.  Reason: the SVI fit on Round-4 data is dominated by the
# wing strikes (K=4000, K=6500) — at the σ-bound 1.0 it overestimates ATM
# σ by ~35% which causes spurious voucher buying on day 1 (MtM loss before
# EWMA converges).  Direct medians are simpler and stable.  SVI module
# kept for reference/extension only.
HISTORICAL_SIGMA: Dict[int, float] = {
    4000: 0.0388,
    4500: 0.0230,
    5000: 0.0112,
    5100: 0.0109,
    5200: 0.0112,
    5300: 0.0114,
    5400: 0.0107,
    5500: 0.0115,
    6000: 0.0199,
    6500: 0.0302,
}

# ───────────────────────────────────────────────────────────────────────────
# Avellaneda-Stoikov parameters
# σ values: Round-4 EDA confirms σ_log/day = 0.0217 for both products,
# identical to Round 3.  In price units:
#   HYDROGEL    σ ≈ 0.0217 × 9990 ≈ 217 / day
#   VELVETFRUIT σ ≈ 0.0217 × 5250 ≈ 114 / day
# γ tuned values inherited from Round 3 sweep.
GAMMA_HYDROGEL = 2.1e-7
GAMMA_VELVETFRUIT = 6.8e-7
SIGMA_HYDROGEL = 217.0
SIGMA_VELVETFRUIT = 114.0
K_HYDROGEL = 0.128
K_VELVETFRUIT = 0.443

MAX_HALF_SPREAD = {
    HYDROGEL: 9,
    UNDERLYING: 2,
}
MIN_HALF_SPREAD = {
    HYDROGEL: 1,
    UNDERLYING: 1,
}

MM_LEVELS = 4
MM_TOTAL_SIZE = {
    HYDROGEL: 178,
    UNDERLYING: 174,
}

# ───────────────────────────────────────────────────────────────────────────
# Voucher vol-arb tunables.
#
# CHANGED in round-4 v3 after live submission #2 showed -$1,950 PnL with
# voucher-driven losses:
#   - VOUCHER_MIN_EDGE 4.24 → 30:  effectively gates voucher trading off for
#     normal market conditions.  Live σ_K diverges enough from historical seed
#     that even post-cold-start trades lose.  Backtest shows vouchers
#     contribute $0 at matched TTE; live showed -$1,950 ⇒ asymmetric downside,
#     remove the trade.  Door stays open for extreme dislocations (>30 XIRECs).
#   - EOD_VOUCHER_FLATTEN_ENABLED False:  EOD flatten was costing ~$1,150 by
#     dumping any incidental voucher inventory into the touch.  Better to mark
#     positions at mid (free) than liquidate them at bid/ask (spread × size).
VOUCHER_MIN_EDGE = 30.0
VOUCHER_EDGE_FRAC_OF_SPREAD = 0.5
VOUCHER_MAX_TRADE_SIZE = 25
EOD_VOUCHER_FLATTEN_ENABLED = False

# Asymmetric edge multipliers for high-K vouchers (≥ 5300).
# Reference: Glosten & Milgrom (1985) — narrow spread to noise-direction
# flow, widen vs informed-direction flow.
# Round-4 EDA: high-K asks are lifted by Mark 01 (87% buy bias, mildly
# informed); high-K bids are hit by Mark 22 (97% sell bias, near-neutral).
# So: easy to BUY (Mark 22 sells to us), hard to SELL (avoid Mark 01 lift).
HIGH_K_EDGE_BUY_MULT = 0.7    # cheaper to take Mark 22's offer
HIGH_K_EDGE_SELL_MULT = 1.4   # require more edge before selling to Mark 01

SIGMA_EWMA_ALPHA = 0.0026

# Voucher cold-start gate: skip voucher trading until σ EWMA has converged
# from the seed.  At α=0.0026, the seed weight after N ticks is (1-α)^N;
# at N=1000 the seed weight is ~0.073 — EWMA is dominated by live IV.
# Live submission showed −$1,400 of cold-start voucher PnL in the first
# ~7K timestamps; this gate kills that loss window while sacrificing only
# ~1% of the day's voucher trading opportunities.
VOUCHER_COLD_START_TICKS = 1000

DELTA_DEAD_ZONE = 75.0
# EOD windows scaled to live DAY_LENGTH_TS = 100K; backtest monkey-patches
# both to 950_000 / 1_000_000 to match historical day length.
EOD_FLATTEN_VOUCHERS_START_TS = 95_000
EOD_FLATTEN_LINEAR_START_TS = 95_000


# ═══════════════════════════════════════════════════════════════════════════
# COUNTERPARTY MODULE  (NEW for Round 4)
# Reference: Glosten & Milgrom (1985) "Bid, ask and transaction prices in
# a specialist market with heterogeneously informed traders" (JFE).
#
# Tiers are HARD-CODED from offline notebooks/round4_eda.py.  Live updates
# in mark_stats CORROBORATE the hardcoded tiers; they don't override them
# (drift detection is a Step-5 future item).  Unknown Marks default to
# "neutral" until enough live data accumulates.
# ═══════════════════════════════════════════════════════════════════════════

# Role meaning:
#   sharp        — passive market-maker, +22.7/leg @+30 ticks.  Our competitor.
#                  Avoid crossing spread into their quotes (only when forced
#                  to delta-hedge).  The AS quoter never aggresses, so this is
#                  automatic for HYDR/VEV market making.
#   mild_buyer   — passive, mildly informed (+5.5/leg).  Buys vouchers, posts
#                  voucher bids in the underlying.  Implication:
#                  voucher asks → likely lifted by them; widen ask.
#   vol_seller   — aggressive seller of high-K vouchers, near-neutral (-2.3/leg).
#                  Implication: voucher bids → likely hit by them; tighten bid.
#   noise        — losing per-leg consistently.  Mark 38 ($-28.9/leg) on
#                  HYDROGEL+VEV_4000, Mark 55 ($-11.3/leg) on VELVETFRUIT.
#                  Welcome them with default tight quotes.
#   neutral      — unknown Mark or insufficient signal.  Default behavior.
MARK_TIERS_DEFAULT: Dict[str, str] = {
    "Mark 14": "sharp",
    "Mark 01": "mild_buyer",
    "Mark 67": "mild_buyer",
    "Mark 22": "vol_seller",
    "Mark 49": "noise",
    "Mark 55": "noise",
    "Mark 38": "noise",
}


def is_mark(name: Optional[str]) -> bool:
    """A counterparty string is a Mark iff it begins with 'Mark '.  Anything
    else (None, our own agent name, anomalies) is filtered out."""
    return isinstance(name, str) and name.startswith("Mark ")


def get_mark_tier(mark_id: Optional[str]) -> str:
    """Step-2 spec: hard-coded from offline analysis.  Unknown → neutral."""
    if not is_mark(mark_id):
        return "neutral"
    return MARK_TIERS_DEFAULT.get(mark_id, "neutral")


def update_mark_stats(state: TradingState, td: dict) -> None:
    """Per-tick incremental update of per-Mark profile from market_trades.

    Tracked per Mark (in td['mark_stats']):
      - n           : total trade-legs observed
      - buy_n       : legs where Mark was the buyer
      - sell_n      : legs where Mark was the seller
      - spread_sum  : Σ (favorable spread per leg).  +ve = Mark captured the
                      spread (passive MM-like behavior); -ve = Mark paid the
                      spread (aggressive customer).  This is the live
                      corroborator of the hard-coded tier.
      - last_ts     : last timestamp at which we observed this Mark.

    NOTE: corroboration only.  Tier dispatch always uses the hard-coded
    MARK_TIERS_DEFAULT — a brief live window can't override 4,281 historical
    trade-legs of signal.  We just collect the data for post-mortem analysis.
    """
    stats = td.setdefault("mark_stats", {})
    market_trades = state.market_trades or {}
    for sym, trades in market_trades.items():
        od = state.order_depths.get(sym)
        # Use prior-tick mid where available; falls back to current mid.
        mid = _mid(od) if od is not None else None
        if mid is None:
            continue
        for trade in trades:
            for role, mark_id in (("buy", trade.buyer), ("sell", trade.seller)):
                if not is_mark(mark_id):
                    continue
                m = stats.setdefault(mark_id, {
                    "n": 0, "buy_n": 0, "sell_n": 0,
                    "spread_sum": 0.0, "last_ts": 0,
                })
                m["n"] += 1
                m["last_ts"] = state.timestamp
                if role == "buy":
                    m["buy_n"] += 1
                    # Mark bought at price.  If price > mid, they paid above
                    # mid (negative for them).  Capture: mid - price.
                    m["spread_sum"] += float(mid - trade.price)
                else:
                    m["sell_n"] += 1
                    m["spread_sum"] += float(trade.price - mid)


# ═══════════════════════════════════════════════════════════════════════════
# PURE PRICING / STATS  (unchanged from Round 3)
# ═══════════════════════════════════════════════════════════════════════════

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call.  r = q = 0; T in days; σ per-day."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _phi(d1) - K * _phi(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _phi(d1)


def bs_iv(price: float, S: float, K: float, T: float,
          lo: float = 1e-5, hi: float = 5.0, tol: float = 1e-5) -> float:
    intrinsic = max(0.0, S - K)
    if price < intrinsic - 1e-6 or price > S + 1e-6:
        return float("nan")
    f_lo = bs_call(S, K, T, lo) - price
    f_hi = bs_call(S, K, T, hi) - price
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = bs_call(S, K, T, mid) - price
        if abs(f) < tol:
            return mid
        if f > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def svi_w(k: float, params: dict = SVI_PARAMS) -> float:
    """Total variance under raw-SVI; Gatheral & Jacquier (2014) eq. (1)."""
    a = params["a"]; b = params["b"]; rho = params["rho"]; m = params["m"]; sg = params["sigma"]
    return a + b * (rho * (k - m) + math.sqrt((k - m) * (k - m) + sg * sg))


def svi_sigma(K: float, S: float, T: float, params: dict = SVI_PARAMS) -> float:
    if S <= 0 or T <= 0 or K <= 0:
        return SIGMA_FLOOR
    w = svi_w(math.log(K / S), params)
    if w <= 0:
        return SIGMA_FLOOR
    return max(SIGMA_FLOOR, min(SIGMA_CEIL, math.sqrt(w / T)))


def tte_days(timestamp: int, td: dict) -> float:
    start = td.get("tte_start", LIVE_TTE_DAY_START)
    progress = timestamp / DAY_LENGTH_TS
    return max(start - progress, 1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# AVELLANEDA-STOIKOV QUOTER  (unchanged from Round 3)
# Reference: Avellaneda & Stoikov (2008), eqs. (32)-(33).
# ═══════════════════════════════════════════════════════════════════════════

def avellaneda_stoikov(
    s: float, q: int, T_minus_t: float, *,
    gamma: float, sigma: float, k: float,
) -> Tuple[float, float]:
    var_remaining = sigma * sigma * max(0.0, T_minus_t)
    r = s - q * gamma * var_remaining
    inventory_term = gamma * var_remaining
    spread_term = (2.0 / gamma) * math.log(1.0 + gamma / max(k, 1e-9))
    half_spread = 0.5 * (inventory_term + spread_term)
    return r, half_spread


# ═══════════════════════════════════════════════════════════════════════════
# ORDER-BOOK / SAFETY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _best_bid(od: OrderDepth) -> Optional[int]:
    return max(od.buy_orders) if od.buy_orders else None


def _best_ask(od: OrderDepth) -> Optional[int]:
    return min(od.sell_orders) if od.sell_orders else None


def _mid(od: OrderDepth) -> Optional[float]:
    bb, ba = _best_bid(od), _best_ask(od)
    if bb is not None and ba is not None:
        return 0.5 * (bb + ba)
    if bb is not None:
        return float(bb)
    if ba is not None:
        return float(ba)
    return None


def _post_orders_safely(orders: List[Order], symbol: str, position: int) -> List[Order]:
    """Truncate per-side cumulative size so we never breach the position
    limit.  Earlier orders in the list have priority (used for hedge>MM)."""
    limit = LIMITS.get(symbol, 10**9)
    buy_room = max(0, limit - position)
    sell_room = max(0, limit + position)
    out: List[Order] = []
    used_buy = 0
    used_sell = 0
    for o in orders:
        if o.quantity > 0:
            allowed = max(0, min(o.quantity, buy_room - used_buy))
            if allowed > 0:
                out.append(Order(o.symbol, o.price, allowed))
                used_buy += allowed
        elif o.quantity < 0:
            allowed = -max(0, min(-o.quantity, sell_room - used_sell))
            if allowed < 0:
                out.append(Order(o.symbol, o.price, allowed))
                used_sell += -allowed
    return out


# ═══════════════════════════════════════════════════════════════════════════
# TRADER STATE
# ═══════════════════════════════════════════════════════════════════════════

def _initial_sigma_seed() -> Dict[int, float]:
    """Seed σ per strike from Round-4 historical median IV per strike
    (notebooks/round4_eda.py).  Values are stable across the 3 days, so
    averaging them into a single seed is fine.  EWMA refines per tick."""
    return dict(HISTORICAL_SIGMA)


def _default_td() -> dict:
    return {
        "last_timestamp": -1,
        "tte_start": LIVE_TTE_DAY_START,
        "last_mid": {HYDROGEL: None, UNDERLYING: None},
        "sigma": _initial_sigma_seed(),
        "mark_stats": {},                      # NEW for Round 4
    }


def _load_td(traderData: str) -> dict:
    if not traderData:
        return _default_td()
    try:
        td = json.loads(traderData)
    except (TypeError, ValueError):
        return _default_td()
    base = _default_td()
    for k, v in base.items():
        td.setdefault(k, v)
    if "sigma" in td:
        td["sigma"] = {int(k): float(v) for k, v in td["sigma"].items()}
    return td


# ═══════════════════════════════════════════════════════════════════════════
# THE TRADER
# ═══════════════════════════════════════════════════════════════════════════

class Trader:

    # Manual-challenge bid (Round 2 carryover; harmless if unused).
    def bid(self) -> int:
        return 100

    # ──────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        td = _load_td(state.traderData)

        # New-day detection: timestamp resets to 0 at boundaries.
        last_ts = td.get("last_timestamp", -1)
        if last_ts > 0 and state.timestamp < last_ts:
            td["tte_start"] = max(td["tte_start"] - 1, 1.0)
            td["last_mid"][HYDROGEL] = None
            td["last_mid"][UNDERLYING] = None
        td["last_timestamp"] = state.timestamp

        # NEW Round-4: update per-Mark profile (corroboration, not control).
        update_mark_stats(state, td)

        result: Dict[str, List[Order]] = {}

        # 1) HYDROGEL_PACK — A-S quoter, no cross-product entanglement.
        # Counterparty layer: AS posts passive limits ⇒ our fills come from
        # AGGRESSIVE Marks (per Round-4 EDA: Mark 38, 55, 67, 22).  Mark 14
        # is also passive; we don't cross spread into them.  No tier-conditional
        # adjustment is needed at quote time — the structure does the filtering.
        if HYDROGEL in state.order_depths:
            pos = state.position.get(HYDROGEL, 0)
            orders = self._trade_as_quoter(
                HYDROGEL, state.order_depths[HYDROGEL], pos,
                state.timestamp, td,
                gamma=GAMMA_HYDROGEL, sigma=SIGMA_HYDROGEL, k=K_HYDROGEL,
                target_position=0,
            )
            result[HYDROGEL] = _post_orders_safely(orders, HYDROGEL, pos)

        # 2) VOUCHERS — vol-arb against BS(σ_K from SVI+EWMA).
        #    Asymmetric edges for high-K (≥5300) per Mark 22/Mark 01 dynamic.
        voucher_orders, voucher_delta_planned = self._trade_vouchers(state, td)
        for sym, ords in voucher_orders.items():
            pos = state.position.get(sym, 0)
            result[sym] = _post_orders_safely(ords, sym, pos)

        # 3) VELVETFRUIT_EXTRACT — A-S quoter (target = -voucher_delta_inventory)
        #    AND delta-hedging order, both posted at the underlying.
        if UNDERLYING in state.order_depths:
            pos = state.position.get(UNDERLYING, 0)
            voucher_delta_inv = self._voucher_delta_inventory(state, td)
            target_pos = int(round(-(voucher_delta_inv + voucher_delta_planned)))
            target_pos = max(-LIMITS[UNDERLYING], min(LIMITS[UNDERLYING], target_pos))

            mm_orders = self._trade_as_quoter(
                UNDERLYING, state.order_depths[UNDERLYING], pos,
                state.timestamp, td,
                gamma=GAMMA_VELVETFRUIT, sigma=SIGMA_VELVETFRUIT, k=K_VELVETFRUIT,
                target_position=target_pos,
            )
            hedge_orders = self._delta_hedge_orders(state, voucher_delta_planned, td)
            combined = hedge_orders + mm_orders
            result[UNDERLYING] = _post_orders_safely(combined, UNDERLYING, pos)

        return result, 0, json.dumps(td)

    # ──────────────────────────────────────────────────────────────────────
    # AVELLANEDA-STOIKOV QUOTER (unchanged from Round 3)
    # ──────────────────────────────────────────────────────────────────────
    def _trade_as_quoter(
        self,
        symbol: str,
        od: OrderDepth,
        position: int,
        timestamp: int,
        td: dict,
        *,
        gamma: float,
        sigma: float,
        k: float,
        target_position: int,
    ) -> List[Order]:
        """Avellaneda-Stoikov bid/ask quoter, multi-level for queue priority.

        Skews reservation around (position − target_position) so the same
        function works for plain MM (target=0) and for an MM that cooperates
        with an external hedger (target = −voucher_delta_inv)."""
        orders: List[Order] = []
        s = _mid(od)
        if s is None:
            return orders

        T_minus_t = max(0.0, (DAY_LENGTH_TS - timestamp) / DAY_LENGTH_TS)

        r, delta_half = avellaneda_stoikov(
            s, q=(position - target_position), T_minus_t=T_minus_t,
            gamma=gamma, sigma=sigma, k=k,
        )

        cap = MAX_HALF_SPREAD.get(symbol, 10)
        floor = MIN_HALF_SPREAD.get(symbol, 1)
        delta_half = max(floor, min(cap, delta_half))

        # End-of-day flatten — pull reservation toward target as ts → 1M.
        if timestamp > EOD_FLATTEN_LINEAR_START_TS:
            tighten = (timestamp - EOD_FLATTEN_LINEAR_START_TS) / 50_000  # 0…1
            if position > target_position:
                delta_half = max(floor, delta_half * (1 - 0.6 * tighten))
            elif position < target_position:
                delta_half = max(floor, delta_half * (1 - 0.6 * tighten))

        bid0 = int(math.floor(r - delta_half))
        ask0 = int(math.ceil(r + delta_half))
        if ask0 <= bid0:
            ask0 = bid0 + 1

        limit = LIMITS[symbol]
        buy_cap = max(0, limit - position)
        sell_cap = max(0, limit + position)
        total = MM_TOTAL_SIZE.get(symbol, 100)

        per_side_buy = min(buy_cap, total)
        per_side_sell = min(sell_cap, total)

        if per_side_buy > 0:
            n = MM_LEVELS
            per_lvl = max(1, per_side_buy // n)
            remaining = per_side_buy
            for i in range(n):
                if remaining <= 0:
                    break
                vol = per_lvl if i < n - 1 else remaining
                vol = min(vol, remaining)
                price = bid0 - i
                orders.append(Order(symbol, price, vol))
                remaining -= vol

        if per_side_sell > 0:
            n = MM_LEVELS
            per_lvl = max(1, per_side_sell // n)
            remaining = per_side_sell
            for i in range(n):
                if remaining <= 0:
                    break
                vol = per_lvl if i < n - 1 else remaining
                vol = min(vol, remaining)
                price = ask0 + i
                price = max(price, bid0 + 1)
                orders.append(Order(symbol, price, -vol))
                remaining -= vol

        return orders

    # ──────────────────────────────────────────────────────────────────────
    # VOUCHER VOL-ARB (Round-4: asymmetric edges for high-K)
    # ──────────────────────────────────────────────────────────────────────
    def _trade_vouchers(
        self, state: TradingState, td: dict
    ) -> Tuple[Dict[str, List[Order]], float]:
        out: Dict[str, List[Order]] = {sym: [] for sym in ALL_VOUCHERS}
        planned_delta = 0.0

        und = state.order_depths.get(UNDERLYING)
        S = _mid(und) if und else None
        if S is None:
            return out, 0.0
        T = tte_days(state.timestamp, td)
        eod_active = state.timestamp >= EOD_FLATTEN_VOUCHERS_START_TS

        # Cold-start gate: still update σ EWMA below, but DO NOT TRADE.
        # We hit this branch only at start of day before EWMA has converged
        # from the seed.  EOD positions still flatten regardless.
        cold_start = (state.timestamp < VOUCHER_COLD_START_TICKS) and not eod_active

        for K in ACTIVE_VOUCHER_STRIKES:
            sym = VOUCHER_SYMS[K]
            od = state.order_depths.get(sym)
            if od is None:
                continue
            bb, ba = _best_bid(od), _best_ask(od)
            if bb is None and ba is None:
                continue

            sigma_k = float(td["sigma"].get(K, svi_sigma(K, S, T)))
            if bb is not None and ba is not None:
                mkt_mid = 0.5 * (bb + ba)
                iv_now = bs_iv(mkt_mid, S, K, T)
                if iv_now == iv_now:                       # NaN guard
                    sigma_k = (1 - SIGMA_EWMA_ALPHA) * sigma_k + SIGMA_EWMA_ALPHA * iv_now
                    sigma_k = min(SIGMA_CEIL, max(SIGMA_FLOOR, sigma_k))
                    td["sigma"][K] = sigma_k

            model_price = bs_call(S, K, T, sigma_k)
            spread = (ba - bb) if (bb is not None and ba is not None) else 4
            base_edge = max(VOUCHER_MIN_EDGE, VOUCHER_EDGE_FRAC_OF_SPREAD * spread)

            # NEW Round-4: asymmetric edges for high-K vouchers (≥5300).
            # Mark 22 (sells those, near-neutral) ↔ Mark 01 (buys, mild informed).
            # Glosten-Milgrom (1985): adjust adverse-selection floor by counterparty.
            if K >= HIGH_K_VOUCHER_THRESHOLD:
                edge_buy = base_edge * HIGH_K_EDGE_BUY_MULT
                edge_sell = base_edge * HIGH_K_EDGE_SELL_MULT
            else:
                edge_buy = base_edge
                edge_sell = base_edge

            pos = state.position.get(sym, 0)
            limit = LIMITS[sym]
            buy_room = limit - pos
            sell_room = limit + pos
            d_per = bs_delta(S, K, T, sigma_k)

            # EOD voucher flatten — DISABLED in round-4 v3.  Letting positions
            # mark at mid (free) is strictly better than crossing the spread
            # to liquidate.  Kept the code path behind a flag in case the
            # multi-day live evaluation reintroduces overnight gap risk.
            if eod_active and pos != 0 and EOD_VOUCHER_FLATTEN_ENABLED:
                if pos > 0 and bb is not None:
                    qty = -min(pos, od.buy_orders[bb])
                    if qty < 0:
                        out[sym].append(Order(sym, bb, qty))
                        planned_delta += qty * d_per
                elif pos < 0 and ba is not None:
                    qty = min(-pos, -od.sell_orders[ba])
                    if qty > 0:
                        out[sym].append(Order(sym, ba, qty))
                        planned_delta += qty * d_per
                continue

            # Cold-start: σ EWMA still updating above (so it converges) but no
            # trades posted.  Eliminates the early-day mispricing risk window.
            if cold_start:
                continue

            # Buy if market ask is cheap relative to model
            if ba is not None and ba < model_price - edge_buy and buy_room > 0:
                avail = -od.sell_orders[ba]
                size = min(VOUCHER_MAX_TRADE_SIZE, buy_room, avail)
                if size > 0:
                    out[sym].append(Order(sym, ba, size))
                    planned_delta += size * d_per

            # Sell if market bid is rich relative to model
            if bb is not None and bb > model_price + edge_sell and sell_room > 0:
                avail = od.buy_orders[bb]
                size = min(VOUCHER_MAX_TRADE_SIZE, sell_room, avail)
                if size > 0:
                    out[sym].append(Order(sym, bb, -size))
                    planned_delta += -size * d_per

        return out, planned_delta

    # ──────────────────────────────────────────────────────────────────────
    # VOUCHER DELTA INVENTORY
    # ──────────────────────────────────────────────────────────────────────
    def _voucher_delta_inventory(self, state: TradingState, td: dict) -> float:
        und = state.order_depths.get(UNDERLYING)
        S = _mid(und) if und else None
        if S is None:
            return 0.0
        T = tte_days(state.timestamp, td)
        total = 0.0
        for K in STRIKES:
            sym = VOUCHER_SYMS[K]
            pos = state.position.get(sym, 0)
            if pos == 0:
                continue
            sigma_k = float(td["sigma"].get(K, svi_sigma(K, S, T)))
            total += pos * bs_delta(S, K, T, sigma_k)
        return total

    # ──────────────────────────────────────────────────────────────────────
    # DELTA HEDGE ON VELVETFRUIT_EXTRACT
    # ──────────────────────────────────────────────────────────────────────
    def _delta_hedge_orders(
        self, state: TradingState, planned_voucher_delta: float, td: dict
    ) -> List[Order]:
        und = state.order_depths.get(UNDERLYING)
        if und is None:
            return []
        S = _mid(und)
        if S is None:
            return []

        net_delta = float(state.position.get(UNDERLYING, 0))
        net_delta += self._voucher_delta_inventory(state, td)
        net_delta += planned_voucher_delta

        if abs(net_delta) <= DELTA_DEAD_ZONE:
            return []

        target = int(round(-net_delta))
        if target > 0:
            ba = _best_ask(und)
            if ba is None:
                return []
            avail = -und.sell_orders[ba]
            qty = min(target, avail, MM_TOTAL_SIZE.get(UNDERLYING, 80))
            if qty <= 0:
                return []
            return [Order(UNDERLYING, ba, qty)]
        else:
            bb = _best_bid(und)
            if bb is None:
                return []
            avail = und.buy_orders[bb]
            qty = min(-target, avail, MM_TOTAL_SIZE.get(UNDERLYING, 80))
            if qty <= 0:
                return []
            return [Order(UNDERLYING, bb, -qty)]
