"""
Round 3 trader v2 — HYDROGEL_PACK, VELVETFRUIT_EXTRACT, and 10 VEV_<K> vouchers.

CHANGES FROM v1
  - Voucher fair-value: σ_K is now seeded from a raw-SVI smile fit
    [Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility surfaces"]
    instead of ad-hoc per-strike historical means. SVI is evaluated, not
    fitted, in the trader (params from offline notebooks/round3_eda_v2.py).
    Per-strike σ_K still drifts via EWMA, so the trader auto-corrects if
    live regime differs from historical.
  - HYDROGEL_PACK and VELVETFRUIT_EXTRACT use an Avellaneda-Stoikov quoter
    [Avellaneda & Stoikov (2008)], with γ chosen so max-position skew is
    ~5 ticks and k fitted from the trade tape (T4 in v2 EDA).
  - Pairs trading: tested via Engle-Granger (T2). Cointegration was only
    significant on day 0 and broke on days 1/2 — DROPPED.
  - Hedge cooperation: VELVETFRUIT MM skews its reservation around
    target_pos = -voucher_delta_inventory so the MM and the delta hedger
    don't fight each other. This is the v1 fix (kept).
  - EOD flatten window unchanged.

DESIGN
  HYDROGEL_PACK (limit 200)
    A-S quoter; γ_HYDR, k_HYDR from EDA T4.

  VELVETFRUIT_EXTRACT (limit 200)
    A-S quoter; γ_VEV, k_VEV from EDA T4. target_pos = -voucher_delta.

  Vouchers (limit 300/strike)
    Active universe: K ∈ {5000, 5100, 5200, 5300, 5400, 5500} (S2 in v1 EDA).
    σ_K(0) = sqrt(SVI_w(ln(K/S₀)) / T_today).
    σ_K(t) = (1−α)σ_K(t−1) + α·observed_iv_K(t).
    Trade when |market − BS(S, K, T, σ_K)| > δ_K.
    Net portfolio delta hedged on VELVETFRUIT_EXTRACT.

CONVENTIONS
  r = 0; T in days (no annualization); σ in per-day units.
  Live round 3: TTE_start = 5 days; one round = 1M timestamps.

SAFETY
  - `_post_orders_safely` enforces position limits.
  - All numerical paths handle empty books, NaN IVs, and SVI floor < 0.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

HYDROGEL = "HYDROGEL_PACK"
UNDERLYING = "VELVETFRUIT_EXTRACT"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHER_SYMS: Dict[int, str] = {k: f"VEV_{k}" for k in STRIKES}
ALL_VOUCHERS = tuple(VOUCHER_SYMS.values())
ACTIVE_VOUCHER_STRIKES = (5000, 5100, 5200, 5300, 5400, 5500)

LIMITS: Dict[str, int] = {
    HYDROGEL: 200,
    UNDERLYING: 200,
    **{sym: 300 for sym in ALL_VOUCHERS},
}

DAY_LENGTH_TS = 1_000_000
LIVE_TTE_DAY_START = 5  # days at start of round 3

# ───────────────────────────────────────────────────────────────────────────
# Raw-SVI parameters from offline fit (notebooks/round3_eda_v2.py, T5).
# Picked day 2 (lowest RMSE 0.000331 in total-variance space; closest to
# live TTE = 5 days). Form: w(k) = a + b·(ρ·(k-m) + sqrt((k-m)² + σ²))
# where k = ln(K/S) and w = σ_BS² · T (total variance).
# Reference: Gatheral & Jacquier (2014), eq. (1).
SVI_PARAMS = {
    "a":     -0.000669,
    "b":      0.02165,
    "rho":    0.444,
    "m":      0.012,
    "sigma":  0.0659,
}

# Hard floor / ceiling on per-strike σ EWMA so a single corrupted IV
# can't break us. Range chosen to cover all observed IVs (S3 in v1 EDA: 0.012–0.031).
SIGMA_FLOOR = 0.005
SIGMA_CEIL = 0.06

# ───────────────────────────────────────────────────────────────────────────
# Avellaneda-Stoikov parameters
# Reference: Avellaneda & Stoikov (2008), "High-frequency trading in a limit
# order book", Quant. Finance 8(3), eqs. (32)-(33).
#
# Reservation: r(s, q, t) = s − q·γ·σ²·(T−t)
# Optimal half-spread (each side):
#   δ* = (1/2)·γ·σ²·(T−t) + (1/γ)·ln(1 + γ/k)
#
# Calibration (EDA T3, T4):
#   HYDROGEL:    σ_per_day ≈ 0.0217 → in price terms σ_HYDR ≈ 9990·0.0217 = 217/day
#                k̂ ≈ 0.128 (mean trade-mid offset 7.84)
#   VELVETFRUIT: σ_per_day ≈ 0.0216 → σ_VEV ≈ 5250·0.0216 = 113/day
#                k̂ ≈ 0.443 (mean trade-mid offset 2.26)
#
# γ chosen so q·γ·σ²·(T−t) at q=200 (limit) gives ~5-tick max skew.
#   HYDROGEL:    γ_HYDR = 5 / (200 · 217² · 1) = 5.3e-7  → use 5e-7
#   VELVETFRUIT: γ_VEV  = 5 / (200 · 113² · 1) = 1.96e-6 → use 2e-6
# Tuned by parameter sweep (backtesting/round3_sweep.py, phase 3 winner).
# Baseline → tuned: 7,390 → 10,979 PnL on 3 days (+49%). See
# writeups/round3_sweep_results.json for full provenance.
# WARNING: tuned on the same 3 historical days the trader will not see live;
# expect some shrinkage. The pre-sweep defaults are in the comments below
# for reference — fall back to those if live PnL diverges sharply.
GAMMA_HYDROGEL = 2.1e-7         # was 5e-7 (lower → less skew, more spread captured)
GAMMA_VELVETFRUIT = 6.8e-7      # was 2e-6
SIGMA_HYDROGEL = 217.0
SIGMA_VELVETFRUIT = 113.0
K_HYDROGEL = 0.128
K_VELVETFRUIT = 0.443

# A-S theoretical δ* may be wider than the empirical best-bid/ask gap; we cap
# the half-spread so we don't quote outside the visible book.
MAX_HALF_SPREAD = {
    HYDROGEL: 9,        # was 8
    UNDERLYING: 2,      # was 3 (tighter quoting on underlying)
}
MIN_HALF_SPREAD = {
    HYDROGEL: 1,
    UNDERLYING: 1,
}

MM_LEVELS = 4
MM_TOTAL_SIZE = {
    HYDROGEL: 178,      # was 120 (bigger book → more fills at favorable A-S spread)
    UNDERLYING: 174,    # was 80 (bigger size, paired with looser hedge dead zone)
}

# ───────────────────────────────────────────────────────────────────────────
# Voucher vol-arb tunables

# Tuned: VOUCHER_MIN_EDGE=4.24 came out of sweep (was 2.0). Larger edge means
# we trade vouchers only on real mispricings; sweep showed +1100 PnL from this
# alone. EDGE_FRAC stays at 0.5 (sweep was indifferent within ±300).
VOUCHER_MIN_EDGE = 4.24
VOUCHER_EDGE_FRAC_OF_SPREAD = 0.5
VOUCHER_MAX_TRADE_SIZE = 25     # was 10 (larger trades when edge is real)

# α from sweep: 0.0026 (was 0.005). Slower σ drift = more edge persistence.
SIGMA_EWMA_ALPHA = 0.0026

# DELTA_DEAD_ZONE from sweep: 75 (was 25). Bigger band = far fewer hedges,
# we accept ~75 units of net delta drift in exchange for not paying spread.
# Voucher EOD flatten still removes accumulated exposure.
DELTA_DEAD_ZONE = 75.0

# EOD flatten windows
EOD_FLATTEN_VOUCHERS_START_TS = 950_000
EOD_FLATTEN_LINEAR_START_TS = 950_000


# ═══════════════════════════════════════════════════════════════════════════
# PURE PRICING / STATS
# ═══════════════════════════════════════════════════════════════════════════

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call. r = q = 0; T in days; σ per-day."""
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
    """Bisection IV solver. NaN on no-arb violation."""
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
    """Total variance under raw-SVI; Gatheral & Jacquier (2014) eq. (1).
    Returns w(k) = a + b·(ρ(k-m) + sqrt((k-m)² + σ²)). Strictly ≥ 0 when
    SVI parameters satisfy the no-arb floor a + b·σ·sqrt(1-ρ²) ≥ 0."""
    a = params["a"]; b = params["b"]; rho = params["rho"]; m = params["m"]; sg = params["sigma"]
    return a + b * (rho * (k - m) + math.sqrt((k - m) * (k - m) + sg * sg))


def svi_sigma(K: float, S: float, T: float, params: dict = SVI_PARAMS) -> float:
    """σ_BS implied by SVI at strike K with spot S, time T (days).
    Falls back to a small floor if SVI returns negative variance (numerical)."""
    if S <= 0 or T <= 0 or K <= 0:
        return SIGMA_FLOOR
    w = svi_w(math.log(K / S), params)
    if w <= 0:
        return SIGMA_FLOOR
    return max(SIGMA_FLOOR, min(SIGMA_CEIL, math.sqrt(w / T)))


def tte_days(timestamp: int, td: dict) -> float:
    """Linear decay across the live day. Cached start TTE in traderData."""
    start = td.get("tte_start", LIVE_TTE_DAY_START)
    progress = timestamp / DAY_LENGTH_TS
    return max(start - progress, 1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# AVELLANEDA-STOIKOV QUOTER
# Reference: Avellaneda & Stoikov (2008), eqs. (32)-(33).
# ═══════════════════════════════════════════════════════════════════════════

def avellaneda_stoikov(
    s: float, q: int, T_minus_t: float, *,
    gamma: float, sigma: float, k: float,
) -> Tuple[float, float]:
    """Return (reservation_price, half_spread) for a market maker holding
    inventory q with mid s, remaining round-time T-t (in days), risk
    aversion γ, mid-price diffusion σ (per day, in price units), and
    LOB arrival rate k.

      r = s − q·γ·σ²·(T−t)                   [eq. 32]
      δ_total = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/k)
      δ_half  = δ_total / 2
    """
    var_remaining = sigma * sigma * max(0.0, T_minus_t)
    r = s - q * gamma * var_remaining
    inventory_term = gamma * var_remaining
    spread_term = (2.0 / gamma) * math.log(1.0 + gamma / max(k, 1e-9))
    half_spread = 0.5 * (inventory_term + spread_term)
    return r, half_spread


# ═══════════════════════════════════════════════════════════════════════════
# ORDER-BOOK HELPERS
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


# ═══════════════════════════════════════════════════════════════════════════
# POSITION-LIMIT GUARD
# ═══════════════════════════════════════════════════════════════════════════

def _post_orders_safely(orders: List[Order], symbol: str, position: int) -> List[Order]:
    """Truncate per-side cumulative size so we never breach the position limit.
    Earlier orders in the list have priority (used for hedge > MM ordering)."""
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
# TRADER STATE (traderData JSON schema)
# ═══════════════════════════════════════════════════════════════════════════

def _initial_sigma_seed() -> Dict[int, float]:
    """Seed σ per strike from the SVI fit, evaluated at S₀ ≈ 5250
    (mean of historical VELVETFRUIT mids) and T = LIVE_TTE_DAY_START.
    Live trader will refine these via EWMA on observed IVs."""
    S0 = 5250.0
    T0 = LIVE_TTE_DAY_START
    return {K: svi_sigma(K, S0, T0) for K in STRIKES}


def _default_td() -> dict:
    return {
        "last_timestamp": -1,
        "tte_start": LIVE_TTE_DAY_START,
        "last_mid": {HYDROGEL: None, UNDERLYING: None},
        "sigma": _initial_sigma_seed(),  # per-strike σ EWMA
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
            # Note: do NOT reset td["sigma"] — empirically the day-0 voucher PnL
            # was *not* vol-arb but long-delta exposure during an upward S drift.
            # Resetting σ on day boundary forces re-trading on days 1+ where
            # the drift didn't pay off, costing -$8K in 3-day backtest. Keep
            # the EWMA carry-over so days 1+ stay quiet.
        td["last_timestamp"] = state.timestamp

        result: Dict[str, List[Order]] = {}

        # 1) HYDROGEL_PACK — A-S quoter, no cross-product entanglement.
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
    # AVELLANEDA-STOIKOV QUOTER (replaces v1's _trade_mm)
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

        # Use round-fraction remaining as A-S's T-t (per Step 1 confirmation).
        T_minus_t = max(0.0, (DAY_LENGTH_TS - timestamp) / DAY_LENGTH_TS)

        r, delta_half = avellaneda_stoikov(
            s, q=(position - target_position), T_minus_t=T_minus_t,
            gamma=gamma, sigma=sigma, k=k,
        )

        # Cap A-S half-spread to empirical bounds (T4): keeps us inside the
        # bot quotes when γ/σ²(T-t) blow it up at start of day.
        cap = MAX_HALF_SPREAD.get(symbol, 10)
        floor = MIN_HALF_SPREAD.get(symbol, 1)
        delta_half = max(floor, min(cap, delta_half))

        # End-of-day flatten — pull reservation toward target as ts → 1M.
        if timestamp > EOD_FLATTEN_LINEAR_START_TS:
            tighten = (timestamp - EOD_FLATTEN_LINEAR_START_TS) / 50_000  # 0…1
            # Squeeze quotes inward on the side that pushes us toward target
            if position > target_position:
                delta_half = max(floor, delta_half * (1 - 0.6 * tighten))
            elif position < target_position:
                delta_half = max(floor, delta_half * (1 - 0.6 * tighten))

        bid0 = int(math.floor(r - delta_half))
        ask0 = int(math.ceil(r + delta_half))
        if ask0 <= bid0:
            ask0 = bid0 + 1  # never self-cross

        limit = LIMITS[symbol]
        buy_cap = max(0, limit - position)
        sell_cap = max(0, limit + position)
        total = MM_TOTAL_SIZE.get(symbol, 100)

        per_side_buy = min(buy_cap, total)
        per_side_sell = min(sell_cap, total)

        # Multi-level quote: spread orders across MM_LEVELS prices, getting
        # progressively further from r.
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
                # Make sure we never cross our own bids
                price = max(price, bid0 + 1)
                orders.append(Order(symbol, price, -vol))
                remaining -= vol

        return orders

    # ──────────────────────────────────────────────────────────────────────
    # VOUCHER VOL-ARB (unchanged in spirit from v1; σ seeded from SVI now)
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

        for K in ACTIVE_VOUCHER_STRIKES:
            sym = VOUCHER_SYMS[K]
            od = state.order_depths.get(sym)
            if od is None:
                continue
            bb, ba = _best_bid(od), _best_ask(od)
            if bb is None and ba is None:
                continue

            # σ_K EWMA update from market mid IV
            sigma_k = float(td["sigma"].get(K, svi_sigma(K, S, T)))
            if bb is not None and ba is not None:
                mkt_mid = 0.5 * (bb + ba)
                iv_now = bs_iv(mkt_mid, S, K, T)
                if iv_now == iv_now:
                    sigma_k = (1 - SIGMA_EWMA_ALPHA) * sigma_k + SIGMA_EWMA_ALPHA * iv_now
                    sigma_k = min(SIGMA_CEIL, max(SIGMA_FLOOR, sigma_k))
                    td["sigma"][K] = sigma_k

            model_price = bs_call(S, K, T, sigma_k)
            spread = (ba - bb) if (bb is not None and ba is not None) else 4
            edge = max(VOUCHER_MIN_EDGE, VOUCHER_EDGE_FRAC_OF_SPREAD * spread)

            pos = state.position.get(sym, 0)
            limit = LIMITS[sym]
            buy_room = limit - pos
            sell_room = limit + pos
            d_per = bs_delta(S, K, T, sigma_k)

            # EOD: liquidate at touch
            if eod_active and pos != 0:
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

            # Buy if market ask is cheap relative to model
            if ba is not None and ba < model_price - edge and buy_room > 0:
                avail = -od.sell_orders[ba]
                size = min(VOUCHER_MAX_TRADE_SIZE, buy_room, avail)
                if size > 0:
                    out[sym].append(Order(sym, ba, size))
                    planned_delta += size * d_per

            # Sell if market bid is rich relative to model
            if bb is not None and bb > model_price + edge and sell_room > 0:
                avail = od.buy_orders[bb]
                size = min(VOUCHER_MAX_TRADE_SIZE, sell_room, avail)
                if size > 0:
                    out[sym].append(Order(sym, bb, -size))
                    planned_delta += -size * d_per

        return out, planned_delta

    # ──────────────────────────────────────────────────────────────────────
    # VOUCHER DELTA INVENTORY (sum of pos·delta over current voucher pos)
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
