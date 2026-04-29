"""
Round 5 — Trader v1.5.

= v1 (trend + AR(1), AR(1) enabled) + the two safety nets from v2 that were
silent on backtest:
  - EOD flatten at ts >= 999,800 (P4 from v2).
  - Per-product MtM kill-switch at −$10,000 (P5 from v2).

Rationale: the v1→v2 changelog showed v2's safety nets contributed near-zero
backtest delta on their own — the only thing that moved the v1 vs v2 numbers
was disabling AR(1) (P3b), which in retrospect was the wrong call (AR(1)
appears to have contributed positively on live day 5). v1.5 keeps the v1
alpha surface intact and adds only the defensive infrastructure.

Strategy (unchanged from v1):
  Edge 1: TREND-FOLLOW on 7 products with stable directional drift over 3 historical
          days. Direction is hardcoded from the offline screen (PANEL_2X4 +21%
          across days 2-4, PEBBLES_XS −50%, etc.). Adaptive sizing: position scales
          with trailing-return confirmation, so decelerators get a smaller position
          if the trend exhausts on day 5 and we automatically flatten if trend
          reverses entirely.

  Edge 2: AR(1) CONTRARIAN MM on 3 products with stable lag-1 negative
          autocorrelation (ROBOT_IRONING −0.117, OXYGEN_SHAKE_EVENING_BREATH
          −0.112, OXYGEN_SHAKE_CHOCOLATE −0.076). Tight quotes with skew biased
          opposite to the last tick's return.

References:
  - Lo & MacKinlay (1988). Variance ratio test — basis for screening.
  - Box & Jenkins (1976). Lag-1 ACF for AR(1) detection.
  - Avellaneda & Stoikov (2008). Quoting framework for AR(1) strategy.
  - Lehmann (1990) and Jegadeesh-Titman (1993) considered but rejected after
    per-day stability checks falsified cross-sectional reversal/momentum signals.

Architecture: single file, position limit 10 per product hardcoded, stdlib only,
feature flags at the top to disable any individual product live without re-uploading.
"""
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


# ── Constants pre-computed from offline screening (do NOT re-fit at runtime) ──

POSITION_LIMIT = 10  # per Round 5 brief: hard cap per product

# Trend-follower products with hardcoded expected direction.
# Each entry: product → (+1 expected long; -1 expected short).
# Direction inferred from cumulative drift over historical days 2-4.
TREND_DIRECTIONS: Dict[str, int] = {
    "MICROCHIP_OVAL":              -1,  # -59% (accelerating down)
    "PEBBLES_XS":                  -1,  # -50%
    "UV_VISOR_AMBER":              -1,  # -34% (decelerating)
    "PEBBLES_S":                   -1,  # -22%
    "OXYGEN_SHAKE_GARLIC":         +1,  # +33%
    "GALAXY_SOUNDS_BLACK_HOLES":   +1,  # +30%
    "PANEL_2X4":                   +1,  # +21% (steady, cleanest)
}

# AR(1) products with stable, statistically significant negative lag-1 AC.
# Value is the AC1 magnitude (negative; passes per-day stability check).
AR1_AC1: Dict[str, float] = {
    "ROBOT_IRONING":               -0.117,
    "OXYGEN_SHAKE_EVENING_BREATH": -0.112,
    "OXYGEN_SHAKE_CHOCOLATE":      -0.076,
}

# Feature flags — set to False to disable any individual product.
ENABLED: Dict[str, bool] = {p: True for p in
                            list(TREND_DIRECTIONS) + list(AR1_AC1)}

# Strategy params
# Trend strategy: hold to ±limit by default; flatten only on catastrophic reversal.
# Adaptive-confirmation logic was tested and lost ~$500k in backtest because tick-level
# noise dominates the slow drift signal at short lookbacks (signal:noise ~0.13 at K=500).
# Hold + hard reversal stop is the right architecture given the position-limit-10
# constraint: each position flip pays the bid-ask spread, so flipping must be rare.
TREND_REVERSAL_LOOKBACK = 2000     # ticks for reversal detection (~1/5 of a day)
TREND_REVERSAL_THRESHOLD = 0.10    # 10% trailing move opposite to direction → flatten.
                                   # Tuned via sweep: 0.05 caused ~$28k churn on PEBBLES_S
                                   # (per-2000-tick noise σ≈7.6%, threshold below 1σ).
                                   # 0.10 is ~1.3σ — fires on real reversal, not noise.
TREND_MAX_HISTORY = 2100           # cap per trender (TREND_REVERSAL_LOOKBACK + buffer)
AR1_MAX_HISTORY = 4

# v1.5 (from v2 P4): EOD flatten. Day timestamps run 0-999,900 in steps of 100
# (10,000 ticks per day). Trigger at the last 2 ticks so we don't burn alpha
# during the still-active trend in the final 1% of the day.
EOD_FLATTEN_TS = 999_800

# v1.5 (from v2 P5): per-product MtM kill-switch. If a trend product's running
# PnL (cash + position × mid, accumulated since start of day) drops below
# KILL_SWITCH_THRESHOLD, that product is disabled for the remainder of the
# day. Threshold of −$10,000 keeps the switch silent on historical data
# (worst observed normal entry drawdown was −$8,600); only fires on genuinely
# unrecoverable scenarios. Cash counter approximates fill price as the
# current tick's mid since own_trades isn't surfaced in the backtest harness.
KILL_SWITCH_THRESHOLD = -10_000.0


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def bid(self) -> int:
        """MAF was Round 2 only; Round 5 brief explicitly says bid() is ignored."""
        return 0

    def run(self, state: TradingState):
        td = self._load_state(state.traderData)

        # v1.5: reset MtM accounting on day boundary (timestamp wraps to 0).
        if state.timestamp < td.get("last_timestamp", -1):
            td["cash"] = {}
            td["prev_pos"] = {}
            td["killed"] = []
        td["last_timestamp"] = state.timestamp

        # v1.5: EOD flatten — short-circuits all per-product strategies.
        if state.timestamp >= EOD_FLATTEN_TS:
            return self._flatten_all_positions(state), 0, json.dumps(td)

        # v1.5: update per-product MtM equity from position deltas; flag killers.
        killed = self._update_kill_switch(td, state)

        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            if not ENABLED.get(product, False):
                continue
            if product in killed:
                continue
            position = state.position.get(product, 0)

            if product in TREND_DIRECTIONS:
                orders = self._trend_strategy(product, depth, position, td)
            elif product in AR1_AC1:
                orders = self._ar1_strategy(product, depth, position, td)
            else:
                orders = []

            if orders:
                result[product] = orders

        return result, 0, json.dumps(td)

    # ── Kill-switch ───────────────────────────────────────────────────────

    def _update_kill_switch(self, td: dict, state: TradingState) -> set:
        """Track per-product MtM equity since start of day; return killed set.

        Cash counter approximation: own_trades isn't surfaced in the backtest
        harness, so each position delta is attributed to the current tick's
        mid. equity = cash + position × mid. When equity < KILL_SWITCH_THRESHOLD
        the product is added to td['killed'] (persisted across ticks via
        traderData) and skipped for the rest of the day.
        """
        cash = td.setdefault("cash", {})
        prev_pos = td.setdefault("prev_pos", {})
        killed = set(td.setdefault("killed", []))

        for product in TREND_DIRECTIONS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            mid = self._get_mid(depth)
            if mid is None:
                continue
            new_pos = state.position.get(product, 0)
            old_pos = prev_pos.get(product, 0)
            delta = new_pos - old_pos
            cash[product] = cash.get(product, 0.0) - delta * mid
            equity = cash[product] + new_pos * mid
            prev_pos[product] = new_pos
            if equity < KILL_SWITCH_THRESHOLD and product not in killed:
                killed.add(product)
                print(f"[KILL_SWITCH] {product} equity={equity:.0f} "
                      f"ts={state.timestamp} pos={new_pos}")

        td["cash"] = cash
        td["prev_pos"] = prev_pos
        td["killed"] = sorted(killed)
        return killed

    # ── EOD flatten ───────────────────────────────────────────────────────

    def _flatten_all_positions(self, state: TradingState) -> Dict[str, List[Order]]:
        """Submit aggressive crossing orders for every non-zero position.

        Walks the book level-by-level so we'll partial-fill against thin levels
        instead of submitting one order at a price that might not have liquidity.
        Tolerates empty books (skips that product this tick; will retry next).
        """
        result: Dict[str, List[Order]] = {}
        for product, position in state.position.items():
            if position == 0:
                continue
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            orders: List[Order] = []
            if position > 0:
                # long → sell into bids
                need = position
                for bid in sorted(depth.buy_orders.keys(), reverse=True):
                    if need <= 0:
                        break
                    avail = depth.buy_orders[bid]
                    fill = min(need, avail)
                    if fill > 0:
                        orders.append(Order(product, bid, -fill))
                        need -= fill
            else:
                # short → buy from asks
                need = -position
                for ask in sorted(depth.sell_orders.keys()):
                    if need <= 0:
                        break
                    avail = -depth.sell_orders[ask]
                    fill = min(need, avail)
                    if fill > 0:
                        orders.append(Order(product, ask, fill))
                        need -= fill
            if orders:
                result[product] = orders
        return result

    # ── state plumbing ────────────────────────────────────────────────────

    def _load_state(self, td_str: str) -> dict:
        try:
            td = json.loads(td_str) if td_str else {}
        except (json.JSONDecodeError, TypeError):
            td = {}
        td.setdefault("mids", {})
        td.setdefault("last_timestamp", -1)
        return td

    def _push_mid(self, td: dict, product: str, mid: float, max_len: int) -> list:
        mids = td["mids"].setdefault(product, [])
        mids.append(round(mid, 2))
        if len(mids) > max_len:
            del mids[: len(mids) - max_len]
        return mids

    @staticmethod
    def _get_mid(depth: OrderDepth):
        bb = max(depth.buy_orders) if depth.buy_orders else None
        ba = min(depth.sell_orders) if depth.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        if bb is not None:
            return float(bb)
        if ba is not None:
            return float(ba)
        return None

    @staticmethod
    def _clamp_buy(qty: int, current_pos: int) -> int:
        return max(0, min(qty, POSITION_LIMIT - current_pos))

    @staticmethod
    def _clamp_sell(qty: int, current_pos: int) -> int:
        # qty is positive (size to sell); returned as positive; sign applied at Order build
        return max(0, min(qty, POSITION_LIMIT + current_pos))

    # ── trend follower with adaptive sizing ───────────────────────────────

    def _trend_strategy(
        self, product: str, depth: OrderDepth, position: int, td: dict
    ) -> List[Order]:
        mid = self._get_mid(depth)
        if mid is None:
            return []
        mids = self._push_mid(td, product, mid, TREND_MAX_HISTORY)

        direction = TREND_DIRECTIONS[product]
        target = direction * POSITION_LIMIT

        # Catastrophic-reversal stop: if trailing K-tick return is strongly opposite
        # to our direction, flatten. Threshold of 5% on a 2000-tick lookback is ~1σ
        # against random walk noise (per-tick σ ~12 bps × √2000 ≈ 540 bps ≈ 5.4%) so
        # only triggers on genuinely directional reversal, not chop.
        if len(mids) >= TREND_REVERSAL_LOOKBACK:
            ref_mid = mids[-TREND_REVERSAL_LOOKBACK]
            if ref_mid > 0:
                trailing_ret = (mid - ref_mid) / ref_mid
                if trailing_ret * direction < -TREND_REVERSAL_THRESHOLD:
                    target = 0

        delta = target - position
        if delta == 0:
            return []

        # Aggressively cross the book — trend captured > spread paid
        orders: List[Order] = []
        if delta > 0:
            running = position
            for ask in sorted(depth.sell_orders.keys()):
                if delta <= 0:
                    break
                avail = -depth.sell_orders[ask]
                fill = self._clamp_buy(min(delta, avail), running)
                if fill > 0:
                    orders.append(Order(product, ask, fill))
                    delta -= fill
                    running += fill
        else:  # delta < 0; need to sell
            running = position
            need = -delta
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if need <= 0:
                    break
                avail = depth.buy_orders[bid]
                fill = self._clamp_sell(min(need, avail), running)
                if fill > 0:
                    orders.append(Order(product, bid, -fill))
                    need -= fill
                    running -= fill
        return orders

    # ── AR(1) contrarian quoter ───────────────────────────────────────────

    def _ar1_strategy(
        self, product: str, depth: OrderDepth, position: int, td: dict
    ) -> List[Order]:
        mid = self._get_mid(depth)
        if mid is None or not depth.buy_orders or not depth.sell_orders:
            return []
        mids = self._push_mid(td, product, mid, AR1_MAX_HISTORY)
        if len(mids) < 2:
            return []

        last_return = mids[-1] - mids[-2]
        ac1 = AR1_AC1[product]
        # Expected next move (sign tells us direction; magnitude is small)
        expected = ac1 * last_return

        bb = max(depth.buy_orders.keys())
        ba = min(depth.sell_orders.keys())
        if ba - bb < 2:
            # No room to improve quotes inside the spread — pull
            return []

        # Skew quotes contrarian to last move
        if expected > 0:    # last_return < 0; expect bounce up → bid more aggressively
            bid_p, ask_p = bb + 1, ba
        elif expected < 0:  # last_return > 0; expect reversal down → ask more aggressively
            bid_p, ask_p = bb, ba - 1
        else:
            bid_p, ask_p = bb, ba

        if bid_p >= ask_p:  # safety: never cross
            bid_p, ask_p = bb, ba

        orders: List[Order] = []
        buy_size = self._clamp_buy(POSITION_LIMIT, position)
        if buy_size > 0:
            orders.append(Order(product, bid_p, buy_size))
        sell_size = self._clamp_sell(POSITION_LIMIT, position)
        if sell_size > 0:
            orders.append(Order(product, ask_p, -sell_size))
        return orders
