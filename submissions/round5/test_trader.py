"""
Tests for Round 5 trader_v1. Covers:
  - import and run() interface
  - bid() returns int
  - position-limit enforcement (hard cap 10)
  - trend strategy: builds in correct direction with full confirmation
  - trend strategy: confirmation scales position toward 0 / opposite when reversed
  - AR(1) strategy: skews quote inside the spread
  - graceful handling of empty book
  - traderData persists and round-trips through json
"""
import json

from datamodel import Order, OrderDepth, TradingState, Observation
from trader_v1 import (
    AR1_AC1,
    POSITION_LIMIT,
    TREND_DIRECTIONS,
    TREND_REVERSAL_LOOKBACK,
    TREND_REVERSAL_THRESHOLD,
    Trader,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_depth(bids: dict, asks: dict) -> OrderDepth:
    """bids = {price: +volume}, asks = {price: -volume}."""
    od = OrderDepth()
    od.buy_orders = dict(bids)
    od.sell_orders = dict(asks)
    return od


def state_for(product: str, depth: OrderDepth, position: int = 0,
              trader_data: str = "", timestamp: int = 100) -> TradingState:
    return TradingState(
        traderData=trader_data,
        timestamp=timestamp,
        listings={},
        order_depths={product: depth},
        own_trades={product: []},
        market_trades={product: []},
        position={product: position},
        observations=Observation({}, {}),
    )


def fresh_td_with_history(product: str, mids: list) -> str:
    return json.dumps({
        "last_timestamp": 0,
        "mids": {product: mids},
    })


# ── basic interface ───────────────────────────────────────────────────────────

def test_import():
    t = Trader()
    assert callable(t.run)
    assert callable(t.bid)


def test_bid_returns_int():
    assert isinstance(Trader().bid(), int)


def test_run_returns_three_values():
    t = Trader()
    depth = make_depth({10000: 5}, {10010: -5})
    result = t.run(state_for("PANEL_2X4", depth))
    assert len(result) == 3
    orders, conv, td = result
    assert isinstance(orders, dict)
    assert isinstance(conv, int)
    assert isinstance(td, str)


def test_handles_empty_book():
    t = Trader()
    depth = make_depth({}, {})
    result = t.run(state_for("PANEL_2X4", depth))
    # Should not crash; orders dict may or may not contain the product, but no exceptions
    assert isinstance(result[0], dict)


def test_trader_data_round_trips_json():
    t = Trader()
    depth = make_depth({10700: 5}, {10710: -5})
    state1 = state_for("PANEL_2X4", depth)
    _, _, td1 = t.run(state1)
    parsed = json.loads(td1)
    assert "mids" in parsed
    assert "PANEL_2X4" in parsed["mids"]
    # Second call accepts the returned td
    state2 = state_for("PANEL_2X4", depth, trader_data=td1, timestamp=200)
    _, _, td2 = t.run(state2)
    p2 = json.loads(td2)
    assert p2["last_timestamp"] == 200
    # mid history grew
    assert len(p2["mids"]["PANEL_2X4"]) == len(parsed["mids"]["PANEL_2X4"]) + 1


# ── trend strategy ────────────────────────────────────────────────────────────

def test_trender_with_short_history_uses_prior():
    """Before TREND_LOOKBACK ticks of history, use direction prior at full size."""
    t = Trader()
    # PANEL_2X4 expects LONG (+1)
    depth = make_depth({10700: 20}, {10710: -20})
    result, _, _ = t.run(state_for("PANEL_2X4", depth))
    orders = result.get("PANEL_2X4", [])
    assert len(orders) > 0
    # All orders should be BUYs (positive qty)
    assert all(o.quantity > 0 for o in orders)
    # Total bought = position limit
    total = sum(o.quantity for o in orders)
    assert total == POSITION_LIMIT


def test_trender_short_direction_uses_short_position():
    """PEBBLES_XS expects SHORT (-1) — should sell into bids."""
    t = Trader()
    depth = make_depth({6000: 15}, {6010: -15})
    result, _, _ = t.run(state_for("PEBBLES_XS", depth))
    orders = result.get("PEBBLES_XS", [])
    assert len(orders) > 0
    assert all(o.quantity < 0 for o in orders)
    total_sold = -sum(o.quantity for o in orders)
    assert total_sold == POSITION_LIMIT


def test_trender_respects_position_limit():
    """If already at limit, no further orders in same direction."""
    t = Trader()
    depth = make_depth({10700: 20}, {10710: -20})
    # Already long 10 on PANEL_2X4; should not buy more
    result, _, _ = t.run(state_for("PANEL_2X4", depth, position=POSITION_LIMIT))
    orders = result.get("PANEL_2X4", [])
    buys = [o for o in orders if o.quantity > 0]
    assert sum(o.quantity for o in buys) == 0


def test_trender_flattens_when_trend_reverses():
    """If trailing TREND_REVERSAL_LOOKBACK return is more than TREND_REVERSAL_THRESHOLD
    against the hardcoded direction, target flips to 0 and we flatten."""
    t = Trader()
    # PANEL_2X4 expects LONG. Build 2000+ mids descending from 11000 to 10000
    # → trailing 2000-tick return ~ -9% (strongly opposite, exceeds 5% threshold)
    # Need trailing 2000-tick return < -10% to trigger flatten — descend 12000 → 10000 (-16.7%)
    history = [12000.0 - i * (2000.0 / 2100) for i in range(2100)]
    depth = make_depth({10000: 20}, {10010: -20})
    state = state_for("PANEL_2X4", depth, position=POSITION_LIMIT,
                      trader_data=fresh_td_with_history("PANEL_2X4", history))
    result, _, _ = t.run(state)
    orders = result.get("PANEL_2X4", [])
    # Reversal detected → target = 0; we're at +10, so we should SELL 10
    total_signed = sum(o.quantity for o in orders)
    assert total_signed == -POSITION_LIMIT, f"Expected to flatten by selling 10, got {orders}"


# ── AR(1) strategy ────────────────────────────────────────────────────────────

def test_ar1_after_up_move_skews_ask_down():
    """ROBOT_IRONING has AC1<0; after last_return>0, expect mean reversion down → tighten ask.
    The trader computes last_return AFTER pushing the current mid onto history, so we set
    history so that mid_now - history[-1] is positive."""
    t = Trader()
    # depth mid = (9995 + 10005) / 2 = 10000; history[-1] = 9990 → last_return = +10
    history = [9990.0]
    depth = make_depth({9995: 10}, {10005: -10})
    state = state_for("ROBOT_IRONING", depth,
                      trader_data=fresh_td_with_history("ROBOT_IRONING", history))
    result, _, _ = t.run(state)
    orders = result.get("ROBOT_IRONING", [])
    asks = [o for o in orders if o.quantity < 0]
    assert any(o.price == 10004 for o in asks), f"Expected an ask at 10004 (best_ask - 1), got: {orders}"


def test_ar1_after_down_move_skews_bid_up():
    """After last_return<0, expect mean-reversion up → tighten bid (bid_p = best_bid + 1)."""
    t = Trader()
    # depth mid = 10000; history[-1] = 10010 → last_return = -10
    history = [10010.0]
    depth = make_depth({9995: 10}, {10005: -10})
    state = state_for("OXYGEN_SHAKE_EVENING_BREATH", depth,
                      trader_data=fresh_td_with_history("OXYGEN_SHAKE_EVENING_BREATH", history))
    result, _, _ = t.run(state)
    orders = result.get("OXYGEN_SHAKE_EVENING_BREATH", [])
    bids = [o for o in orders if o.quantity > 0]
    assert any(o.price == 9996 for o in bids), f"Expected a bid at 9996 (best_bid + 1), got: {orders}"


def test_ar1_respects_position_limit():
    t = Trader()
    history = [10000.0, 10005.0]
    depth = make_depth({9998: 10}, {10010: -10})
    # Already long 8; buy_size capped at 2
    state = state_for("ROBOT_IRONING", depth, position=8,
                      trader_data=fresh_td_with_history("ROBOT_IRONING", history))
    result, _, _ = t.run(state)
    orders = result.get("ROBOT_IRONING", [])
    buys = sum(o.quantity for o in orders if o.quantity > 0)
    sells = -sum(o.quantity for o in orders if o.quantity < 0)
    assert buys <= POSITION_LIMIT - 8
    assert sells <= POSITION_LIMIT + 8


# ── product universe ──────────────────────────────────────────────────────────

def test_skipped_products_get_no_orders():
    """Products outside our cherry-picked list should generate zero orders."""
    t = Trader()
    depth = make_depth({10000: 10}, {10010: -10})
    state = state_for("SLEEP_POD_COTTON", depth)  # not in TREND_DIRECTIONS or AR1_AC1
    result, _, _ = t.run(state)
    assert result.get("SLEEP_POD_COTTON", []) == [] or "SLEEP_POD_COTTON" not in result


def test_strategy_classes_disjoint():
    """A product must not appear in both TREND_DIRECTIONS and AR1_AC1."""
    overlap = set(TREND_DIRECTIONS).intersection(AR1_AC1)
    assert overlap == set(), f"product(s) in both strategy buckets: {overlap}"
