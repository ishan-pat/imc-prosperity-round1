import json
import pytest
from datamodel import Order, OrderDepth, TradingState, Observation

# ── helpers ────────────────────────────────────────────────────────────────

def make_depth(bids: dict, asks: dict) -> OrderDepth:
    """bids = {price: volume}  asks = {price: -volume} (negative, per IMC spec)"""
    od = OrderDepth()
    od.buy_orders = dict(bids)
    od.sell_orders = dict(asks)
    return od

def make_state(order_depths: dict, position: dict,
               traderData: str = "", timestamp: int = 100) -> TradingState:
    return TradingState(
        traderData=traderData,
        timestamp=timestamp,
        listings={},
        order_depths=order_depths,
        own_trades={k: [] for k in order_depths},
        market_trades={k: [] for k in order_depths},
        position=position,
        observations=Observation({}, {}),
    )

def pepper_state(position: int = 0, traderData: str = "",
                 timestamp: int = 100,
                 bids: dict = None, asks: dict = None) -> TradingState:
    bids = bids or {11991: 19, 11990: 23}
    asks = asks or {12006: -10, 12009: -19}
    return make_state(
        {"INTARIAN_PEPPER_ROOT": make_depth(bids, asks)},
        {"INTARIAN_PEPPER_ROOT": position},
        traderData=traderData,
        timestamp=timestamp,
    )

def osmium_state(position: int = 0, traderData: str = "",
                 timestamp: int = 100,
                 bids: dict = None, asks: dict = None) -> TradingState:
    bids = bids or {9993: 15, 9992: 20}
    asks = asks or {10010: -15, 10011: -20}
    return make_state(
        {"ASH_COATED_OSMIUM": make_depth(bids, asks)},
        {"ASH_COATED_OSMIUM": position},
        traderData=traderData,
        timestamp=timestamp,
    )

def fresh_td(pepper_base: float = 12000.0, osmium_last_mid: float = 10000.0,
             last_timestamp: int = 50) -> str:
    return json.dumps({
        "pepper_base": pepper_base,
        "pepper_initialized": True,
        "osmium_last_mid": osmium_last_mid,
        "last_timestamp": last_timestamp,
    })

# ── placeholder tests (will fail until Trader is implemented) ───────────────

def test_import():
    from trader import Trader
    t = Trader()
    assert callable(t.run)

def test_run_returns_three_values():
    from trader import Trader
    t = Trader()
    state = pepper_state(traderData=fresh_td())
    result = t.run(state)
    assert len(result) == 3, "run() must return (orders_dict, conversions, traderData)"

# ── PEPPER ROOT tests ───────────────────────────────────────────────────────

def test_pepper_initializes_base_on_first_tick():
    """On first tick with no prior state, base_price must be set from mid."""
    from trader import Trader
    t = Trader()
    # mid = (11991 + 12006) / 2 = 11998.5, ts=0 → base = 11998.5 - 0 = 11998.5
    state = pepper_state(traderData="", timestamp=0,
                         bids={11991: 19}, asks={12006: -10})
    _, _, new_td_str = t.run(state)
    td = json.loads(new_td_str)
    assert td["pepper_initialized"] is True
    assert abs(td["pepper_base"] - 11998.5) < 0.1

def test_pepper_fair_value_uses_timestamp():
    """fair_value = base + ts/1000, so at ts=500000 with base=12000 → FV=12500."""
    from trader import Trader
    t = Trader()
    # base=12000, ts=500000 → FV=12500; ask at 12506 (FV+6 ≤ FV+10) should be taken
    state = pepper_state(
        traderData=fresh_td(pepper_base=12000.0),
        timestamp=500000,
        bids={12494: 19},
        asks={12506: -10},
    )
    result, _, _ = t.run(state)
    orders = result.get("INTARIAN_PEPPER_ROOT", [])
    buys = [o for o in orders if o.quantity > 0]
    assert len(buys) >= 1
    assert buys[0].price == 12506

def test_pepper_respects_position_limit():
    """Never sends buy orders that would take position above 80."""
    from trader import Trader
    t = Trader()
    # Already at 75 long; asks have 20 units available — should only buy 5
    state = pepper_state(
        position=75,
        traderData=fresh_td(pepper_base=12000.0),
        timestamp=100,
        asks={12006: -20},
    )
    result, _, _ = t.run(state)
    buys = [o for o in result.get("INTARIAN_PEPPER_ROOT", []) if o.quantity > 0]
    total_buy = sum(o.quantity for o in buys)
    assert total_buy <= 5  # 80 - 75 = 5

def test_pepper_does_not_buy_when_at_limit():
    """No buy orders when position == 80."""
    from trader import Trader
    t = Trader()
    state = pepper_state(
        position=80,
        traderData=fresh_td(pepper_base=12000.0),
        timestamp=100,
    )
    result, _, _ = t.run(state)
    buys = [o for o in result.get("INTARIAN_PEPPER_ROOT", []) if o.quantity > 0]
    assert buys == []

def test_pepper_skips_expensive_asks():
    """Does not take asks more than 10 above fair value."""
    from trader import Trader
    t = Trader()
    # FV = 12000 + 0.1 = 12000.1; ask at 12015 (> FV+10=12010.1) should be skipped
    state = pepper_state(
        position=0,
        traderData=fresh_td(pepper_base=12000.0),
        timestamp=100,
        bids={11991: 19},
        asks={12015: -10},
    )
    result, _, _ = t.run(state)
    buys = [o for o in result.get("INTARIAN_PEPPER_ROOT", []) if o.quantity > 0]
    prices = [o.price for o in buys]
    assert 12015 not in prices

def test_pepper_posts_passive_bid_when_book_not_full():
    """After sweeping asks, post passive bid at best_bid+1 for remaining capacity."""
    from trader import Trader
    t = Trader()
    # FV=12000.1, ask at 12015 skipped (> FV+10), position=0 → full 80 cap remaining
    state = pepper_state(
        position=0,
        traderData=fresh_td(pepper_base=12000.0),
        timestamp=100,
        bids={11991: 19},
        asks={12015: -10},  # too expensive, won't be taken
    )
    result, _, _ = t.run(state)
    buys = [o for o in result.get("INTARIAN_PEPPER_ROOT", []) if o.quantity > 0]
    # Should have passive bid at 11991 + 1 = 11992
    assert any(o.price == 11992 for o in buys)

def test_pepper_detects_new_day():
    """When timestamp resets (new day), re-initialize base_price."""
    from trader import Trader
    t = Trader()
    # Simulate: last tick was ts=999900 (end of day 1), now ts=0 (start of day 2)
    # Old base was 12000; new mid is 13000 at ts=0 → new base should be ~13000
    state = pepper_state(
        traderData=json.dumps({
            "pepper_base": 12000.0,
            "pepper_initialized": True,
            "osmium_last_mid": 10000.0,
            "last_timestamp": 999900,
        }),
        timestamp=0,
        bids={12993: 19},
        asks={13006: -10},  # mid ≈ 12999.5 → new base ≈ 12999.5
    )
    _, _, new_td_str = t.run(state)
    td = json.loads(new_td_str)
    # base should have reset to ~12999.5 (not remain at 12000)
    assert abs(td["pepper_base"] - 12999.5) < 1.0
