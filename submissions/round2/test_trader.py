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
    from final import Trader
    t = Trader()
    assert callable(t.run)

def test_run_returns_three_values():
    from final import Trader
    t = Trader()
    state = pepper_state(traderData=fresh_td())
    result = t.run(state)
    assert len(result) == 3, "run() must return (orders_dict, conversions, traderData)"

# ── PEPPER ROOT tests ───────────────────────────────────────────────────────

def test_pepper_initializes_base_on_first_tick():
    """On first tick with no prior state, base_price must be set from mid."""
    from final import Trader
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
    from final import Trader
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
    from final import Trader
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
    from final import Trader
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
    from final import Trader
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
    from final import Trader
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
    from final import Trader
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

# ── OSMIUM tests ────────────────────────────────────────────────────────────

def test_osmium_posts_multiple_bid_and_ask_levels():
    """At neutral position, posts at least 3 bid and 3 ask levels."""
    from final import Trader
    t = Trader()
    state = osmium_state(position=0, traderData=fresh_td())
    result, _, _ = t.run(state)
    orders = result.get("ASH_COATED_OSMIUM", [])
    buys = [o for o in orders if o.quantity > 0]
    sells = [o for o in orders if o.quantity < 0]
    assert len(buys) >= 3
    assert len(sells) >= 3

def test_osmium_total_buy_volume_within_limit():
    """Total buy volume never exceeds limit - position."""
    from final import Trader
    t = Trader()
    state = osmium_state(position=60, traderData=fresh_td())
    result, _, _ = t.run(state)
    orders = result.get("ASH_COATED_OSMIUM", [])
    total_buy = sum(o.quantity for o in orders if o.quantity > 0)
    assert total_buy <= 20  # 80 - 60

def test_osmium_total_sell_volume_within_limit():
    """Total sell volume never exceeds limit + position."""
    from final import Trader
    t = Trader()
    state = osmium_state(position=-60, traderData=fresh_td())
    result, _, _ = t.run(state)
    orders = result.get("ASH_COATED_OSMIUM", [])
    total_sell = sum(-o.quantity for o in orders if o.quantity < 0)
    assert total_sell <= 20  # 80 - 60

def test_osmium_no_buy_at_max_long():
    """Posts zero buy orders when position == 80."""
    from final import Trader
    t = Trader()
    state = osmium_state(position=80, traderData=fresh_td())
    result, _, _ = t.run(state)
    buys = [o for o in result.get("ASH_COATED_OSMIUM", []) if o.quantity > 0]
    assert buys == []

def test_osmium_no_sell_at_max_short():
    """Posts zero sell orders when position == -80."""
    from final import Trader
    t = Trader()
    state = osmium_state(position=-80, traderData=fresh_td())
    result, _, _ = t.run(state)
    sells = [o for o in result.get("ASH_COATED_OSMIUM", []) if o.quantity < 0]
    assert sells == []

def test_osmium_bids_below_asks():
    """No bid price is >= any ask price (no self-crossing orders)."""
    from final import Trader
    t = Trader()
    state = osmium_state(position=0, traderData=fresh_td())
    result, _, _ = t.run(state)
    orders = result.get("ASH_COATED_OSMIUM", [])
    bid_prices = [o.price for o in orders if o.quantity > 0]
    ask_prices = [o.price for o in orders if o.quantity < 0]
    if bid_prices and ask_prices:
        assert max(bid_prices) < min(ask_prices)

def test_osmium_inventory_skew_long():
    """When long (position > 0), reservation price shifts down — asks move lower."""
    from final import Trader
    t = Trader()
    state_neutral = osmium_state(position=0, traderData=fresh_td())
    state_long = osmium_state(position=60, traderData=fresh_td())
    t2 = Trader()
    result_n, _, _ = t.run(state_neutral)
    result_l, _, _ = t2.run(state_long)
    asks_n = sorted(o.price for o in result_n.get("ASH_COATED_OSMIUM", []) if o.quantity < 0)
    asks_l = sorted(o.price for o in result_l.get("ASH_COATED_OSMIUM", []) if o.quantity < 0)
    # When long, asks should be lower (to sell more aggressively)
    assert min(asks_l) <= min(asks_n)

def test_osmium_ac_signal_tightens_bid_after_dip():
    """After a price dip (last_return < -3), innermost bid is 1 tick closer to fair."""
    from final import Trader
    t_dip = Trader()
    t_flat = Trader()
    # Neutral last_mid = 10000, current mid = 9995 → last_return = -5 (dip)
    dip_td = json.dumps({
        "pepper_base": 12000.0, "pepper_initialized": True,
        "osmium_last_mid": 10000.0, "last_timestamp": 50,
    })
    flat_td = json.dumps({
        "pepper_base": 12000.0, "pepper_initialized": True,
        "osmium_last_mid": 9995.0,  # last_mid same as current → no return
        "last_timestamp": 50,
    })
    # current mid = 9995 → bids at 9988, asks at 10002
    state_dip = osmium_state(traderData=dip_td, bids={9988: 15}, asks={10002: -15})
    state_flat = osmium_state(traderData=flat_td, bids={9988: 15}, asks={10002: -15})

    result_dip, _, _ = t_dip.run(state_dip)
    result_flat, _, _ = t_flat.run(state_flat)

    bids_dip = sorted((o.price for o in result_dip.get("ASH_COATED_OSMIUM", []) if o.quantity > 0), reverse=True)
    bids_flat = sorted((o.price for o in result_flat.get("ASH_COATED_OSMIUM", []) if o.quantity > 0), reverse=True)

    # After dip, innermost (highest) bid should be >= flat innermost bid
    if bids_dip and bids_flat:
        assert bids_dip[0] >= bids_flat[0]

def test_osmium_stores_last_mid():
    """After each run, osmium_last_mid in traderData equals current mid."""
    from final import Trader
    t = Trader()
    state = osmium_state(
        traderData=fresh_td(),
        bids={9995: 15},
        asks={10005: -15},
    )
    _, _, new_td_str = t.run(state)
    td = json.loads(new_td_str)
    expected_mid = (9995 + 10005) / 2.0
    assert abs(td["osmium_last_mid"] - expected_mid) < 0.1

# ── edge case tests ─────────────────────────────────────────────────────────

def test_empty_order_book_pepper():
    """Trader handles empty order book without crashing."""
    from final import Trader
    t = Trader()
    state = pepper_state(
        traderData=fresh_td(),
        bids={},
        asks={},
    )
    result, conv, td_str = t.run(state)
    assert isinstance(result, dict)
    assert conv == 0
    assert isinstance(td_str, str)

def test_empty_order_book_osmium():
    """Trader handles empty OSMIUM order book without crashing."""
    from final import Trader
    t = Trader()
    state = osmium_state(
        traderData=fresh_td(),
        bids={},
        asks={},
    )
    result, conv, td_str = t.run(state)
    assert isinstance(result, dict)

def test_both_products_in_same_state():
    """run() handles both products in a single TradingState."""
    from final import Trader
    t = Trader()
    state = make_state(
        order_depths={
            "INTARIAN_PEPPER_ROOT": make_depth({11991: 19}, {12006: -10}),
            "ASH_COATED_OSMIUM": make_depth({9993: 15}, {10010: -15}),
        },
        position={"INTARIAN_PEPPER_ROOT": 0, "ASH_COATED_OSMIUM": 0},
        traderData=fresh_td(),
    )
    result, conv, td_str = t.run(state)
    assert "INTARIAN_PEPPER_ROOT" in result
    assert "ASH_COATED_OSMIUM" in result
    assert conv == 0

def test_trader_data_persists_between_calls():
    """traderData returned from tick N is accepted as input on tick N+1."""
    from final import Trader
    t = Trader()
    state1 = pepper_state(traderData="", timestamp=100)
    _, _, td1 = t.run(state1)
    # td1 must be valid JSON
    parsed = json.loads(td1)
    assert "pepper_base" in parsed
    assert "last_timestamp" in parsed
    # Second call uses returned traderData
    state2 = pepper_state(traderData=td1, timestamp=200)
    result2, _, td2 = t.run(state2)
    parsed2 = json.loads(td2)
    assert parsed2["last_timestamp"] == 200

def test_osmium_ask_never_below_bid():
    """In extreme inventory skew, ask prices never go below bid prices."""
    from final import Trader
    for pos in [-80, -60, -40, 0, 40, 60, 80]:
        t = Trader()
        state = osmium_state(position=pos, traderData=fresh_td())
        result, _, _ = t.run(state)
        orders = result.get("ASH_COATED_OSMIUM", [])
        bids = [o.price for o in orders if o.quantity > 0]
        asks = [o.price for o in orders if o.quantity < 0]
        if bids and asks:
            assert max(bids) < min(asks), f"Crossed market at position={pos}: bid={max(bids)}, ask={min(asks)}"

def test_osmium_eod_flattening_tightens_asks_when_long():
    """At ts > 950000 with long position, asks are lower than at ts=100."""
    from final import Trader
    t_eod = Trader()
    t_normal = Trader()
    state_eod = osmium_state(position=40, traderData=fresh_td(), timestamp=970000)
    state_normal = osmium_state(position=40, traderData=fresh_td(), timestamp=100)
    result_eod, _, _ = t_eod.run(state_eod)
    result_normal, _, _ = t_normal.run(state_normal)
    asks_eod = sorted(o.price for o in result_eod.get("ASH_COATED_OSMIUM", []) if o.quantity < 0)
    asks_normal = sorted(o.price for o in result_normal.get("ASH_COATED_OSMIUM", []) if o.quantity < 0)
    if asks_eod and asks_normal:
        assert min(asks_eod) <= min(asks_normal), "EOD asks should be tighter (lower) when long"

# ── Round 2: MAF auction bid ────────────────────────────────────────────────

def test_bid_returns_int():
    """bid() must return an integer (Prosperity rejects non-ints; negatives treated as 0)."""
    from final import Trader
    t = Trader()
    b = t.bid()
    assert isinstance(b, int), f"bid() must return int, got {type(b).__name__}"
    assert b >= 0, f"bid() should be non-negative (negatives treated as 0 anyway); got {b}"
