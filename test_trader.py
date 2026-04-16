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
