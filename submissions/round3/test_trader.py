"""Smoke + property tests for round 3 trader."""
import json
import math

import pytest
from datamodel import Order, OrderDepth, TradingState, Observation


# ── helpers ────────────────────────────────────────────────────────────────

def make_depth(bids: dict, asks: dict) -> OrderDepth:
    od = OrderDepth()
    od.buy_orders = dict(bids)
    od.sell_orders = dict(asks)
    return od


def make_state(order_depths, position, traderData="", timestamp=100):
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


def full_state(positions=None, traderData="", timestamp=100,
               S=5250, hyd_mid=9990):
    """A state with all 12 products quoted around realistic mids."""
    positions = positions or {}
    depths = {
        "HYDROGEL_PACK": make_depth(
            {hyd_mid - 2: 30, hyd_mid - 5: 50},
            {hyd_mid + 2: -30, hyd_mid + 5: -50},
        ),
        "VELVETFRUIT_EXTRACT": make_depth(
            {S - 1: 40, S - 3: 60},
            {S + 1: -40, S + 3: -60},
        ),
    }
    # Rough realistic voucher mids using a flat 0.013 vol, T=5 days
    for K in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500):
        sym = f"VEV_{K}"
        intrinsic = max(0, S - K)
        # very crude mid; tests only care about presence of book, not realism
        mid = max(1, intrinsic + 5)
        depths[sym] = make_depth({mid - 1: 10}, {mid + 1: -10})
    return make_state(depths, positions, traderData, timestamp)


# ── core contract ──────────────────────────────────────────────────────────

def test_run_returns_three_values():
    from trader import Trader
    state = full_state()
    result = Trader().run(state)
    assert len(result) == 3


def test_run_returns_dict_orders_int_str():
    from trader import Trader
    orders, conv, td = Trader().run(full_state())
    assert isinstance(orders, dict)
    assert isinstance(conv, int)
    assert isinstance(td, str)
    json.loads(td)


def test_orders_have_correct_types():
    from trader import Trader
    orders_dict, _, _ = Trader().run(full_state())
    for sym, orders in orders_dict.items():
        for o in orders:
            assert isinstance(o, Order)
            assert isinstance(o.price, int)
            assert isinstance(o.quantity, int)
            assert o.symbol == sym


# ── pricing primitives ─────────────────────────────────────────────────────

def test_bs_call_intrinsic_when_T_zero():
    from trader import bs_call
    assert bs_call(5300, 5000, 0, 0.01) == pytest.approx(300)
    assert bs_call(5000, 5300, 0, 0.01) == pytest.approx(0)


def test_bs_iv_round_trip():
    from trader import bs_call, bs_iv
    S, K, T, sigma = 5250, 5200, 5, 0.013
    p = bs_call(S, K, T, sigma)
    sigma_hat = bs_iv(p, S, K, T)
    assert sigma_hat == pytest.approx(sigma, abs=1e-3)


def test_bs_delta_bounds():
    from trader import bs_delta
    assert 0.0 <= bs_delta(5250, 5200, 5, 0.013) <= 1.0
    # deep ITM → delta ≈ 1
    assert bs_delta(6000, 5000, 5, 0.013) > 0.95
    # deep OTM → delta ≈ 0
    assert bs_delta(5000, 6500, 5, 0.013) < 0.05


def test_svi_w_nonnegative_in_band():
    """SVI w(k) should be ≥ 0 for k in [-0.5, 0.5] under our calibrated params
    (Gatheral & Jacquier no-arb floor is enforced at fit time)."""
    from trader import svi_w
    for k in (-0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5):
        assert svi_w(k) >= 0, f"SVI w({k}) negative — calibration broke no-arb"


def test_svi_sigma_in_range():
    """SVI-implied σ for our 10 strikes at S=5250, T=5 should fall in [floor, ceil]."""
    from trader import svi_sigma, SIGMA_FLOOR, SIGMA_CEIL, STRIKES
    for K in STRIKES:
        s = svi_sigma(K, 5250, 5)
        assert SIGMA_FLOOR <= s <= SIGMA_CEIL


def test_avellaneda_stoikov_skews_with_inventory():
    """A-S reservation must shift down when long, up when short."""
    from trader import avellaneda_stoikov
    r_flat, _ = avellaneda_stoikov(5250.0, q=0, T_minus_t=1.0,
                                    gamma=2e-6, sigma=113.0, k=0.443)
    r_long, _ = avellaneda_stoikov(5250.0, q=100, T_minus_t=1.0,
                                    gamma=2e-6, sigma=113.0, k=0.443)
    r_short, _ = avellaneda_stoikov(5250.0, q=-100, T_minus_t=1.0,
                                     gamma=2e-6, sigma=113.0, k=0.443)
    assert r_long < r_flat < r_short


def test_avellaneda_stoikov_half_spread_positive():
    from trader import avellaneda_stoikov
    _, hs = avellaneda_stoikov(5250.0, q=0, T_minus_t=1.0,
                                gamma=2e-6, sigma=113.0, k=0.443)
    assert hs > 0


# ── position limit guard ───────────────────────────────────────────────────

def test_position_limits_respected_hydrogel():
    from trader import Trader, LIMITS
    state = full_state(positions={"HYDROGEL_PACK": 195})
    orders_dict, _, _ = Trader().run(state)
    buys = sum(o.quantity for o in orders_dict.get("HYDROGEL_PACK", []) if o.quantity > 0)
    assert buys <= LIMITS["HYDROGEL_PACK"] - 195


def test_position_limits_respected_velvetfruit():
    from trader import Trader, LIMITS
    state = full_state(positions={"VELVETFRUIT_EXTRACT": -195})
    orders_dict, _, _ = Trader().run(state)
    sells = -sum(o.quantity for o in orders_dict.get("VELVETFRUIT_EXTRACT", []) if o.quantity < 0)
    assert sells <= LIMITS["VELVETFRUIT_EXTRACT"] - 195


def test_no_orders_at_max_long_hydrogel():
    from trader import Trader
    state = full_state(positions={"HYDROGEL_PACK": 200})
    orders_dict, _, _ = Trader().run(state)
    buys = [o for o in orders_dict.get("HYDROGEL_PACK", []) if o.quantity > 0]
    assert buys == []


def test_voucher_position_limits():
    from trader import Trader
    state = full_state(positions={"VEV_5400": 295})
    orders_dict, _, _ = Trader().run(state)
    buys = sum(o.quantity for o in orders_dict.get("VEV_5400", []) if o.quantity > 0)
    assert buys <= 5


# ── empty-book robustness ─────────────────────────────────────────────────

def test_empty_books_do_not_crash():
    from trader import Trader
    depths = {
        "HYDROGEL_PACK": make_depth({}, {}),
        "VELVETFRUIT_EXTRACT": make_depth({}, {}),
    }
    state = make_state(depths, {})
    result, conv, td = Trader().run(state)
    assert isinstance(result, dict)


def test_missing_underlying_skips_voucher_trading():
    from trader import Trader
    depths = {
        "HYDROGEL_PACK": make_depth({9988: 30}, {9992: -30}),
        # no VELVETFRUIT_EXTRACT, just a voucher
        "VEV_5200": make_depth({90: 10}, {100: -10}),
    }
    state = make_state(depths, {})
    result, _, _ = Trader().run(state)
    # No underlying mid → no voucher trades
    assert result.get("VEV_5200", []) == []


# ── traderData persistence ────────────────────────────────────────────────

def test_trader_data_persists_between_calls():
    from trader import Trader
    t = Trader()
    state1 = full_state()
    _, _, td1 = t.run(state1)
    parsed = json.loads(td1)
    assert "sigma" in parsed
    assert "last_timestamp" in parsed
    state2 = full_state(traderData=td1, timestamp=200)
    _, _, td2 = t.run(state2)
    parsed2 = json.loads(td2)
    assert parsed2["last_timestamp"] == 200


def test_new_day_reset():
    """When timestamp resets, TTE start decrements by 1 day."""
    from trader import Trader
    t = Trader()
    td = json.dumps({
        "last_timestamp": 999900,
        "tte_start": 5,
        "last_mid": {"HYDROGEL_PACK": 9990, "VELVETFRUIT_EXTRACT": 5250},
        "sigma": {str(k): 0.013 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
        "underlying_mid_at_last_hedge": None,
    })
    state = full_state(traderData=td, timestamp=0)
    _, _, td_new = t.run(state)
    parsed = json.loads(td_new)
    assert parsed["tte_start"] == 4


# ── voucher vol-arb ────────────────────────────────────────────────────────

def test_voucher_buy_when_market_below_model():
    """Quote a voucher far below BS model → trader should buy."""
    from trader import Trader, bs_call
    S, K, T, sigma = 5250, 5200, 5, 0.013
    fair = bs_call(S, K, T, sigma)  # ~ 95 XIRECs
    # Set a wildly cheap ask of fair − 30 → must buy
    cheap_ask = max(1, int(fair) - 30)
    depths = {
        "HYDROGEL_PACK": make_depth({9988: 30}, {9992: -30}),
        "VELVETFRUIT_EXTRACT": make_depth({S - 1: 40}, {S + 1: -40}),
        "VEV_5200": make_depth({cheap_ask - 2: 5}, {cheap_ask: -10}),
    }
    state = make_state(depths, {}, timestamp=100)
    result, _, _ = Trader().run(state)
    buys = [o for o in result.get("VEV_5200", []) if o.quantity > 0]
    assert buys, "Trader should buy when ask is far below model fair"
    assert all(o.price == cheap_ask for o in buys)


def test_voucher_sell_when_market_above_model():
    from trader import Trader, bs_call
    S, K, T, sigma = 5250, 5200, 5, 0.013
    fair = bs_call(S, K, T, sigma)
    rich_bid = int(fair) + 30
    depths = {
        "HYDROGEL_PACK": make_depth({9988: 30}, {9992: -30}),
        "VELVETFRUIT_EXTRACT": make_depth({S - 1: 40}, {S + 1: -40}),
        "VEV_5200": make_depth({rich_bid: 10}, {rich_bid + 2: -5}),
    }
    state = make_state(depths, {"VEV_5200": 0}, timestamp=100)
    result, _, _ = Trader().run(state)
    sells = [o for o in result.get("VEV_5200", []) if o.quantity < 0]
    assert sells, "Trader should sell when bid is far above model fair"


def test_eod_voucher_flatten():
    """At EOD, long voucher position should hit the bid to liquidate."""
    from trader import Trader, EOD_FLATTEN_VOUCHERS_START_TS
    state = full_state(positions={"VEV_5200": 50}, timestamp=EOD_FLATTEN_VOUCHERS_START_TS + 1000)
    result, _, _ = Trader().run(state)
    sells = [o for o in result.get("VEV_5200", []) if o.quantity < 0]
    assert sells, "Should flatten long voucher inventory at EOD"


# ── delta hedging ──────────────────────────────────────────────────────────

def test_long_voucher_position_triggers_underlying_sell():
    """Long voucher = positive delta → trader should sell VELVETFRUIT to hedge."""
    from trader import Trader, DELTA_DEAD_ZONE
    # 50 vouchers × delta ≈ 0.5 → ~25 underlying delta, > dead zone (8)
    state = full_state(positions={"VEV_5200": 50}, timestamp=100)
    result, _, _ = Trader().run(state)
    velv_orders = result.get("VELVETFRUIT_EXTRACT", [])
    sell_orders = [o for o in velv_orders if o.quantity < 0]
    # With 50 long vouchers (delta ~25), the trader should at least *want* to sell
    # — even if MM also posts asks, the hedge order should appear at the touch.
    assert sell_orders, "Should emit at least one sell on VELVETFRUIT to hedge long voucher delta"


def test_no_self_cross_hydrogel():
    from trader import Trader
    for pos in (-180, -100, 0, 100, 180):
        state = full_state(positions={"HYDROGEL_PACK": pos})
        result, _, _ = Trader().run(state)
        orders = result.get("HYDROGEL_PACK", [])
        bids = [o.price for o in orders if o.quantity > 0]
        asks = [o.price for o in orders if o.quantity < 0]
        if bids and asks:
            assert max(bids) < min(asks), f"Crossed at pos={pos}: bid={max(bids)}, ask={min(asks)}"
