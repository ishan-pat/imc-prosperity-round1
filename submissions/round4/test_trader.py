"""Smoke + property tests for Round 4 trader.

Inherits the structure of Round 3's tests; adds counterparty-module tests
and updates active-voucher-strike assumptions to the Round 4 universe.
"""
import json
import math

import pytest
from datamodel import Order, OrderDepth, Trade, TradingState, Observation


# ── helpers ────────────────────────────────────────────────────────────────

def make_depth(bids: dict, asks: dict) -> OrderDepth:
    od = OrderDepth()
    od.buy_orders = dict(bids)
    od.sell_orders = dict(asks)
    return od


def make_state(order_depths, position, traderData="", timestamp=100,
               market_trades=None, own_trades=None):
    return TradingState(
        traderData=traderData,
        timestamp=timestamp,
        listings={},
        order_depths=order_depths,
        own_trades=own_trades if own_trades is not None else {k: [] for k in order_depths},
        market_trades=market_trades if market_trades is not None else {k: [] for k in order_depths},
        position=position,
        observations=Observation({}, {}),
    )


def full_state(positions=None, traderData="", timestamp=100,
               S=5250, hyd_mid=9990, market_trades=None):
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
    for K in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500):
        sym = f"VEV_{K}"
        intrinsic = max(0, S - K)
        mid = max(1, intrinsic + 5)
        depths[sym] = make_depth({mid - 1: 10}, {mid + 1: -10})
    return make_state(depths, positions, traderData, timestamp, market_trades=market_trades)


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
    S, K, T, sigma = 5250, 5400, 4, 0.013
    p = bs_call(S, K, T, sigma)
    sigma_hat = bs_iv(p, S, K, T)
    assert sigma_hat == pytest.approx(sigma, abs=1e-3)


def test_bs_delta_bounds():
    from trader import bs_delta
    assert 0.0 <= bs_delta(5250, 5400, 4, 0.013) <= 1.0
    assert bs_delta(6000, 5000, 4, 0.013) > 0.95
    assert bs_delta(5000, 6500, 4, 0.013) < 0.05


def test_svi_w_nonnegative_in_band():
    from trader import svi_w
    for k in (-0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5):
        assert svi_w(k) >= 0, f"SVI w({k}) negative — calibration broke no-arb"


def test_svi_sigma_in_range():
    from trader import svi_sigma, SIGMA_FLOOR, SIGMA_CEIL, STRIKES
    for K in STRIKES:
        s = svi_sigma(K, 5250, 4)
        assert SIGMA_FLOOR <= s <= SIGMA_CEIL


def test_avellaneda_stoikov_skews_with_inventory():
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


# ── COUNTERPARTY MODULE (Round-4 specific) ────────────────────────────────

def test_is_mark_classifier():
    from trader import is_mark
    assert is_mark("Mark 14") is True
    assert is_mark("Mark 99") is True       # unseen Mark still parses
    assert is_mark("SUBMISSION") is False
    assert is_mark(None) is False
    assert is_mark("") is False
    assert is_mark("mark 14") is False      # case sensitive


def test_get_mark_tier_known():
    from trader import get_mark_tier
    assert get_mark_tier("Mark 14") == "sharp"
    assert get_mark_tier("Mark 38") == "noise"
    assert get_mark_tier("Mark 55") == "noise"
    assert get_mark_tier("Mark 22") == "vol_seller"
    assert get_mark_tier("Mark 01") == "mild_buyer"


def test_get_mark_tier_unknown_defaults_neutral():
    from trader import get_mark_tier
    assert get_mark_tier("Mark 99") == "neutral"        # never seen historically
    assert get_mark_tier(None) == "neutral"
    assert get_mark_tier("OURSELVES") == "neutral"


def test_update_mark_stats_accumulates():
    from trader import Trader
    # State with one market trade between two Marks
    trade = Trade("HYDROGEL_PACK", price=9988, quantity=5,
                  buyer="Mark 38", seller="Mark 14", timestamp=100)
    state = full_state(market_trades={
        "HYDROGEL_PACK": [trade],
        "VELVETFRUIT_EXTRACT": [],
        **{f"VEV_{K}": [] for K in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
    })
    _, _, td_str = Trader().run(state)
    td = json.loads(td_str)
    assert "mark_stats" in td
    assert "Mark 38" in td["mark_stats"]
    assert "Mark 14" in td["mark_stats"]
    assert td["mark_stats"]["Mark 38"]["buy_n"] == 1
    assert td["mark_stats"]["Mark 14"]["sell_n"] == 1


def test_update_mark_stats_filters_non_marks():
    """If buyer or seller isn't a Mark (e.g., us), don't count it."""
    from trader import Trader
    trade = Trade("HYDROGEL_PACK", price=9988, quantity=5,
                  buyer="SUBMISSION", seller="Mark 14", timestamp=100)
    state = full_state(market_trades={
        "HYDROGEL_PACK": [trade],
        "VELVETFRUIT_EXTRACT": [],
        **{f"VEV_{K}": [] for K in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
    })
    _, _, td_str = Trader().run(state)
    td = json.loads(td_str)
    # SUBMISSION (us) must not appear in stats
    assert "SUBMISSION" not in td["mark_stats"]
    # Mark 14 still tracked on the seller side
    assert "Mark 14" in td["mark_stats"]
    assert td["mark_stats"]["Mark 14"]["sell_n"] == 1


# ── high-K voucher asymmetric edges ───────────────────────────────────────

def test_high_k_asymmetric_edge_easier_to_buy():
    """For K ≥ 5300 we accept Mark 22's flow more readily — buy edge < sell edge."""
    from trader import (HIGH_K_EDGE_BUY_MULT, HIGH_K_EDGE_SELL_MULT,
                        HIGH_K_VOUCHER_THRESHOLD)
    assert HIGH_K_EDGE_BUY_MULT < 1.0 < HIGH_K_EDGE_SELL_MULT
    assert HIGH_K_VOUCHER_THRESHOLD == 5300


def test_active_voucher_universe_is_round4():
    """Round-4 active universe: 4000 (delta-1) + high-K wings."""
    from trader import ACTIVE_VOUCHER_STRIKES
    assert set(ACTIVE_VOUCHER_STRIKES) == {4000, 5300, 5400, 5500, 6000, 6500}


def test_voucher_cold_start_gate_blocks_early_trades():
    """In the cold-start window (ts < VOUCHER_COLD_START_TICKS), voucher
    trading is suppressed even when there's a clear mispricing.  This is
    the surgical fix for the live -$1,400 early-day drawdown."""
    from trader import Trader, bs_call, VOUCHER_COLD_START_TICKS
    S, K, T, sigma = 5250, 5400, 4, 0.013
    fair = bs_call(S, K, T, sigma)
    cheap_ask = max(1, int(fair) - 30)
    depths = {
        "HYDROGEL_PACK": make_depth({9988: 30}, {9992: -30}),
        "VELVETFRUIT_EXTRACT": make_depth({S - 1: 40}, {S + 1: -40}),
        "VEV_5400": make_depth({cheap_ask - 2: 5}, {cheap_ask: -10}),
    }
    state = make_state(depths, {}, timestamp=VOUCHER_COLD_START_TICKS // 2)
    result, _, _ = Trader().run(state)
    voucher_orders = result.get("VEV_5400", [])
    assert voucher_orders == [], "Cold-start gate must block voucher trades despite mispricing"


def test_day_length_is_100k():
    """LIVE day length is 100K timestamps; verified from PnL chart X-axis."""
    from trader import DAY_LENGTH_TS, EOD_FLATTEN_VOUCHERS_START_TS
    assert DAY_LENGTH_TS == 100_000
    assert EOD_FLATTEN_VOUCHERS_START_TS == 95_000


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
        "VEV_5400": make_depth({90: 10}, {100: -10}),
    }
    state = make_state(depths, {})
    result, _, _ = Trader().run(state)
    assert result.get("VEV_5400", []) == []


# ── traderData persistence ────────────────────────────────────────────────

def test_trader_data_persists_between_calls():
    from trader import Trader
    t = Trader()
    state1 = full_state()
    _, _, td1 = t.run(state1)
    parsed = json.loads(td1)
    assert "sigma" in parsed
    assert "last_timestamp" in parsed
    assert "mark_stats" in parsed
    state2 = full_state(traderData=td1, timestamp=200)
    _, _, td2 = t.run(state2)
    parsed2 = json.loads(td2)
    assert parsed2["last_timestamp"] == 200


def test_new_day_reset_round4():
    """When timestamp resets, TTE start decrements by 1 day. Round 4 starts at 4."""
    from trader import Trader, LIVE_TTE_DAY_START
    assert LIVE_TTE_DAY_START == 4
    t = Trader()
    td = json.dumps({
        "last_timestamp": 999900,
        "tte_start": 4,
        "last_mid": {"HYDROGEL_PACK": 9990, "VELVETFRUIT_EXTRACT": 5250},
        "sigma": {str(k): 0.013 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
        "mark_stats": {},
    })
    state = full_state(traderData=td, timestamp=0)
    _, _, td_new = t.run(state)
    parsed = json.loads(td_new)
    assert parsed["tte_start"] == 3


# ── voucher vol-arb on Round-4 active strikes ─────────────────────────────

def test_voucher_no_trade_at_small_dislocation():
    """With VOUCHER_MIN_EDGE = 30, a 10-XIREC gap shouldn't trigger trades.
    This is the live-fix: the historical-σ-seeded model was producing many
    small "edges" that lost money in live."""
    from trader import Trader, bs_call
    S, K, T, sigma = 5250, 5400, 4, 0.013
    fair = bs_call(S, K, T, sigma)
    smallish_ask = max(1, int(fair) - 10)            # 10 XIRECs cheaper than fair
    depths = {
        "HYDROGEL_PACK": make_depth({9988: 30}, {9992: -30}),
        "VELVETFRUIT_EXTRACT": make_depth({S - 1: 40}, {S + 1: -40}),
        "VEV_5400": make_depth({smallish_ask - 2: 5}, {smallish_ask: -10}),
    }
    state = make_state(depths, {}, timestamp=2000)
    result, _, _ = Trader().run(state)
    voucher_orders = result.get("VEV_5400", [])
    assert voucher_orders == [], "Small mispricing should not trigger voucher trade"


def test_voucher_buy_on_extreme_dislocation():
    """50-XIREC gap (above the new MIN_EDGE=30) still triggers a trade.
    Uses K=4000 (deep ITM, fair ~$1,250) so we can dislocate by 50 without
    hitting the price floor."""
    from trader import Trader, bs_call
    S, K, T, sigma = 5250, 4000, 4, 0.0388
    fair = bs_call(S, K, T, sigma)
    super_cheap_ask = int(fair) - 50                 # 50 below fair: must trade
    depths = {
        "HYDROGEL_PACK": make_depth({9988: 30}, {9992: -30}),
        "VELVETFRUIT_EXTRACT": make_depth({S - 1: 40}, {S + 1: -40}),
        "VEV_4000": make_depth({super_cheap_ask - 2: 5}, {super_cheap_ask: -10}),
    }
    state = make_state(depths, {}, timestamp=2000)
    result, _, _ = Trader().run(state)
    buys = [o for o in result.get("VEV_4000", []) if o.quantity > 0]
    assert buys, "Extreme mispricing must still trigger trade"


def test_eod_voucher_no_flatten_when_disabled():
    """EOD_VOUCHER_FLATTEN_ENABLED=False ⇒ positions ride to MtM at mid
    instead of being liquidated into the touch (spread × size cost)."""
    from trader import Trader, EOD_FLATTEN_VOUCHERS_START_TS, EOD_VOUCHER_FLATTEN_ENABLED
    assert EOD_VOUCHER_FLATTEN_ENABLED is False
    state = full_state(positions={"VEV_5400": 50},
                       timestamp=EOD_FLATTEN_VOUCHERS_START_TS + 1000)
    result, _, _ = Trader().run(state)
    voucher_orders = result.get("VEV_5400", [])
    assert voucher_orders == [], "EOD voucher flatten must be disabled"


# ── delta hedging ──────────────────────────────────────────────────────────

def test_long_voucher_position_triggers_underlying_sell():
    from trader import Trader
    state = full_state(positions={"VEV_5400": 100}, timestamp=2000)
    result, _, _ = Trader().run(state)
    velv_orders = result.get("VELVETFRUIT_EXTRACT", [])
    sell_orders = [o for o in velv_orders if o.quantity < 0]
    assert sell_orders, "Should emit sell on VELVETFRUIT to hedge long voucher delta"


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
