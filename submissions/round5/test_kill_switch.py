"""Synthetic stress test for v2 P5 per-product kill-switch.

The historical backtest doesn't trip the kill-switch (worst real drawdown is
−$8,600; threshold is −$10,000). This test manually injects a −$11,000 paper
loss into traderData and confirms:
  1. The kill-switch trips for the targeted product on the next run().
  2. The killed product no longer receives orders.
  3. Other products are unaffected.

Run with:  pytest submissions/round5/test_kill_switch.py -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datamodel import OrderDepth, Observation, TradingState
from submissions.round5 import trader_v2


def _make_state(traderData: str, position: dict, ts: int = 1000) -> TradingState:
    # Build a small order_depth so _get_mid returns ~10000 for the trend products.
    ods = {}
    for product in trader_v2.TREND_DIRECTIONS:
        od = OrderDepth()
        od.buy_orders = {9999: 5}
        od.sell_orders = {10001: -5}
        ods[product] = od
    return TradingState(
        traderData=traderData,
        timestamp=ts,
        listings={},
        order_depths=ods,
        own_trades={p: [] for p in ods},
        market_trades={p: [] for p in ods},
        position=dict(position),
        observations=Observation({}, {}),
    )


def test_kill_switch_trips_when_equity_below_threshold(capsys):
    """Inject cash counter = -11,000 and a flat position so MtM equity = -11,000."""
    target = "MICROCHIP_OVAL"
    seed_td = {
        "cash": {target: -11_000.0},
        "prev_pos": {target: 0},
        "killed": [],
        "last_timestamp": 0,
    }
    trader = trader_v2.Trader()
    state = _make_state(json.dumps(seed_td), position={target: 0}, ts=100)

    result, _, td_str = trader.run(state)
    td = json.loads(td_str)

    assert target in td["killed"], (
        f"Expected {target} in killed list, got {td['killed']}"
    )
    assert target not in result, (
        f"Killed product {target} still received orders: {result.get(target)}"
    )
    captured = capsys.readouterr().out
    assert "[KILL_SWITCH]" in captured and target in captured, (
        f"Expected kill-switch log line for {target}, got: {captured!r}"
    )


def test_kill_switch_does_not_trip_other_products(capsys):
    """One product underwater should not affect the others."""
    target = "PEBBLES_XS"
    seed_td = {
        "cash": {target: -11_000.0},
        "prev_pos": {target: 0},
        "killed": [],
        "last_timestamp": 0,
    }
    trader = trader_v2.Trader()
    state = _make_state(json.dumps(seed_td), position={target: 0}, ts=100)

    result, _, td_str = trader.run(state)
    td = json.loads(td_str)

    assert td["killed"] == [target], (
        f"Only {target} should be killed; got {td['killed']}"
    )
    # other trend products should still be eligible to trade
    other_trend = next(p for p in trader_v2.TREND_DIRECTIONS if p != target)
    # they may or may not generate orders depending on book state; just verify
    # they're NOT in the killed list — that's the contract.
    assert other_trend not in td["killed"]


def test_kill_switch_resets_on_new_day():
    """When timestamp wraps to 0 (new day), killed/cash should reset."""
    target = "PANEL_2X4"
    seed_td = {
        "cash": {target: -11_000.0},
        "prev_pos": {target: 0},
        "killed": [target],
        "last_timestamp": 999_900,
    }
    trader = trader_v2.Trader()
    state = _make_state(json.dumps(seed_td), position={target: 0}, ts=0)

    _, _, td_str = trader.run(state)
    td = json.loads(td_str)

    assert td["killed"] == [], (
        f"Kill list should reset on day boundary, still has: {td['killed']}"
    )
    assert td["cash"].get(target, 0.0) == 0.0, (
        f"Cash should reset on day boundary, still has: {td['cash']}"
    )


def test_kill_switch_threshold_just_above_does_not_trip():
    """Equity at −$9,999 (just above the −$10,000 threshold) should not trip."""
    target = "OXYGEN_SHAKE_GARLIC"
    seed_td = {
        "cash": {target: -9_999.0},
        "prev_pos": {target: 0},
        "killed": [],
        "last_timestamp": 0,
    }
    trader = trader_v2.Trader()
    state = _make_state(json.dumps(seed_td), position={target: 0}, ts=100)

    _, _, td_str = trader.run(state)
    td = json.loads(td_str)

    assert target not in td["killed"], (
        f"−$9,999 should be above the trip threshold; killed={td['killed']}"
    )
