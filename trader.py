from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
}

class Trader:

    def bid(self) -> int:
        """Required stub for Round 2 compatibility — ignored in Round 1."""
        return 15

    def run(self, state: TradingState):
        td = json.loads(state.traderData) if state.traderData else {}

        # Detect new day: timestamp resets to 0 at day boundaries
        last_ts = td.get("last_timestamp", -1)
        if last_ts > 0 and state.timestamp < last_ts:
            td.pop("pepper_initialized", None)
            td.pop("osmium_last_mid", None)
        td["last_timestamp"] = state.timestamp

        result = {}
        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == "INTARIAN_PEPPER_ROOT":
                orders = self._trade_pepper(od, pos, state.timestamp, td)
            elif product == "ASH_COATED_OSMIUM":
                orders = self._trade_osmium(od, pos, state.timestamp, td)
            else:
                orders = []
            result[product] = orders

        return result, 0, json.dumps(td)

    def _get_mid(self, od: OrderDepth):
        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return float(best_bid) if best_bid is not None else (float(best_ask) if best_ask is not None else None)

    def _trade_pepper(self, od: OrderDepth, position: int, timestamp: int, td: dict) -> List[Order]:
        orders = []
        mid = self._get_mid(od)
        if mid is None:
            return orders

        # Initialize base price on first tick of each day
        if not td.get("pepper_initialized"):
            td["pepper_base"] = mid - timestamp / 1000.0
            td["pepper_initialized"] = True

        fair_value = td["pepper_base"] + timestamp / 1000.0
        buy_cap = LIMITS["INTARIAN_PEPPER_ROOT"] - position

        if buy_cap <= 0:
            return orders

        # Sweep all asks at or below fair_value + 10
        for ask_price in sorted(od.sell_orders.keys()):
            if buy_cap <= 0:
                break
            if ask_price <= fair_value + 10:
                vol = min(buy_cap, -od.sell_orders[ask_price])
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, vol))
                buy_cap -= vol

        # Post passive bid at best_bid + 1 for remaining capacity
        if buy_cap > 0:
            if od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                passive_price = best_bid + 1
            else:
                passive_price = int(fair_value) - 5  # fallback when book is empty
            orders.append(Order("INTARIAN_PEPPER_ROOT", passive_price, buy_cap))

        return orders

    def _trade_osmium(self, od: OrderDepth, position: int, timestamp: int, td: dict) -> List[Order]:
        raise NotImplementedError
