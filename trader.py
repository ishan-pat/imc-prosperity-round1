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
        raise NotImplementedError

    def _trade_osmium(self, od: OrderDepth, position: int, timestamp: int, td: dict) -> List[Order]:
        raise NotImplementedError
