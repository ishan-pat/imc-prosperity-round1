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
                passive_price = min(best_bid + 1, int(fair_value) + 1)
            else:
                passive_price = int(fair_value) - 5  # fallback when book is empty
            orders.append(Order("INTARIAN_PEPPER_ROOT", passive_price, buy_cap))

        return orders

    def _trade_osmium(self, od: OrderDepth, position: int, timestamp: int, td: dict) -> List[Order]:
        FAIR = 10000
        LEVELS = 4
        orders = []

        mid = self._get_mid(od)
        if mid is None:
            return orders

        # AC signal: compare current mid to last stored mid
        last_mid = td.get("osmium_last_mid", float(FAIR))
        last_return = mid - last_mid
        td["osmium_last_mid"] = mid

        if last_return < -3:       # dip → expect bounce → tighten bids
            bid_compress, ask_compress = 1, 0
        elif last_return > 3:      # spike → expect reversal → tighten asks
            bid_compress, ask_compress = 0, 1
        else:
            bid_compress, ask_compress = 0, 0

        # End-of-day: flatten position aggressively in last 500 ticks
        if timestamp > 950000:
            tighten = (timestamp - 950000) // 10000  # increases 0→5 over last ~500k timestamps
            if position > 0:
                ask_compress += tighten   # ask moves closer to res, easier to sell
            elif position < 0:
                bid_compress += tighten   # bid moves closer to res, easier to buy

        # Strong signal: shift reservation price toward expected bounce
        if last_return < -5:
            res_shift = 1.0   # shift res up (expect price to bounce up, want more bids filled)
        elif last_return > 5:
            res_shift = -1.0  # shift res down (expect price to fall, want more asks filled)
        else:
            res_shift = 0.0

        # Inventory-skewed reservation price: lowers quotes when long, raises when short
        res = FAIR - position * (3.0 / 80.0) + res_shift

        buy_cap = LIMITS["ASH_COATED_OSMIUM"] - position
        sell_cap = LIMITS["ASH_COATED_OSMIUM"] + position

        # Post LEVELS bid levels below reservation price
        if buy_cap > 0:
            vol_per_level = max(1, buy_cap // LEVELS)
            remaining = buy_cap
            for i in range(1, LEVELS + 1):
                if remaining <= 0:
                    break
                vol = vol_per_level if i < LEVELS else remaining
                vol = min(vol, remaining)
                bid_price = int(res) - i + bid_compress
                orders.append(Order("ASH_COATED_OSMIUM", bid_price, vol))
                remaining -= vol

        # Post LEVELS ask levels above reservation price
        if sell_cap > 0:
            vol_per_level = max(1, sell_cap // LEVELS)
            remaining = sell_cap
            for i in range(1, LEVELS + 1):
                if remaining <= 0:
                    break
                vol = vol_per_level if i < LEVELS else remaining
                vol = min(vol, remaining)
                ask_price = int(res) + i - ask_compress
                # Safety: ensure no bid/ask crossing (ask must be > all bids)
                ask_price = max(ask_price, int(res) + 1)
                orders.append(Order("ASH_COATED_OSMIUM", ask_price, -vol))
                remaining -= vol

        return orders
