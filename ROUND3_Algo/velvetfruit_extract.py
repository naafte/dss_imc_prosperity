from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    VFE_PRODUCT   = "VELVETFRUIT_EXTRACT"
    VFE_FV        = 5_250
    VFE_LIMIT     = 30
    VFE_MM_HALF   = 8      
    VFE_MM_SIZE   = 5
    VFE_AGGR_EDGE = 20     
    VFE_AGGR_SIZE = 5

    def _orders(self, depth, position):
        orders: List[Order] = []
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        rem_buy  = self.VFE_LIMIT - position
        rem_sell = self.VFE_LIMIT + position

        # 1. Aggressive taking
        if best_ask is not None and best_ask < self.VFE_FV - self.VFE_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.VFE_AGGR_SIZE, rem_buy)
            orders.append(Order(self.VFE_PRODUCT, best_ask, vol))
            rem_buy -= vol
        if best_bid is not None and best_bid > self.VFE_FV + self.VFE_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.VFE_AGGR_SIZE, rem_sell)
            orders.append(Order(self.VFE_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # 2. Passive limit orders with Skew
        skew   = int(position / self.VFE_LIMIT * self.VFE_MM_HALF * 2)
        my_bid = self.VFE_FV - self.VFE_MM_HALF - skew
        my_ask = self.VFE_FV + self.VFE_MM_HALF - skew

        if position > self.VFE_LIMIT // 3 and best_ask is not None:
            my_ask = min(my_ask, best_ask)
        if position < -(self.VFE_LIMIT // 3) and best_bid is not None:
            my_bid = max(my_bid, best_bid)

        if rem_buy > 0 and (best_ask is None or my_bid < best_ask):
            orders.append(Order(self.VFE_PRODUCT, my_bid, min(self.VFE_MM_SIZE, rem_buy)))
        if rem_sell > 0 and (best_bid is None or my_ask > best_bid):
            orders.append(Order(self.VFE_PRODUCT, my_ask, -min(self.VFE_MM_SIZE, rem_sell)))

        return orders

    def run(self, state: TradingState):
        result = {}
        if self.VFE_PRODUCT in state.order_depths:
            depth = state.order_depths[self.VFE_PRODUCT]
            position = state.position.get(self.VFE_PRODUCT, 0)
            result[self.VFE_PRODUCT] = self._orders(depth, position)
        return result, 0, "VFE_ONLY"