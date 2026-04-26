from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    HP_PRODUCT    = "HYDROGEL_PACK"
    HP_FV         = 10_000
    HP_LIMIT      = 30
    HP_MM_HALF    = 20     
    HP_MM_SIZE    = 5
    HP_AGGR_EDGE  = 40     
    HP_AGGR_SIZE  = 10

    def _orders(self, depth, position):
        orders: List[Order] = []
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        rem_buy  = self.HP_LIMIT - position
        rem_sell = self.HP_LIMIT + position

        # 1. Aggressive taking
        if best_ask is not None and best_ask < self.HP_FV - self.HP_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.HP_AGGR_SIZE, rem_buy)
            orders.append(Order(self.HP_PRODUCT, best_ask, vol))
            rem_buy -= vol
        if best_bid is not None and best_bid > self.HP_FV + self.HP_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.HP_AGGR_SIZE, rem_sell)
            orders.append(Order(self.HP_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # 2. Passive limit orders with Skew
        skew   = int(position / self.HP_LIMIT * self.HP_MM_HALF * 2)
        my_bid = self.HP_FV - self.HP_MM_HALF - skew
        my_ask = self.HP_FV + self.HP_MM_HALF - skew

        if position > self.HP_LIMIT // 3 and best_ask is not None:
            my_ask = min(my_ask, best_ask)
        if position < -(self.HP_LIMIT // 3) and best_bid is not None:
            my_bid = max(my_bid, best_bid)

        if rem_buy > 0 and (best_ask is None or my_bid < best_ask):
            orders.append(Order(self.HP_PRODUCT, my_bid, min(self.HP_MM_SIZE, rem_buy)))
        if rem_sell > 0 and (best_bid is None or my_ask > best_bid):
            orders.append(Order(self.HP_PRODUCT, my_ask, -min(self.HP_MM_SIZE, rem_sell)))

        return orders

    def run(self, state: TradingState):
        result = {}
        if self.HP_PRODUCT in state.order_depths:
            depth = state.order_depths[self.HP_PRODUCT]
            position = state.position.get(self.HP_PRODUCT, 0)
            result[self.HP_PRODUCT] = self._orders(depth, position)
        return result, 0, "HP_ONLY"