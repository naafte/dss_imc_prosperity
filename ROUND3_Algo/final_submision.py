from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    # --- HYDROGEL_PACK Constants ---
    HP_PRODUCT    = "HYDROGEL_PACK"
    HP_LIMIT      = 30
    HP_MM_SIZE    = 5
    HP_AGGR_EDGE  = 5     
    HP_AGGR_SIZE  = 15

    # --- VELVETFRUIT_EXTRACT Constants ---
    VFE_PRODUCT   = "VELVETFRUIT_EXTRACT"
    VFE_FV        = 5_250
    VFE_LIMIT     = 30
    VFE_MM_HALF   = 8      
    VFE_MM_SIZE   = 5
    VFE_AGGR_EDGE = 20     
    VFE_AGGR_SIZE = 5

    def _get_hp_orders(self, depth: OrderDepth, position: int):
        orders: List[Order] = []
        
        buy_prices = sorted(depth.buy_orders.keys(), reverse=True)
        sell_prices = sorted(depth.sell_orders.keys())
        
        best_bid = buy_prices[0] if buy_prices else None
        best_ask = sell_prices[0] if sell_prices else None
        
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = 10_000 

        rem_buy  = self.HP_LIMIT - position
        rem_sell = self.HP_LIMIT + position

        # Aggressive Taking
        if best_ask is not None and best_ask <= mid_price - self.HP_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.HP_AGGR_SIZE, rem_buy)
            orders.append(Order(self.HP_PRODUCT, best_ask, vol))
            rem_buy -= vol
            
        if best_bid is not None and best_bid >= mid_price + self.HP_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.HP_AGGR_SIZE, rem_sell)
            orders.append(Order(self.HP_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # Competitive Market Making (Pennying)
        if best_bid is not None:
            my_bid = best_bid + 1 - (1 if position > 10 else 0)
        else:
            my_bid = int(mid_price - 5)

        if best_ask is not None:
            my_ask = best_ask - 1 + (1 if position < -10 else 0)
        else:
            my_ask = int(mid_price + 5)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        if rem_buy > 0:
            orders.append(Order(self.HP_PRODUCT, int(my_bid), min(self.HP_MM_SIZE, rem_buy)))
        if rem_sell > 0:
            orders.append(Order(self.HP_PRODUCT, int(my_ask), -min(self.HP_MM_SIZE, rem_sell)))

        return orders

    def _get_vfe_orders(self, depth: OrderDepth, position: int):
        orders: List[Order] = []
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        rem_buy  = self.VFE_LIMIT - position
        rem_sell = self.VFE_LIMIT + position

        # Aggressive taking
        if best_ask is not None and best_ask < self.VFE_FV - self.VFE_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.VFE_AGGR_SIZE, rem_buy)
            orders.append(Order(self.VFE_PRODUCT, best_ask, vol))
            rem_buy -= vol
        if best_bid is not None and best_bid > self.VFE_FV + self.VFE_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.VFE_AGGR_SIZE, rem_sell)
            orders.append(Order(self.VFE_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # Passive limit orders with Skew
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

        # Handle Hydrogel Pack
        if self.HP_PRODUCT in state.order_depths:
            hp_depth = state.order_depths[self.HP_PRODUCT]
            hp_position = state.position.get(self.HP_PRODUCT, 0)
            result[self.HP_PRODUCT] = self._get_hp_orders(hp_depth, hp_position)

        # Handle Velvetfruit Extract
        if self.VFE_PRODUCT in state.order_depths:
            vfe_depth = state.order_depths[self.VFE_PRODUCT]
            vfe_position = state.position.get(self.VFE_PRODUCT, 0)
            result[self.VFE_PRODUCT] = self._get_vfe_orders(vfe_depth, vfe_position)
        
        return result, 0, "COMBINED_STRATEGY"