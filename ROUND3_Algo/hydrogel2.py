from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    HP_PRODUCT    = "HYDROGEL_PACK"
    HP_LIMIT      = 30
    
    # INCREASED SIZE: From 5 to 10 to capture more volume per fill
    HP_MM_SIZE    = 10
    
    # AGGRESSIVE TAKING: Lowered edge from 5 to 2 
    # This will "snatch" orders more frequently
    HP_AGGR_EDGE  = 2     
    HP_AGGR_SIZE  = 15

    def _orders(self, depth: OrderDepth, position: int):
        orders: List[Order] = []
        
        buy_prices = sorted(depth.buy_orders.keys(), reverse=True)
        sell_prices = sorted(depth.sell_orders.keys())
        
        if not buy_prices or not sell_prices:
            return orders

        best_bid = buy_prices[0]
        best_ask = sell_prices[0]
        mid_price = (best_bid + best_ask) / 2

        rem_buy  = self.HP_LIMIT - position
        rem_sell = self.HP_LIMIT + position

        # 1. More Aggressive Taker Logic
        # We now take liquidity if it's even slightly favorable
        if best_ask <= mid_price - self.HP_AGGR_EDGE and rem_buy > 0:
            vol = min(abs(depth.sell_orders[best_ask]), self.HP_AGGR_SIZE, rem_buy)
            orders.append(Order(self.HP_PRODUCT, best_ask, vol))
            rem_buy -= vol
            
        if best_bid >= mid_price + self.HP_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.HP_AGGR_SIZE, rem_sell)
            orders.append(Order(self.HP_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # 2. Tight Pennying (Staying at the front of the line)
        # We reduce the 'skew' intensity to stay closer to the best price
        # until we are at least 50% full.
        skew = int((position / self.HP_LIMIT) * 3) 
        
        my_bid = best_bid + 1
        my_ask = best_ask - 1

        # Only back off our price if we are getting dangerously close to the limit
        if position > 15: # We are long, lower the bid to stop buying
            my_bid -= 1
        if position < -15: # We are short, raise the ask to stop selling
            my_ask += 1

        # 3. Placement
        if rem_buy > 0 and my_bid < best_ask:
            orders.append(Order(self.HP_PRODUCT, int(my_bid), min(self.HP_MM_SIZE, rem_buy)))
        if rem_sell > 0 and my_ask > best_bid:
            orders.append(Order(self.HP_PRODUCT, int(my_ask), -min(self.HP_MM_SIZE, rem_sell)))

        return orders

    def run(self, state: TradingState):
        result = {}
        if self.HP_PRODUCT in state.order_depths:
            depth = state.order_depths[self.HP_PRODUCT]
            position = state.position.get(self.HP_PRODUCT, 0)
            result[self.HP_PRODUCT] = self._orders(depth, position)
        
        return result, 0, "HP_AGGRESSIVE"