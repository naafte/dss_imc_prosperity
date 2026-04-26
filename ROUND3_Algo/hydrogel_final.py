from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    HP_PRODUCT    = "HYDROGEL_PACK"
    HP_LIMIT      = 30
    HP_MM_SIZE    = 5
    
    # Reduced edge to be more aggressive in capturing opportunities
    HP_AGGR_EDGE  = 5     
    HP_AGGR_SIZE  = 15

    def _orders(self, depth: OrderDepth, position: int):
        orders: List[Order] = []
        
        # 1. Identify Market Tops
        buy_prices = sorted(depth.buy_orders.keys(), reverse=True)
        sell_prices = sorted(depth.sell_orders.keys())
        
        best_bid = buy_prices[0] if buy_prices else None
        best_ask = sell_prices[0] if sell_prices else None
        
        # Calculate a conservative Mid-Price
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
        else:
            mid_price = 10_000 

        rem_buy  = self.HP_LIMIT - position
        rem_sell = self.HP_LIMIT + position

        # 2. Aggressive Taking (Market Crossing)
        # We take liquidity if it's even slightly better than mid-price
        if best_ask is not None and best_ask <= mid_price - self.HP_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.HP_AGGR_SIZE, rem_buy)
            orders.append(Order(self.HP_PRODUCT, best_ask, vol))
            rem_buy -= vol
            
        if best_bid is not None and best_bid >= mid_price + self.HP_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.HP_AGGR_SIZE, rem_sell)
            orders.append(Order(self.HP_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # 3. Competitive Market Making (Pennying)
        # Instead of a fixed offset, we try to be the BEST price in the book
        # We adjust based on inventory skew
        skew = int((position / self.HP_LIMIT) * 5) # Smaller skew for price competitiveness
        
        if best_bid is not None:
            # Bid 1 tick above the best current bid, adjusted for skew
            my_bid = best_bid + 1 - (1 if position > 10 else 0)
        else:
            my_bid = int(mid_price - 5)

        if best_ask is not None:
            # Ask 1 tick below the best current ask, adjusted for skew
            my_ask = best_ask - 1 + (1 if position < -10 else 0)
        else:
            my_ask = int(mid_price + 5)

        # 4. Final Safety & Execution
        # Ensure we don't cross our own spread
        if my_bid >= my_ask:
            my_bid = my_ask - 1

        if rem_buy > 0:
            orders.append(Order(self.HP_PRODUCT, int(my_bid), min(self.HP_MM_SIZE, rem_buy)))
        if rem_sell > 0:
            orders.append(Order(self.HP_PRODUCT, int(my_ask), -min(self.HP_MM_SIZE, rem_sell)))

        return orders

    def run(self, state: TradingState):
        result = {}
        if self.HP_PRODUCT in state.order_depths:
            depth = state.order_depths[self.HP_PRODUCT]
            position = state.position.get(self.HP_PRODUCT, 0)
            result[self.HP_PRODUCT] = self._orders(depth, position)
        
        return result, 0, "HP_COMPETITIVE"