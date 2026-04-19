from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class Trader:
    # ------------------------------------------------------------------ #
    # ASH_COATED_OSMIUM                                                    #
    #   - Stationary mean-reverting process anchored to 10 000             #
    #   - Lag-1 return autocorrelation ≈ -0.50  →  mean-revert, NOT trend  #
    #   - Market bid-ask spread ≈ 16 units, price std ≈ 5 units            #
    #   - Strategy: tight passive market making + aggressive mean-reversion #
    # ------------------------------------------------------------------ #

    PRODUCT = "ASH_COATED_OSMIUM"
    FAIR_VALUE = 10_000          # stable across all days, no drift
    POSITION_LIMIT = 80          # adjust to actual competition limit

    PASSIVE_SPREAD = 3           # quote at FV±3, inside the ~16-unit market spread
    MAX_PASSIVE_SIZE = 15        # match typical book depth
    AGGRESS_EDGE = 10             # take when best_ask < FV-7 or best_bid > FV+7 (~1.4σ, same recovery rate as 10 but 3x more opportunities)
    MAX_AGGRESS_SIZE = 20        # how much to lift/hit per tick when aggressively taking

    def run(self, state: TradingState):
        result = {}
        orders: List[Order] = []

        if self.PRODUCT not in state.order_depths:
            return result, 0, state.traderData or ""

        order_depth: OrderDepth = state.order_depths[self.PRODUCT]
        position = state.position.get(self.PRODUCT, 0)

        best_bid = max(order_depth.buy_orders.keys())  if order_depth.buy_orders  else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        fv = self.FAIR_VALUE

        remaining_buy  = self.POSITION_LIMIT - position
        remaining_sell = self.POSITION_LIMIT + position

        print(f"[{self.PRODUCT}] pos={position} fv={fv} best_bid={best_bid} best_ask={best_ask}")

        # ---------------------------------------------------------------- #
        # 1. AGGRESSIVE TAKING: price has strayed far from fair value       #
        # ---------------------------------------------------------------- #
        # Lift cheap asks (price dipped below FV - edge → expect reversion up)
        if best_ask is not None and best_ask < fv - self.AGGRESS_EDGE and remaining_buy > 0:
            vol = min(-order_depth.sell_orders[best_ask], remaining_buy, self.MAX_AGGRESS_SIZE)
            print(f"  AGGRESS BUY  {vol}x @ {best_ask}  (fv-ask={fv-best_ask:.0f})")
            orders.append(Order(self.PRODUCT, best_ask, vol))
            remaining_buy -= vol

        # Hit rich bids (price spiked above FV + edge → expect reversion down)
        if best_bid is not None and best_bid > fv + self.AGGRESS_EDGE and remaining_sell > 0:
            vol = min(order_depth.buy_orders[best_bid], remaining_sell, self.MAX_AGGRESS_SIZE)
            print(f"  AGGRESS SELL {vol}x @ {best_bid}  (bid-fv={best_bid-fv:.0f})")
            orders.append(Order(self.PRODUCT, best_bid, -vol))
            remaining_sell -= vol

        # ---------------------------------------------------------------- #
        # 2. PASSIVE MARKET MAKING: quote inside the spread                 #
        #    Skew both quotes toward neutrality when position is large.     #
        # ---------------------------------------------------------------- #
        # Skew: shift quotes by up to PASSIVE_SPREAD units based on position
        skew = int(position / self.POSITION_LIMIT * self.PASSIVE_SPREAD)
        my_bid = fv - self.PASSIVE_SPREAD - skew
        my_ask = fv + self.PASSIVE_SPREAD - skew

        if remaining_buy > 0:
            buy_size = min(self.MAX_PASSIVE_SIZE, remaining_buy)
            print(f"  POST BID {buy_size}x @ {my_bid}")
            orders.append(Order(self.PRODUCT, my_bid, buy_size))

        if remaining_sell > 0:
            sell_size = min(self.MAX_PASSIVE_SIZE, remaining_sell)
            print(f"  POST ASK {sell_size}x @ {my_ask}")
            orders.append(Order(self.PRODUCT, my_ask, -sell_size))

        result[self.PRODUCT] = orders
        return result, 0, state.traderData or ""
