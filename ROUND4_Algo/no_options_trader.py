from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:
    # --- HYDROGEL_PACK Constants ---
    HP_PRODUCT   = "HYDROGEL_PACK"
    HP_LIMIT     = 60
    HP_MM_SIZE   = 8
    HP_AGGR_EDGE = 5
    HP_AGGR_SIZE = 20

    # --- VELVETFRUIT_EXTRACT Constants ---
    VFE_PRODUCT   = "VELVETFRUIT_EXTRACT"
    VFE_FV        = 5_250
    VFE_LIMIT     = 60
    VFE_MM_HALF   = 8
    VFE_MM_SIZE   = 8
    VFE_AGGR_EDGE = 20
    VFE_AGGR_SIZE = 8

    # --- Counterparty IDs ---
    # Mark 67: one-sided VEV buyer, ~55 trades/day, buys at ~mid+1
    # Mark 22, Mark 49: Mark 67's exclusive suppliers in VEV (89 and 75 trades resp.)
    # Mark 14, Mark 38: symmetric HGP market makers (~500 buys and ~500 sells each)
    MARK_67 = "Mark 67"
    MARK_22 = "Mark 22"
    MARK_49 = "Mark 49"
    MARK_14 = "Mark 14"
    MARK_38 = "Mark 38"

    VFE_M67_SELL_SIZE  = 15   # sell into Mark 67's next buy
    VFE_FEED_SELL_SIZE = 8    # sell when his known suppliers are active

    def _vfe_signals(self, vfe_trades: list):
        """Parse last-tick VEV trades for counterparty signals."""
        mark67_bought = False
        feeder_sold   = False    # Mark 22 or Mark 49 sold VEV
        for t in vfe_trades:
            if t.buyer == self.MARK_67:
                mark67_bought = True
            if t.seller in (self.MARK_22, self.MARK_49):
                feeder_sold = True
        return mark67_bought, feeder_sold

    def _hp_fair_value(self, hp_trades: list, book_mid: float) -> float:
        """
        Mark 14 and Mark 38 are symmetric HGP market makers whose average trade
        price tracks fair value closely. Use their last-tick trade prices to
        anchor our mid estimate when available.
        """
        prices = []
        for t in hp_trades:
            if t.buyer in (self.MARK_14, self.MARK_38) or t.seller in (self.MARK_14, self.MARK_38):
                prices.append(t.price)
        if prices:
            # blend 50/50 with book mid to avoid overreacting to single prints
            return 0.5 * (sum(prices) / len(prices)) + 0.5 * book_mid
        return book_mid

    def _get_hp_orders(self, depth: OrderDepth, position: int, hp_trades: list):
        orders: List[Order] = []

        buy_prices  = sorted(depth.buy_orders.keys(),  reverse=True)
        sell_prices = sorted(depth.sell_orders.keys())

        best_bid = buy_prices[0]  if buy_prices  else None
        best_ask = sell_prices[0] if sell_prices else None

        book_mid  = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 10_000
        mid_price = self._hp_fair_value(hp_trades, book_mid)

        rem_buy  = self.HP_LIMIT - position
        rem_sell = self.HP_LIMIT + position

        # Aggressive taking against fair value
        if best_ask is not None and best_ask <= mid_price - self.HP_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.HP_AGGR_SIZE, rem_buy)
            orders.append(Order(self.HP_PRODUCT, best_ask, vol))
            rem_buy -= vol

        if best_bid is not None and best_bid >= mid_price + self.HP_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.HP_AGGR_SIZE, rem_sell)
            orders.append(Order(self.HP_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # Competitive market making with inventory skew
        skew_thr = self.HP_LIMIT // 3

        if best_bid is not None:
            my_bid = best_bid + 1 - (1 if position > skew_thr else 0)
        else:
            my_bid = int(mid_price - 5)

        if best_ask is not None:
            my_ask = best_ask - 1 + (1 if position < -skew_thr else 0)
        else:
            my_ask = int(mid_price + 5)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        if rem_buy > 0:
            orders.append(Order(self.HP_PRODUCT, int(my_bid), min(self.HP_MM_SIZE, rem_buy)))
        if rem_sell > 0:
            orders.append(Order(self.HP_PRODUCT, int(my_ask), -min(self.HP_MM_SIZE, rem_sell)))

        return orders

    def _get_vfe_orders(self, depth: OrderDepth, position: int, vfe_trades: list):
        orders: List[Order] = []

        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None

        rem_buy  = self.VFE_LIMIT - position
        rem_sell = self.VFE_LIMIT + position

        mark67_bought, feeder_sold = self._vfe_signals(vfe_trades)

        # Aggressive taking against static fair value
        if best_ask is not None and best_ask < self.VFE_FV - self.VFE_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.VFE_AGGR_SIZE, rem_buy)
            orders.append(Order(self.VFE_PRODUCT, best_ask, vol))
            rem_buy -= vol
        if best_bid is not None and best_bid > self.VFE_FV + self.VFE_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.VFE_AGGR_SIZE, rem_sell)
            orders.append(Order(self.VFE_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # Mark 67 bought last tick → he's likely to hit the ask again next tick.
        # Post a dedicated sell at best_ask to capture his demand.
        if mark67_bought and rem_sell > 0 and best_ask is not None:
            vol = min(self.VFE_M67_SELL_SIZE, rem_sell)
            orders.append(Order(self.VFE_PRODUCT, best_ask, -vol))
            rem_sell -= vol

        # Mark 22 or Mark 49 sold VEV last tick (his known feeders).
        # Their activity signals Mark 67 is in an active buying cycle even if
        # he didn't fill last tick himself — post a smaller sell at the ask.
        elif feeder_sold and rem_sell > 0 and best_ask is not None:
            vol = min(self.VFE_FEED_SELL_SIZE, rem_sell)
            orders.append(Order(self.VFE_PRODUCT, best_ask, -vol))
            rem_sell -= vol

        # Passive limit orders with inventory skew
        skew   = int(position / self.VFE_LIMIT * self.VFE_MM_HALF * 2)
        my_bid = self.VFE_FV - self.VFE_MM_HALF - skew
        my_ask = self.VFE_FV + self.VFE_MM_HALF - skew

        skew_thr = self.VFE_LIMIT // 3
        if position > skew_thr and best_ask is not None:
            my_ask = min(my_ask, best_ask)
        if position < -skew_thr and best_bid is not None:
            my_bid = max(my_bid, best_bid)

        if rem_buy > 0 and (best_ask is None or my_bid < best_ask):
            orders.append(Order(self.VFE_PRODUCT, my_bid, min(self.VFE_MM_SIZE, rem_buy)))
        if rem_sell > 0 and (best_bid is None or my_ask > best_bid):
            orders.append(Order(self.VFE_PRODUCT, my_ask, -min(self.VFE_MM_SIZE, rem_sell)))

        return orders

    def run(self, state: TradingState):
        result = {}

        vfe_trades = state.market_trades.get(self.VFE_PRODUCT, [])
        hp_trades  = state.market_trades.get(self.HP_PRODUCT, [])

        if self.HP_PRODUCT in state.order_depths:
            hp_depth    = state.order_depths[self.HP_PRODUCT]
            hp_position = state.position.get(self.HP_PRODUCT, 0)
            result[self.HP_PRODUCT] = self._get_hp_orders(hp_depth, hp_position, hp_trades)

        if self.VFE_PRODUCT in state.order_depths:
            vfe_depth    = state.order_depths[self.VFE_PRODUCT]
            vfe_position = state.position.get(self.VFE_PRODUCT, 0)
            result[self.VFE_PRODUCT] = self._get_vfe_orders(vfe_depth, vfe_position, vfe_trades)

        return result, 0, "COMBINED_STRATEGY"
