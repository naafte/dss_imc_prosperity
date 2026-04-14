from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json

class Trader:
    PRODUCT = "INTARIAN_PEPPER_ROOT"
    POSITION_LIMIT = 80  # adjust to actual limit if different

    # Fair value per day: day -2 -> 10500, day -1 -> 11500, day 0 -> 12500
    # Formula: 10500 + 1000 * (day + 2)
    FAIR_VALUES = {-2: 10500, -1: 11500, 0: 12500}

    # How aggressively to quote around fair value
    SPREAD = 2          # post bid at FV-2, ask at FV+2
    MAX_ORDER_SIZE = 5  # size per order
    AGGRESS_EDGE = 7    # if best_ask < FV - edge, lift it (take edge); same on sell side

    def fair_value(self, day: int, mid_prices: list) -> float:
        """
        Fair value = known daily baseline + EMA adjustment for intraday drift.
        Falls back to daily baseline if no mid price history yet.
        """
        baseline = self.FAIR_VALUES.get(day, 10500 + 1000 * (day + 2))
        if not mid_prices:
            return baseline
        # Blend baseline with recent mid price (EMA-style)
        alpha = 0.15
        ema = mid_prices[-1]
        for m in reversed(mid_prices[-10:]):
            ema = alpha * m + (1 - alpha) * ema
        # Stay anchored to baseline -- don't drift more than 150 from it
        return baseline + max(-150, min(150, ema - baseline))

    def run(self, state: TradingState):
        result = {}
        orders: List[Order] = []

        # --- Load persisted state ---
        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}

        mid_history: list = trader_state.get("mid_history", [])

        # --- Only trade INTARIAN_PEPPER_ROOT ---
        if self.PRODUCT not in state.order_depths:
            return result, 0, json.dumps({"mid_history": mid_history})

        order_depth: OrderDepth = state.order_depths[self.PRODUCT]
        position = state.position.get(self.PRODUCT, 0)

        # --- Compute current mid price and update history ---
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2
            mid_history.append(mid)
            mid_history = mid_history[-50:]  # keep last 50 ticks

        fv = self.fair_value(state.timestamp // 1_000_000 if state.timestamp > 1000 else 0,
                             mid_history)
        # Day is stored in the day field of TradingState in Prosperity
        day = getattr(state, "day", 0)
        fv = self.fair_value(day, mid_history)

        print(f"[{self.PRODUCT}] day={day} pos={position} fv={fv:.1f} "
              f"best_bid={best_bid} best_ask={best_ask}")

        remaining_buy  = self.POSITION_LIMIT - position   # how much more we can buy
        remaining_sell = self.POSITION_LIMIT + position   # how much more we can sell

        # -------------------------------------------------------
        # 1. AGGRESSIVE TAKING: lift/hit mispriced orders
        # -------------------------------------------------------
        if best_ask is not None and best_ask < fv - self.AGGRESS_EDGE and remaining_buy > 0:
            vol = min(-order_depth.sell_orders[best_ask], remaining_buy, self.MAX_ORDER_SIZE)
            print(f"  TAKE BUY {vol}x @ {best_ask}")
            orders.append(Order(self.PRODUCT, best_ask, vol))
            remaining_buy -= vol

        if best_bid is not None and best_bid > fv + self.AGGRESS_EDGE and remaining_sell > 0:
            vol = min(order_depth.buy_orders[best_bid], remaining_sell, self.MAX_ORDER_SIZE)
            print(f"  TAKE SELL {vol}x @ {best_bid}")
            orders.append(Order(self.PRODUCT, best_bid, -vol))
            remaining_sell -= vol

        # -------------------------------------------------------
        # 2. PASSIVE MARKET MAKING: post bid/ask around fair value
        # -------------------------------------------------------
        my_bid = round(fv - self.SPREAD)
        my_ask = round(fv + self.SPREAD)

        # Skew quotes toward neutrality when position is large
        skew = int(position / self.POSITION_LIMIT * self.SPREAD)
        my_bid -= skew
        my_ask -= skew

        if remaining_buy > 0:
            buy_size = min(self.MAX_ORDER_SIZE, remaining_buy)
            print(f"  POST BID {buy_size}x @ {my_bid}")
            orders.append(Order(self.PRODUCT, my_bid, buy_size))

        if remaining_sell > 0:
            sell_size = min(self.MAX_ORDER_SIZE, remaining_sell)
            print(f"  POST ASK {sell_size}x @ {my_ask}")
            orders.append(Order(self.PRODUCT, my_ask, -sell_size))

        result[self.PRODUCT] = orders

        # --- Persist state ---
        traderData = json.dumps({"mid_history": mid_history})
        return result, 0, traderData