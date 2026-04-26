import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Any, Dict


class Trader:
    HP  = "HYDROGEL_PACK"
    VFE = "VELVETFRUIT_EXTRACT"
    HP_LIMIT  = 50
    VFE_LIMIT = 50

    # EMA periods for fast/slow crossover
    FAST_ALPHA = 2 / (8  + 1)   # ~8-tick EMA
    SLOW_ALPHA = 2 / (32 + 1)   # ~32-tick EMA

    MAX_ORDER = 5
    WARMUP_TICKS = 32

    def _update_emas(self, state: dict, symbol: str, mid: float):
        fast = state.get(f"{symbol}_fast", mid)
        slow = state.get(f"{symbol}_slow", mid)
        state[f"{symbol}_fast"] = self.FAST_ALPHA * mid + (1 - self.FAST_ALPHA) * fast
        state[f"{symbol}_slow"] = self.SLOW_ALPHA * mid + (1 - self.SLOW_ALPHA) * slow

    def _momentum_orders(
        self,
        symbol: str,
        state_ts: TradingState,
        trader_state: dict,
        limit: int,
    ) -> List[Order]:
        od: OrderDepth = state_ts.order_depths.get(symbol)
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2

        self._update_emas(trader_state, symbol, mid)

        tick = trader_state.get("tick_count", 0)
        if tick < self.WARMUP_TICKS:
            return []

        fast = trader_state[f"{symbol}_fast"]
        slow = trader_state[f"{symbol}_slow"]
        pos  = state_ts.position.get(symbol, 0)
        orders: List[Order] = []

        if fast > slow:
            # Bullish: buy up to limit
            room = limit - pos
            qty  = min(self.MAX_ORDER, room, abs(od.sell_orders.get(best_ask, 0)))
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))
        elif fast < slow:
            # Bearish: sell down to -limit
            room = limit + pos
            qty  = min(self.MAX_ORDER, room, od.buy_orders.get(best_bid, 0))
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        trader_state: dict = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except json.JSONDecodeError:
                pass

        trader_state["tick_count"] = trader_state.get("tick_count", 0) + 1

        result: Dict[str, List[Order]] = {self.HP: [], self.VFE: []}

        result[self.HP]  = self._momentum_orders(self.HP,  state, trader_state, self.HP_LIMIT)
        result[self.VFE] = self._momentum_orders(self.VFE, state, trader_state, self.VFE_LIMIT)

        return result, 0, json.dumps(trader_state)
