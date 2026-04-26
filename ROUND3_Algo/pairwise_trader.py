import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Any, Dict

class Trader:
    """
    Pair trade: HYDROGEL_PACK vs VELVETFRUIT_EXTRACT.

    v5 enhancements:
    - Rolling OLS hedge ratio (200-tick window) instead of hardcoded 1.9.
    - EMA warm-up guard: no trading until EMA and variance have converged.
    - Z-score thresholds (entry z>2.0, exit z<0.5) instead of fixed edge constants.
    """

    FALLBACK_HEDGE_RATIO = 1.9
    OLS_CLAMP = (1.5, 2.3)       # prevent noisy OLS from drifting too far from prior

    HISTORICAL_MEAN = 15.6       # computed from days 0-1-2 price data
    HISTORICAL_VAR  = 1718.0     # spread std≈41.45; initialises ema_var from tick 1
    EMA_ALPHA = 0.005

    ENTRY_Z = 2.0                # statistically principled, not backtested
    EXIT_Z = 0.5

    HP = "HYDROGEL_PACK"
    VFE = "VELVETFRUIT_EXTRACT"
    HP_LIMIT = 50
    VFE_LIMIT = 50
    MAX_ORDER_HP = 5

    HEDGE_WINDOW = 200
    WARMUP_TICKS = 50            # reduced: ema_var pre-seeded so convergence is fast

    def _ols_beta(self, hp_prices: list, vfe_prices: list) -> float:
        n = len(hp_prices)
        if n < 2:
            return self.FALLBACK_HEDGE_RATIO
        sum_x = sum(vfe_prices)
        sum_y = sum(hp_prices)
        sum_xy = sum(x * y for x, y in zip(vfe_prices, hp_prices))
        sum_x2 = sum(x * x for x in vfe_prices)
        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return self.FALLBACK_HEDGE_RATIO
        return (n * sum_xy - sum_x * sum_y) / denom

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        result = {self.HP: [], self.VFE: []}

        trader_state = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except json.JSONDecodeError:
                pass

        ema = trader_state.get("spread_ema", self.HISTORICAL_MEAN)
        ema_var = trader_state.get("spread_ema_var", self.HISTORICAL_VAR)
        tick_count = trader_state.get("tick_count", 0)
        hp_history = trader_state.get("hp_history", [])
        vfe_history = trader_state.get("vfe_history", [])

        hp_depth: OrderDepth = state.order_depths.get(self.HP)
        vfe_depth: OrderDepth = state.order_depths.get(self.VFE)

        if not hp_depth or not vfe_depth:
            return result, 0, json.dumps(trader_state)

        hp_bids, hp_asks = hp_depth.buy_orders, hp_depth.sell_orders
        vfe_bids, vfe_asks = vfe_depth.buy_orders, vfe_depth.sell_orders

        if not hp_bids or not hp_asks or not vfe_bids or not vfe_asks:
            return result, 0, json.dumps(trader_state)

        hp_bid, hp_ask = max(hp_bids), min(hp_asks)
        vfe_bid, vfe_ask = max(vfe_bids), min(vfe_asks)

        hp_mid = (hp_bid + hp_ask) / 2
        vfe_mid = (vfe_bid + vfe_ask) / 2

        # 1. Update rolling price history for OLS
        hp_history.append(hp_mid)
        vfe_history.append(vfe_mid)
        if len(hp_history) > self.HEDGE_WINDOW:
            hp_history = hp_history[-self.HEDGE_WINDOW:]
            vfe_history = vfe_history[-self.HEDGE_WINDOW:]

        if len(hp_history) >= self.HEDGE_WINDOW:
            raw_beta = self._ols_beta(hp_history, vfe_history)
            hedge_ratio = max(self.OLS_CLAMP[0], min(self.OLS_CLAMP[1], raw_beta))
        else:
            hedge_ratio = self.FALLBACK_HEDGE_RATIO

        # 2. Update spread EMA and exponential moving variance
        spread = hp_mid - hedge_ratio * vfe_mid
        deviation_from_old_ema = spread - ema
        ema = self.EMA_ALPHA * spread + (1 - self.EMA_ALPHA) * ema
        ema_var = self.EMA_ALPHA * deviation_from_old_ema ** 2 + (1 - self.EMA_ALPHA) * ema_var

        tick_count += 1

        trader_state["spread_ema"] = ema
        trader_state["spread_ema_var"] = ema_var
        trader_state["tick_count"] = tick_count
        trader_state["hp_history"] = hp_history
        trader_state["vfe_history"] = vfe_history

        # 3. Warm-up guard: wait for EMA and variance to converge
        if tick_count <= self.WARMUP_TICKS or ema_var <= 0:
            return result, 0, json.dumps(trader_state)

        deviation = spread - ema
        z_score = deviation / math.sqrt(ema_var)

        hp_pos = state.position.get(self.HP, 0)
        vfe_pos = state.position.get(self.VFE, 0)

        # ------------------------------------------------------------------ #
        # EXIT LOGIC: spread reverted toward dynamic mean                     #
        # ------------------------------------------------------------------ #
        if hp_pos < 0 and z_score <= self.EXIT_Z:
            # Short HP, long VFE -> Buy HP, Sell VFE
            close_hp = min(abs(hp_pos), self.MAX_ORDER_HP, abs(hp_asks.get(hp_ask, 0)))
            close_vfe = round(close_hp * hedge_ratio)
            if close_hp > 0 and vfe_bids.get(vfe_bid, 0) >= close_vfe:
                result[self.HP].append(Order(self.HP, hp_ask, close_hp))
                result[self.VFE].append(Order(self.VFE, vfe_bid, -close_vfe))

        elif hp_pos > 0 and z_score >= -self.EXIT_Z:
            # Long HP, short VFE -> Sell HP, Buy VFE
            close_hp = min(hp_pos, self.MAX_ORDER_HP, hp_bids.get(hp_bid, 0))
            close_vfe = round(close_hp * hedge_ratio)
            if close_hp > 0 and abs(vfe_asks.get(vfe_ask, 0)) >= close_vfe:
                result[self.HP].append(Order(self.HP, hp_bid, -close_hp))
                result[self.VFE].append(Order(self.VFE, vfe_ask, close_vfe))

        # ------------------------------------------------------------------ #
        # ENTRY LOGIC: spread deviates significantly from dynamic mean        #
        # ------------------------------------------------------------------ #
        elif z_score > self.ENTRY_Z:
            # Spread too high: Sell HP, Buy VFE
            space_hp = self.HP_LIMIT + hp_pos
            trade_hp = min(self.MAX_ORDER_HP, space_hp, hp_bids.get(hp_bid, 0))
            trade_vfe = round(trade_hp * hedge_ratio)
            space_vfe = self.VFE_LIMIT - vfe_pos
            if trade_vfe > space_vfe:
                trade_vfe = space_vfe
                trade_hp = round(trade_vfe / hedge_ratio)
            if trade_hp > 0 and trade_vfe > 0 and abs(vfe_asks.get(vfe_ask, 0)) >= trade_vfe:
                result[self.HP].append(Order(self.HP, hp_bid, -trade_hp))
                result[self.VFE].append(Order(self.VFE, vfe_ask, trade_vfe))

        elif z_score < -self.ENTRY_Z:
            # Spread too low: Buy HP, Sell VFE
            space_hp = self.HP_LIMIT - hp_pos
            trade_hp = min(self.MAX_ORDER_HP, space_hp, abs(hp_asks.get(hp_ask, 0)))
            trade_vfe = round(trade_hp * hedge_ratio)
            space_vfe = self.VFE_LIMIT + vfe_pos
            if trade_vfe > space_vfe:
                trade_vfe = space_vfe
                trade_hp = round(trade_vfe / hedge_ratio)
            if trade_hp > 0 and trade_vfe > 0 and vfe_bids.get(vfe_bid, 0) >= trade_vfe:
                result[self.HP].append(Order(self.HP, hp_ask, trade_hp))
                result[self.VFE].append(Order(self.VFE, vfe_bid, -trade_vfe))

        return result, 0, json.dumps(trader_state)
