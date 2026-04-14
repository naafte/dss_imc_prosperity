"""
Ultra-naive INTARIAN_PEPPER_ROOT only:
1) Buy up to TARGET_LONG as fast as possible (lift every ask until capped).
  2) Send no orders for HOLD_RUNS calls (wait as long as we dare in a ~10k sim).
  3) Dump the long by hitting bids until flat.

Contract: run -> (result, conversions, traderData). Only INTARIAN_PEPPER_ROOT appears in result.

If the official position limit is below TARGET_LONG, the exchange will reject oversized buys — align
TARGET_LONG with the wiki limit before submitting.
"""

from __future__ import annotations

import json
from datamodel import Order, TradingState
from typing import Any, Dict, List

PRODUCT = "INTARIAN_PEPPER_ROOT"
TARGET_LONG = 80
# After we reach TARGET_LONG, do nothing this many run() invocations, then liquidate.
# ~10k-step finals: leaves ~1k steps to sell. For 1k-step sandbox testing, lower this (e.g. 850).
HOLD_RUNS_BEFORE_SELL = 9000


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        stored: Dict[str, Any] = {}

        #LOADS TRADER DATA
        if state.traderData:
            try:
                stored = json.loads(state.traderData)
            except json.JSONDecodeError:
                stored = {}
        
        stage = stored.get("stage", "accumulate")
        hold_runs = int(stored.get("hold_runs", 0))

        result: Dict[str, List[Order]] = {}
        conversions = 0

        depth = state.order_depths.get(PRODUCT)
        if depth is None:
            trader_data = json.dumps(
                {"stage": stage, "hold_runs": hold_runs}, separators=(",", ":")
            )
            return result, conversions, trader_data

        position = state.position.get(PRODUCT, 0)
        orders: List[Order] = []

        if stage == "accumulate":
            if position >= TARGET_LONG:
                stage = "hold"
                hold_runs = 0
            else:
                buy_cap = TARGET_LONG - position
                for ask_price in sorted(depth.sell_orders.keys()):
                    if buy_cap <= 0:
                        break
                    vol = depth.sell_orders[ask_price]
                    qty = min(-vol, buy_cap)
                    if qty > 0:
                        orders.append(Order(PRODUCT, int(ask_price), qty))
                        buy_cap -= qty

        elif stage == "hold":
            if position < TARGET_LONG:
                stage = "accumulate"
                hold_runs = 0
            else:
                hold_runs += 1
                if hold_runs >= HOLD_RUNS_BEFORE_SELL:
                    stage = "dump"

        elif stage == "dump":
            if position <= 0:
                stage = "done"
            else:
                to_sell = position
                for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                    if to_sell <= 0:
                        break
                    vol = depth.buy_orders[bid_price]
                    qty = min(vol, to_sell)
                    if qty > 0:
                        orders.append(Order(PRODUCT, int(bid_price), -qty))
                        to_sell -= qty

        elif stage == "done":
            pass

        if orders:
            result[PRODUCT] = orders

        trader_data = json.dumps({"stage": stage, "hold_runs": hold_runs}, separators=(",", ":"))
        return result, conversions, trader_data
