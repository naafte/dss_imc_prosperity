"""
no_options_trader_v3.py — HYDROGEL_PACK + VELVETFRUIT_EXTRACT only (no options).

Key improvements over v2 (variance-neutral changes only):
  - Larger fixed M67 sell sizes (50 vs 25) and feeder sell sizes (30 vs 15).
  - Always-on small VFE sell at best_ask (8 contracts) even without M67 signal.
  - VFE FV anchored using Mark 67's trade prices directly (he buys at FV+1, so
    FV = M67_price - 1) — more accurate than pure mid EMA.
  - HP: MM size 20 (was 15), aggr size 60 (was 40). Edge kept at 3 to avoid
    picking up noise on marginal levels.
  - VFE MM: symmetric 8-tick spread (unchanged) — no directional bias.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Optional
import json

# ── Position limits ────────────────────────────────────────────────────────────
HP_LIMIT  = 200
VFE_LIMIT = 200

# ── HYDROGEL_PACK parameters ───────────────────────────────────────────────────
HP_PRODUCT    = "HYDROGEL_PACK"
HP_FV_INIT    = 10_000
HP_FV_ALPHA   = 0.20
HP_AGGR_EDGE  = 3        # same as v2 — keeps edge quality high
HP_AGGR_SIZE  = 60       # larger sweep size (was 40)
HP_MM_SIZE    = 20       # larger passive size (was 15)
HP_SKEW_THR   = 60

# ── VELVETFRUIT_EXTRACT parameters ────────────────────────────────────────────
VFE_PRODUCT   = "VELVETFRUIT_EXTRACT"
VFE_FV_INIT   = 5_250
VFE_FV_ALPHA  = 0.15
VFE_AGGR_EDGE = 15       # same as v2 — avoid picking off on noise
VFE_AGGR_SIZE = 60       # larger sweep size (was 40)
VFE_MM_HALF   = 8        # symmetric half-spread (same as v2)
VFE_MM_SIZE   = 20
VFE_SKEW_THR  = 60

# ── Mark 67 sell parameters ────────────────────────────────────────────────────
M67_SELL_SIZE  = 50   # sell size when M67 bought last tick (was 25)
FEED_SELL_SIZE = 30   # sell size when feeders active (was 15)
M67_ALWAYS_SELL = 8   # always-on small sell at ask even without M67 signal

# ── Counterparty IDs ───────────────────────────────────────────────────────────
MARK_14 = "Mark 14"
MARK_38 = "Mark 38"
MARK_55 = "Mark 55"   # symmetric VFE market maker (~600 buys, ~600 sells/day)
MARK_67 = "Mark 67"
MARK_22 = "Mark 22"
MARK_49 = "Mark 49"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _best_bid(depth: OrderDepth) -> Optional[int]:
    return max(depth.buy_orders) if depth.buy_orders else None

def _best_ask(depth: OrderDepth) -> Optional[int]:
    return min(depth.sell_orders) if depth.sell_orders else None

def _mid(depth: OrderDepth) -> Optional[float]:
    b, a = _best_bid(depth), _best_ask(depth)
    return (b + a) / 2.0 if (b is not None and a is not None) else None

def _ema(prev: float, obs: float, alpha: float) -> float:
    return alpha * obs + (1.0 - alpha) * prev


# ── Trader ─────────────────────────────────────────────────────────────────────

class Trader:

    # ── HGP fair value ──────────────────────────────────────────────────────────

    def _hp_fv_update(self, hp_trades: list, book_mid: float, cached_fv: float) -> float:
        mm_prices = [
            t.price for t in hp_trades
            if t.buyer in (MARK_14, MARK_38) or t.seller in (MARK_14, MARK_38)
        ]
        if mm_prices:
            mm_avg = sum(mm_prices) / len(mm_prices)
            blended = 0.5 * mm_avg + 0.5 * book_mid
            return _ema(cached_fv, blended, HP_FV_ALPHA)
        return _ema(cached_fv, book_mid, HP_FV_ALPHA * 0.5)

    # ── VFE counterparty signals ────────────────────────────────────────────────

    @staticmethod
    def _vfe_signals(vfe_trades: list):
        mark67_bought = any(t.buyer == MARK_67 for t in vfe_trades)
        feeder_sold   = any(t.seller in (MARK_22, MARK_49, MARK_55) for t in vfe_trades)
        m67_prices    = [t.price for t in vfe_trades if t.buyer == MARK_67]
        m55_prices    = [t.price for t in vfe_trades
                         if t.buyer == MARK_55 or t.seller == MARK_55]
        return mark67_bought, feeder_sold, m67_prices, m55_prices

    # ── HGP orders ─────────────────────────────────────────────────────────────

    def _get_hp_orders(self, depth: OrderDepth, position: int,
                       hp_trades: list, cached_fv: float):
        orders: List[Order] = []

        bb = _best_bid(depth)
        ba = _best_ask(depth)
        book_mid = (bb + ba) / 2.0 if (bb is not None and ba is not None) else cached_fv

        fv = self._hp_fv_update(hp_trades, book_mid, cached_fv)

        rem_buy  = HP_LIMIT - position
        rem_sell = HP_LIMIT + position

        # Multi-level aggressive sweep
        for ask_px in sorted(depth.sell_orders):
            if ask_px > fv - HP_AGGR_EDGE:
                break
            if rem_buy <= 0:
                break
            vol = min(-depth.sell_orders[ask_px], HP_AGGR_SIZE, rem_buy)
            orders.append(Order(HP_PRODUCT, ask_px, vol))
            rem_buy -= vol

        for bid_px in sorted(depth.buy_orders, reverse=True):
            if bid_px < fv + HP_AGGR_EDGE:
                break
            if rem_sell <= 0:
                break
            vol = min(depth.buy_orders[bid_px], HP_AGGR_SIZE, rem_sell)
            orders.append(Order(HP_PRODUCT, bid_px, -vol))
            rem_sell -= vol

        # Passive market making with inventory skew
        if bb is not None:
            my_bid = bb + 1 - (1 if position > HP_SKEW_THR else 0)
        else:
            my_bid = int(fv - 5)

        if ba is not None:
            my_ask = ba - 1 + (1 if position < -HP_SKEW_THR else 0)
        else:
            my_ask = int(fv + 5)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        if rem_buy > 0:
            orders.append(Order(HP_PRODUCT, int(my_bid), min(HP_MM_SIZE, rem_buy)))
        if rem_sell > 0:
            orders.append(Order(HP_PRODUCT, int(my_ask), -min(HP_MM_SIZE, rem_sell)))

        return orders, fv

    # ── VFE orders ──────────────────────────────────────────────────────────────

    def _get_vfe_orders(self, depth: OrderDepth, position: int,
                        vfe_trades: list, cached_fv: float):
        orders: List[Order] = []

        bb = _best_bid(depth)
        ba = _best_ask(depth)
        mid = (bb + ba) / 2.0 if (bb is not None and ba is not None) else None

        mark67_bought, feeder_sold, m67_prices, m55_prices = self._vfe_signals(vfe_trades)

        # VFE FV: priority — Mark 67 prints > Mark 55 MM prints > mid EMA
        # Mark 67 buys at FV+1, so implied FV = his price - 1.
        # Mark 55 is a symmetric MM; blend his avg trade price with mid (same
        # logic as Mark 14/38 for HGP).
        if m67_prices:
            m67_implied_fv = sum(m67_prices) / len(m67_prices) - 1.0
            fv = _ema(cached_fv, m67_implied_fv, VFE_FV_ALPHA * 2)
        elif m55_prices:
            m55_avg = sum(m55_prices) / len(m55_prices)
            blended = 0.5 * m55_avg + 0.5 * (mid if mid is not None else cached_fv)
            fv = _ema(cached_fv, blended, VFE_FV_ALPHA)
        elif mid is not None:
            fv = _ema(cached_fv, mid, VFE_FV_ALPHA * 0.5)
        else:
            fv = cached_fv

        rem_buy  = VFE_LIMIT - position
        rem_sell = VFE_LIMIT + position

        # Multi-level aggressive take against fair value
        for ask_px in sorted(depth.sell_orders):
            if ask_px > fv - VFE_AGGR_EDGE:
                break
            if rem_buy <= 0:
                break
            vol = min(-depth.sell_orders[ask_px], VFE_AGGR_SIZE, rem_buy)
            orders.append(Order(VFE_PRODUCT, ask_px, vol))
            rem_buy -= vol

        for bid_px in sorted(depth.buy_orders, reverse=True):
            if bid_px < fv + VFE_AGGR_EDGE:
                break
            if rem_sell <= 0:
                break
            vol = min(depth.buy_orders[bid_px], VFE_AGGR_SIZE, rem_sell)
            orders.append(Order(VFE_PRODUCT, bid_px, -vol))
            rem_sell -= vol

        # Mark 67 exploitation — fixed sizes (no scaling to avoid variance)
        if mark67_bought and rem_sell > 0 and ba is not None:
            vol = min(M67_SELL_SIZE, rem_sell)
            orders.append(Order(VFE_PRODUCT, ba, -vol))
            rem_sell -= vol
        elif feeder_sold and rem_sell > 0 and ba is not None:
            vol = min(FEED_SELL_SIZE, rem_sell)
            orders.append(Order(VFE_PRODUCT, ba, -vol))
            rem_sell -= vol

        # Always-on small sell at best_ask — M67 may buy on any tick
        if rem_sell > 0 and ba is not None:
            vol = min(M67_ALWAYS_SELL, rem_sell)
            orders.append(Order(VFE_PRODUCT, ba, -vol))
            rem_sell -= vol

        # Symmetric passive MM with inventory skew (same as v2)
        skew = int(position / VFE_LIMIT * VFE_MM_HALF * 2)
        my_bid = int(fv) - VFE_MM_HALF - skew
        my_ask = int(fv) + VFE_MM_HALF - skew

        if position > VFE_SKEW_THR and ba is not None:
            my_ask = min(my_ask, ba)
        if position < -VFE_SKEW_THR and bb is not None:
            my_bid = max(my_bid, bb)

        if rem_buy > 0 and (ba is None or my_bid < ba):
            orders.append(Order(VFE_PRODUCT, my_bid, min(VFE_MM_SIZE, rem_buy)))
        if rem_sell > 0 and (bb is None or my_ask > bb):
            orders.append(Order(VFE_PRODUCT, my_ask, -min(VFE_MM_SIZE, rem_sell)))

        return orders, fv

    # ── Main ────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result = {}

        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        hp_fv  = saved.get("hp_fv",  HP_FV_INIT)
        vfe_fv = saved.get("vfe_fv", VFE_FV_INIT)

        vfe_trades = state.market_trades.get(VFE_PRODUCT, [])
        hp_trades  = state.market_trades.get(HP_PRODUCT, [])

        if HP_PRODUCT in state.order_depths:
            depth    = state.order_depths[HP_PRODUCT]
            position = state.position.get(HP_PRODUCT, 0)
            hp_orders, hp_fv = self._get_hp_orders(depth, position, hp_trades, hp_fv)
            result[HP_PRODUCT] = hp_orders

        if VFE_PRODUCT in state.order_depths:
            depth    = state.order_depths[VFE_PRODUCT]
            position = state.position.get(VFE_PRODUCT, 0)
            vfe_orders, vfe_fv = self._get_vfe_orders(depth, position, vfe_trades, vfe_fv)
            result[VFE_PRODUCT] = vfe_orders

        new_state = json.dumps({"hp_fv": hp_fv, "vfe_fv": vfe_fv})
        return result, 0, new_state
