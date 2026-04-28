"""
no_options_trader_v2.py — HYDROGEL_PACK + VELVETFRUIT_EXTRACT only (no options).

Key improvements over no_options_trader.py:
  - Position limits raised to 200 (was 60) — exchange cap.
  - Larger aggr/MM sizes to fill limits faster.
  - VFE fair value tracked via EMA across ticks (persisted in traderData).
  - HGP fair value anchored via EMA of Mark 14/38 trade prices.
  - Smarter Mark 67 exploitation: post aggressive sell into his buying cycles.
  - Multi-level book sweep: take ALL mispriced levels, not just best.
  - Inventory skew applied more aggressively to mean-revert toward flat.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Optional
import json

# ── Position limits (exchange caps) ───────────────────────────────────────────
HP_LIMIT  = 200
VFE_LIMIT = 200

# ── HYDROGEL_PACK parameters ───────────────────────────────────────────────────
HP_PRODUCT    = "HYDROGEL_PACK"
HP_FV_INIT    = 10_000   # fallback if no Mark 14/38 trades yet
HP_FV_ALPHA   = 0.20     # EMA weight on new MM-trade observation
HP_AGGR_EDGE  = 3        # take if ask < fv - edge (or bid > fv + edge)
HP_AGGR_SIZE  = 40       # max contracts per aggressive take order
HP_MM_SIZE    = 15       # passive limit order size
HP_SKEW_THR   = 60       # inventory at which we start skewing quotes

# ── VELVETFRUIT_EXTRACT parameters ────────────────────────────────────────────
VFE_PRODUCT   = "VELVETFRUIT_EXTRACT"
VFE_FV_INIT   = 5_250    # fallback fair value
VFE_FV_ALPHA  = 0.15     # EMA weight on new mid-price observation
VFE_AGGR_EDGE = 15       # take if ask < fv - edge (or bid > fv + edge)
VFE_AGGR_SIZE = 40       # max per aggressive take order
VFE_MM_HALF   = 8        # half-spread for passive quotes around FV
VFE_MM_SIZE   = 15       # passive limit order size
VFE_SKEW_THR  = 60       # inventory threshold for quote skew

# ── Counterparty IDs ───────────────────────────────────────────────────────────
MARK_14 = "Mark 14"   # symmetric HGP market maker (buys ≈ sells ≈ 500/day)
MARK_38 = "Mark 38"   # symmetric HGP market maker
MARK_67 = "Mark 67"   # one-sided VEV buyer (~55 fills/day at mid+1)
MARK_22 = "Mark 22"   # Mark 67's primary VEV supplier
MARK_49 = "Mark 49"   # Mark 67's secondary VEV supplier

# When Mark 67 is buying, post a dedicated sell to capture his demand.
M67_SELL_SIZE  = 25
# When his feeder (22/49) is active, anticipate his next buy — slightly smaller.
FEED_SELL_SIZE = 15


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

    # ── HGP fair value from Mark 14/38 ─────────────────────────────────────────

    def _hp_fv_update(self, hp_trades: list, book_mid: float, cached_fv: float) -> float:
        """EMA of Mark 14/38 trade prices as HGP fair value anchor."""
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
        feeder_sold   = any(t.seller in (MARK_22, MARK_49) for t in vfe_trades)
        return mark67_bought, feeder_sold

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

        # ── Multi-level aggressive sweep ─────────────────────────────────────
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

        # ── Passive market making with inventory skew ─────────────────────────
        skew = 0
        if position > HP_SKEW_THR:
            skew = 1    # widen our bid, tighten our ask → unload longs
        elif position < -HP_SKEW_THR:
            skew = -1   # tighten bid, widen ask → unload shorts

        # Competitive: join/improve best quote, adjusted for inventory
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

        # EMA update of VFE fair value from mid price
        fv = _ema(cached_fv, mid, VFE_FV_ALPHA) if mid is not None else cached_fv

        rem_buy  = VFE_LIMIT - position
        rem_sell = VFE_LIMIT + position

        mark67_bought, feeder_sold = _vfe_signals(vfe_trades)

        # ── Multi-level aggressive take against fair value ────────────────────
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

        # ── Mark 67 exploitation ──────────────────────────────────────────────
        # Mark 67 buys ~55x/day, consistently at or above mid. When he bought
        # last tick, he is very likely to buy again next tick — post a sell
        # at best_ask to capture his demand at an elevated price.
        if mark67_bought and rem_sell > 0 and ba is not None:
            vol = min(M67_SELL_SIZE, rem_sell)
            orders.append(Order(VFE_PRODUCT, ba, -vol))
            rem_sell -= vol
        elif feeder_sold and rem_sell > 0 and ba is not None:
            # Mark 22/49 active → Mark 67 is in a buying cycle even if he
            # didn't fill last tick. Post a slightly smaller sell at ask.
            vol = min(FEED_SELL_SIZE, rem_sell)
            orders.append(Order(VFE_PRODUCT, ba, -vol))
            rem_sell -= vol

        # ── Passive market making with inventory skew ─────────────────────────
        skew = int(position / VFE_LIMIT * VFE_MM_HALF * 2)
        my_bid = int(fv) - VFE_MM_HALF - skew
        my_ask = int(fv) + VFE_MM_HALF - skew

        # When heavily long, push ask toward best_ask to increase fill chance
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


# Re-export the standalone helper so tests can import it.
def _vfe_signals(vfe_trades: list):
    return Trader._vfe_signals(vfe_trades)
