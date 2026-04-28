"""
final_submission.py — HYDROGEL_PACK + VELVETFRUIT_EXTRACT (spot) + VEV options.

Combines no_options_trader_v2.py and options_trader.py into one Trader class.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
from math import log, sqrt, exp, erf
import json

# ── Maths (Black-Scholes) ──────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / 1.4142135623730951))

def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) * 0.3989422804014327

def bs_call(S: float, K: int, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 1e-9:
        return max(S - K, 0.0)
    sqrtT = sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def bs_vega(S: float, K: int, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 1e-9:
        return 1e-9
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    return S * sqrt(T) * _norm_pdf(d1)

def implied_vol(price: float, S: float, K: int, T: float, init: float = 0.267) -> float:
    if T <= 1e-9 or price <= max(S - K, 0.0):
        return init
    sigma = init
    for _ in range(30):
        pv = bs_call(S, K, T, sigma)
        v  = bs_vega(S, K, T, sigma)
        if v < 1e-8:
            break
        step = (pv - price) / v
        sigma -= step
        sigma = max(0.01, min(sigma, 8.0))
        if abs(step) < 1e-6:
            break
    return sigma

# ── Spot helpers ───────────────────────────────────────────────────────────────

def _best_bid(depth: OrderDepth) -> Optional[int]:
    return max(depth.buy_orders) if depth.buy_orders else None

def _best_ask(depth: OrderDepth) -> Optional[int]:
    return min(depth.sell_orders) if depth.sell_orders else None

def _mid(depth: OrderDepth) -> Optional[float]:
    b, a = _best_bid(depth), _best_ask(depth)
    return (b + a) / 2.0 if (b is not None and a is not None) else None

def _ema(prev: float, obs: float, alpha: float) -> float:
    return alpha * obs + (1.0 - alpha) * prev

# ── Spot constants ─────────────────────────────────────────────────────────────

HP_LIMIT  = 200
VFE_LIMIT = 200

HP_PRODUCT    = "HYDROGEL_PACK"
HP_FV_INIT    = 10_000
HP_FV_ALPHA   = 0.20
HP_AGGR_EDGE  = 3
HP_AGGR_SIZE  = 40
HP_MM_SIZE    = 15
HP_SKEW_THR   = 60

VFE_PRODUCT   = "VELVETFRUIT_EXTRACT"
VFE_FV_INIT   = 5_250
VFE_FV_ALPHA  = 0.15
VFE_AGGR_EDGE = 15
VFE_AGGR_SIZE = 40
VFE_MM_HALF   = 8
VFE_MM_SIZE   = 15
VFE_SKEW_THR  = 60

MARK_14 = "Mark 14"
MARK_38 = "Mark 38"
MARK_67 = "Mark 67"
MARK_22 = "Mark 22"
MARK_49 = "Mark 49"

M67_SELL_SIZE  = 25
FEED_SELL_SIZE = 15

# ── Options constants ──────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMBOLS = {K: f"VEV_{K}" for K in STRIKES}

OPT_LIMIT     = 250
OPT_AGGR_SIZE = 30

TTE_START_DAYS = 4
TICKS_PER_DAY  = 1_000_000

MARK_01 = "Mark 01"

# ── Trader ─────────────────────────────────────────────────────────────────────

class Trader:

    # ── HGP ────────────────────────────────────────────────────────────────────

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

    def _get_hp_orders(self, depth: OrderDepth, position: int,
                       hp_trades: list, cached_fv: float):
        orders: List[Order] = []
        bb = _best_bid(depth)
        ba = _best_ask(depth)
        book_mid = (bb + ba) / 2.0 if (bb is not None and ba is not None) else cached_fv
        fv = self._hp_fv_update(hp_trades, book_mid, cached_fv)

        rem_buy  = HP_LIMIT - position
        rem_sell = HP_LIMIT + position

        for ask_px in sorted(depth.sell_orders):
            if ask_px > fv - HP_AGGR_EDGE or rem_buy <= 0:
                break
            vol = min(-depth.sell_orders[ask_px], HP_AGGR_SIZE, rem_buy)
            orders.append(Order(HP_PRODUCT, ask_px, vol))
            rem_buy -= vol

        for bid_px in sorted(depth.buy_orders, reverse=True):
            if bid_px < fv + HP_AGGR_EDGE or rem_sell <= 0:
                break
            vol = min(depth.buy_orders[bid_px], HP_AGGR_SIZE, rem_sell)
            orders.append(Order(HP_PRODUCT, bid_px, -vol))
            rem_sell -= vol

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

    # ── VFE spot ───────────────────────────────────────────────────────────────

    @staticmethod
    def _vfe_signals(vfe_trades: list):
        mark67_bought = any(t.buyer == MARK_67 for t in vfe_trades)
        feeder_sold   = any(t.seller in (MARK_22, MARK_49) for t in vfe_trades)
        return mark67_bought, feeder_sold

    def _get_vfe_orders(self, depth: OrderDepth, position: int,
                        vfe_trades: list, cached_fv: float):
        orders: List[Order] = []
        bb = _best_bid(depth)
        ba = _best_ask(depth)
        mid = (bb + ba) / 2.0 if (bb is not None and ba is not None) else None
        fv = _ema(cached_fv, mid, VFE_FV_ALPHA) if mid is not None else cached_fv

        rem_buy  = VFE_LIMIT - position
        rem_sell = VFE_LIMIT + position

        mark67_bought, feeder_sold = self._vfe_signals(vfe_trades)

        for ask_px in sorted(depth.sell_orders):
            if ask_px > fv - VFE_AGGR_EDGE or rem_buy <= 0:
                break
            vol = min(-depth.sell_orders[ask_px], VFE_AGGR_SIZE, rem_buy)
            orders.append(Order(VFE_PRODUCT, ask_px, vol))
            rem_buy -= vol

        for bid_px in sorted(depth.buy_orders, reverse=True):
            if bid_px < fv + VFE_AGGR_EDGE or rem_sell <= 0:
                break
            vol = min(depth.buy_orders[bid_px], VFE_AGGR_SIZE, rem_sell)
            orders.append(Order(VFE_PRODUCT, bid_px, -vol))
            rem_sell -= vol

        if mark67_bought and rem_sell > 0 and ba is not None:
            vol = min(M67_SELL_SIZE, rem_sell)
            orders.append(Order(VFE_PRODUCT, ba, -vol))
            rem_sell -= vol
        elif feeder_sold and rem_sell > 0 and ba is not None:
            vol = min(FEED_SELL_SIZE, rem_sell)
            orders.append(Order(VFE_PRODUCT, ba, -vol))
            rem_sell -= vol

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

    # ── Options ────────────────────────────────────────────────────────────────

    @staticmethod
    def _tte(timestamp: int) -> float:
        tte_days = TTE_START_DAYS - timestamp / TICKS_PER_DAY
        return max(tte_days, 1e-9) / 252.0

    @staticmethod
    def _calibrate_iv(state: TradingState, S: float, T: float, cached_iv: float) -> float:
        depth = state.order_depths.get(SYMBOLS[5000])
        if not depth:
            return cached_iv
        bids = depth.buy_orders
        asks = depth.sell_orders
        if not bids or not asks:
            return cached_iv
        mid_5k = (max(bids) + min(asks)) / 2.0
        if mid_5k <= max(S - 5000, 0.0):
            return cached_iv
        iv = implied_vol(mid_5k, S, 5000, T, init=cached_iv)
        if 0.05 < iv < 3.0:
            return iv
        return cached_iv

    @staticmethod
    def _orders_for_strike(
        K: int,
        depth: OrderDepth,
        position: int,
        S: float,
        T: float,
        sigma: float,
        mark01_buying: bool,
        mark22_selling: bool,
    ) -> List[Order]:
        sym      = SYMBOLS[K]
        orders: List[Order] = []
        bids     = depth.buy_orders
        asks     = depth.sell_orders
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None
        rem_buy  = OPT_LIMIT - position
        rem_sell = OPT_LIMIT + position

        if K in (6000, 6500):
            if best_bid and best_bid > 0 and rem_sell > 0:
                vol = min(bids[best_bid], OPT_AGGR_SIZE, rem_sell)
                orders.append(Order(sym, best_bid, -vol))
            return orders

        if K not in (5300, 5400, 5500):
            return orders

        fv = bs_call(S, K, T, sigma)

        if K == 5400:
            if best_ask is not None and best_ask < fv + 1.0 and rem_buy > 0:
                vol = min(-asks[best_ask], OPT_AGGR_SIZE, rem_buy)
                orders.append(Order(sym, best_ask, vol))
                rem_buy -= vol
            if mark01_buying and rem_buy > 0 and best_ask is not None:
                vol = min(-asks[best_ask], OPT_AGGR_SIZE // 2, rem_buy)
                orders.append(Order(sym, best_ask, vol))
        else:
            if best_bid is not None and best_bid > fv - 1.0 and rem_sell > 0:
                vol = min(bids[best_bid], OPT_AGGR_SIZE, rem_sell)
                orders.append(Order(sym, best_bid, -vol))
                rem_sell -= vol
            if mark22_selling and rem_sell > 0 and best_bid is not None:
                vol = min(bids[best_bid], OPT_AGGR_SIZE // 2, rem_sell)
                orders.append(Order(sym, best_bid, -vol))

        return orders

    # ── Main ───────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        hp_fv     = saved.get("hp_fv",  HP_FV_INIT)
        vfe_fv    = saved.get("vfe_fv", VFE_FV_INIT)
        cached_iv = saved.get("iv",     0.267)

        vfe_trades = state.market_trades.get(VFE_PRODUCT, [])
        hp_trades  = state.market_trades.get(HP_PRODUCT, [])

        # ── Spot: HYDROGEL_PACK ───────────────────────────────────────────
        if HP_PRODUCT in state.order_depths:
            depth    = state.order_depths[HP_PRODUCT]
            position = state.position.get(HP_PRODUCT, 0)
            hp_orders, hp_fv = self._get_hp_orders(depth, position, hp_trades, hp_fv)
            result[HP_PRODUCT] = hp_orders

        # ── Spot: VELVETFRUIT_EXTRACT ─────────────────────────────────────
        S = None
        if VFE_PRODUCT in state.order_depths:
            depth    = state.order_depths[VFE_PRODUCT]
            position = state.position.get(VFE_PRODUCT, 0)
            S = _mid(depth)
            vfe_orders, vfe_fv = self._get_vfe_orders(depth, position, vfe_trades, vfe_fv)
            result[VFE_PRODUCT] = vfe_orders

        # ── Options: VEV_* ────────────────────────────────────────────────
        if S is not None:
            T     = self._tte(state.timestamp)
            sigma = self._calibrate_iv(state, S, T, cached_iv)

            mark01_buying_sym:  set = set()
            mark22_selling_sym: set = set()
            for sym, trades in state.market_trades.items():
                if not sym.startswith("VEV_"):
                    continue
                for t in trades:
                    if t.buyer  == MARK_01:
                        mark01_buying_sym.add(sym)
                    if t.seller == MARK_22:
                        mark22_selling_sym.add(sym)

            for K in STRIKES:
                sym = SYMBOLS[K]
                if sym not in state.order_depths:
                    continue
                depth    = state.order_depths[sym]
                position = state.position.get(sym, 0)
                orders   = self._orders_for_strike(
                    K, depth, position, S, T, sigma,
                    mark01_buying  = sym in mark01_buying_sym,
                    mark22_selling = sym in mark22_selling_sym,
                )
                if orders:
                    result[sym] = orders

            cached_iv = sigma

        new_state = json.dumps({"hp_fv": hp_fv, "vfe_fv": vfe_fv, "iv": cached_iv})
        return result, 0, new_state
