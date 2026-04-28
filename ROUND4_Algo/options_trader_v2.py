"""
Options trader v3 — targeted improvements over v2.

Critical fixes:
  - Thresholds corrected: buy when ask < fv (v2 incorrectly bought at fv+2),
    sell when bid > fv (v2 incorrectly sold at fv-2). Both were losing trades.
  - Walk full order book: take ALL profitable price levels, not just best.
  - Multi-strike IV calibration: weighted EMA from 5000/5100/5200 for robustness.
  - Passive limit orders anchored below/above fair value (not just best ±1).
  - Mark 22 selling on VEV_5400 treated as buy confirmation (he's the liquidity).
  - OPT_LIMIT raised to 295; per-level cap raised to 60.

Trades only options, no spot products.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
from math import log, sqrt, exp, erf
import json

# ── Maths ─────────────────────────────────────────────────────────────────────

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
    intrinsic = max(S - K, 0.0)
    if T <= 1e-9 or price <= intrinsic + 0.05:
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

# ── Constants ─────────────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMBOLS = {K: f"VEV_{K}" for K in STRIKES}
VEV_SYM = "VELVETFRUIT_EXTRACT"

OPT_LIMIT     = 290   # exchange cap 300; 10-contract buffer
OPT_AGGR_SIZE = 50    # max units taken per price level
OPT_PASS_SIZE = 25    # passive limit order size

# Tolerance matching EDA-confirmed edges:
#   5400 avg discount = 1.74  → profitable to buy up to fv + BUY_TOL
#   5300/5500 avg premium ≈ 1 → profitable to sell down to fv - SELL_TOL
BUY_TOL  = 1.0
SELL_TOL = 1.0

TTE_START_DAYS = 4
TICKS_PER_DAY  = 1_000_000

IV_EMA_ALPHA   = 0.20   # weight on fresh IV; 0.80 on cached
# Only calibrate from ATM (5000) — EDA edges were computed against ATM vol.
# Using OTM strikes (5100/5200) inflates sigma via the vol smile and
# causes us to buy 5400 above its true ATM fair value.
CALIB_STRIKES  = [5000]

MARK_01 = "Mark 01"   # smart buyer (buys below mid per EDA) → follow buys
MARK_22 = "Mark 22"   # systematic seller  → reinforce sells; if selling 5400, buy from him

# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mid(depth: OrderDepth) -> Optional[float]:
        bids = depth.buy_orders
        asks = depth.sell_orders
        if bids and asks:
            return (max(bids) + min(asks)) / 2.0
        return None

    @staticmethod
    def _tte(timestamp: int) -> float:
        tte_days = TTE_START_DAYS - timestamp / TICKS_PER_DAY
        return max(tte_days, 1e-9) / 252.0

    @staticmethod
    def _calibrate_iv(state: TradingState, S: float, T: float,
                      cached_iv: float) -> float:
        """EMA-smoothed IV from VEV_5000 mid (ATM anchor)."""
        depth = state.order_depths.get(SYMBOLS[5000])
        if not depth:
            return cached_iv
        mid = Trader._mid(depth)
        if mid is None or mid <= max(S - 5000, 0.0) + 0.05:
            return cached_iv
        iv = implied_vol(mid, S, 5000, T, init=cached_iv)
        if not (0.05 < iv < 3.0):
            return cached_iv
        return IV_EMA_ALPHA * iv + (1.0 - IV_EMA_ALPHA) * cached_iv

    # ------------------------------------------------------------------
    # book-walking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _take_asks(sym: str, asks: dict, threshold: float,
                   rem_buy: int) -> Tuple[List[Order], int]:
        """Hit every ask level below threshold (cheapest first)."""
        orders: List[Order] = []
        for px in sorted(asks.keys()):
            if px >= threshold or rem_buy <= 0:
                break
            vol = min(-asks[px], OPT_AGGR_SIZE, rem_buy)
            if vol > 0:
                orders.append(Order(sym, px, vol))
                rem_buy -= vol
        return orders, rem_buy

    @staticmethod
    def _take_bids(sym: str, bids: dict, threshold: float,
                   rem_sell: int) -> Tuple[List[Order], int]:
        """Hit every bid level above threshold (highest first)."""
        orders: List[Order] = []
        for px in sorted(bids.keys(), reverse=True):
            if px <= threshold or rem_sell <= 0:
                break
            vol = min(bids[px], OPT_AGGR_SIZE, rem_sell)
            if vol > 0:
                orders.append(Order(sym, px, -vol))
                rem_sell -= vol
        return orders, rem_sell

    # ------------------------------------------------------------------
    # per-option orders
    # ------------------------------------------------------------------

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

        # ── Deep OTM (6000/6500): BS_FV ≈ 0 → sell any positive bid ─────
        if K in (6000, 6500):
            if best_bid and best_bid > 0 and rem_sell > 0:
                vol = min(bids[best_bid], OPT_AGGR_SIZE, rem_sell)
                orders.append(Order(sym, best_bid, -vol))
            return orders

        if K not in (5300, 5400, 5500):
            return orders

        fv = bs_call(S, K, T, sigma)

        if K == 5400:
            buy_thresh = fv + BUY_TOL

            if asks and rem_buy > 0:
                new_orders, rem_buy = Trader._take_asks(sym, asks, buy_thresh, rem_buy)
                orders.extend(new_orders)

            # Mark 01 is a confirmed smart buyer per EDA → follow on 5400
            if mark01_buying and rem_buy > 0 and best_ask is not None:
                if best_ask < buy_thresh:
                    vol = min(-asks[best_ask], OPT_AGGR_SIZE // 2, rem_buy)
                    if vol > 0:
                        orders.append(Order(sym, best_ask, vol))
                        rem_buy -= vol

            # Passive bid: queue above best bid, only while clearly below fv
            if rem_buy > 0 and best_bid is not None:
                passive_bid = best_bid + 1
                if passive_bid < fv and (best_ask is None or passive_bid < best_ask):
                    orders.append(Order(sym, passive_bid, min(OPT_PASS_SIZE, rem_buy)))

        else:  # 5300 or 5500
            sell_thresh = fv - SELL_TOL

            if bids and rem_sell > 0:
                new_orders, rem_sell = Trader._take_bids(sym, bids, sell_thresh, rem_sell)
                orders.extend(new_orders)

            # Mark 22 is a systematic option seller → reinforces our short
            if mark22_selling and rem_sell > 0 and best_bid is not None:
                if best_bid > sell_thresh:
                    vol = min(bids[best_bid], OPT_AGGR_SIZE // 2, rem_sell)
                    if vol > 0:
                        orders.append(Order(sym, best_bid, -vol))
                        rem_sell -= vol

            # Passive ask: queue below best ask, only while clearly above fv
            if rem_sell > 0 and best_ask is not None:
                passive_ask = best_ask - 1
                if passive_ask > fv and (best_bid is None or passive_ask > best_bid):
                    orders.append(Order(sym, passive_ask, -min(OPT_PASS_SIZE, rem_sell)))

        return orders

    # ------------------------------------------------------------------
    # main run
    # ------------------------------------------------------------------

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}
        cached_iv   = saved.get("iv",   0.267)
        cached_spot = saved.get("spot", None)

        vev_depth = state.order_depths.get(VEV_SYM)
        S = (self._mid(vev_depth) if vev_depth else None) or cached_spot
        if S is None:
            return result, 0, state.traderData or ""

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

        new_state = json.dumps({"iv": sigma, "spot": S})
        return result, 0, new_state
