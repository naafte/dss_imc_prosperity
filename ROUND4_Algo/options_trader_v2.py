"""
Options trader v2 — maximises PnL on EDA-confirmed edges.

Improvements over v1:
  - Position limit raised to 290 (exchange cap 300).
  - Wider take thresholds (fv ± 2) capture more fills without chasing.
  - Passive competitive bids/asks posted after aggressive takes so we also
    capture flow that comes in between best_bid and best_ask.
  - IV smoothed with EMA (α=0.15) to dampen single-tick VEV_5000 spikes.
  - Spot price cached so ticks with a one-sided VEV book still trade.
  - Larger per-order aggr size (50) fills position limits faster.

Trades only the 5 EDA-confirmed strikes:
  VEV_5400         → BUY  (ask < BS_FV 83% of time, avg discount ~1.74)
  VEV_5300/5500    → SELL (bid > BS_FV 55% of time)
  VEV_6000/6500    → SELL (BS_FV ≈ 0, sell any positive bid)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
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

# ── Constants ─────────────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMBOLS = {K: f"VEV_{K}" for K in STRIKES}
VEV_SYM = "VELVETFRUIT_EXTRACT"

OPT_LIMIT     = 290   # exchange cap 300; 10-contract safety buffer
OPT_AGGR_SIZE = 50    # aggressive take size per order
OPT_PASS_SIZE = 20    # passive limit order size

TTE_START_DAYS = 4
TICKS_PER_DAY  = 1_000_000

IV_EMA_ALPHA = 0.15   # weight on new IV observation; 0.85 on cached

MARK_01 = "Mark 01"   # smart buyer per EDA → follow on VEV_5400
MARK_22 = "Mark 22"   # systematic seller   → reinforce sell on VEV_5300/5500

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
        """EMA-smoothed implied vol from VEV_5000 mid."""
        depth = state.order_depths.get(SYMBOLS[5000])
        if depth is None:
            return cached_iv
        mid_5k = Trader._mid(depth)
        if mid_5k is None or mid_5k <= 0:
            return cached_iv
        iv = implied_vol(mid_5k, S, 5000, T, init=cached_iv)
        if not (0.10 < iv < 1.50):   # tighter sanity range vs v1 (0.05–3.0)
            return cached_iv
        return IV_EMA_ALPHA * iv + (1.0 - IV_EMA_ALPHA) * cached_iv

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

        # ── Deep OTM (6000/6500): BS ≈ 0 → sell any positive bid ─────────
        if K in (6000, 6500):
            if best_bid and best_bid > 0 and rem_sell > 0:
                vol = min(bids[best_bid], OPT_AGGR_SIZE, rem_sell)
                orders.append(Order(sym, best_bid, -vol))
            return orders

        # ── Only the three confirmed EDA-edge strikes ─────────────────────
        if K not in (5300, 5400, 5500):
            return orders

        fv = bs_call(S, K, T, sigma)

        if K == 5400:
            # ── BUY edge: ask < BS_FV 83% of time, avg discount ~1.74 ────
            # 1. Aggressive take at best ask
            if best_ask is not None and best_ask < fv + 2.0 and rem_buy > 0:
                vol = min(-asks[best_ask], OPT_AGGR_SIZE, rem_buy)
                orders.append(Order(sym, best_ask, vol))
                rem_buy -= vol

            # 2. Follow Mark 01 (smart buyer per EDA, buying below mid)
            if mark01_buying and rem_buy > 0 and best_ask is not None:
                vol = min(-asks[best_ask], OPT_AGGR_SIZE // 2, rem_buy)
                orders.append(Order(sym, best_ask, vol))
                rem_buy -= vol

            # 3. Passive competitive bid: outbid current best to capture
            #    sellers who come in between best_bid and best_ask
            if rem_buy > 0 and best_bid is not None:
                passive_bid = best_bid + 1
                if best_ask is None or passive_bid < best_ask:
                    orders.append(Order(sym, passive_bid,
                                        min(OPT_PASS_SIZE, rem_buy)))

        else:  # 5300 or 5500
            # ── SELL edge: bid > BS_FV 55% of time ───────────────────────
            # 1. Aggressive take at best bid
            if best_bid is not None and best_bid > fv - 2.0 and rem_sell > 0:
                vol = min(bids[best_bid], OPT_AGGR_SIZE, rem_sell)
                orders.append(Order(sym, best_bid, -vol))
                rem_sell -= vol

            # 2. Mark 22 reinforces sell (systematic option seller per EDA)
            if mark22_selling and rem_sell > 0 and best_bid is not None:
                vol = min(bids[best_bid], OPT_AGGR_SIZE // 2, rem_sell)
                orders.append(Order(sym, best_bid, -vol))
                rem_sell -= vol

            # 3. Passive competitive ask: undercut current best ask to
            #    capture buyers who come in between best_bid and best_ask
            if rem_sell > 0 and best_ask is not None:
                passive_ask = best_ask - 1
                if best_bid is None or passive_ask > best_bid:
                    orders.append(Order(sym, passive_ask,
                                        -min(OPT_PASS_SIZE, rem_sell)))

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

        # Spot price with fallback to last known
        vev_depth = state.order_depths.get(VEV_SYM)
        S = (self._mid(vev_depth) if vev_depth else None) or cached_spot
        if S is None:
            return result, 0, state.traderData or ""

        T     = self._tte(state.timestamp)
        sigma = self._calibrate_iv(state, S, T, cached_iv)

        # Counterparty signals (per option symbol)
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
