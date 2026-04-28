"""
Options trader for Round 4.
Trades VEV vouchers (VEV_4000 .. VEV_6500) only.

Strategy:
  - Price each option via Black-Scholes, calibrating implied vol from the
    VEV_5000 book mid every tick (most liquid near-ATM option).
  - Key mispricings found in EDA:
      VEV_5400: ask < BS_FV 83% of the time  → BUY
      VEV_5300: bid > BS_FV 55% of the time  → SELL
      VEV_5500: bid > BS_FV 55% of the time  → SELL
      VEV_6000/6500: BS_FV ≈ 0 but mid = 0.5 → SELL when bid > 0
  - Counterparty signals (Round 4 feature):
      Mark 01 in market_trades as buyer → sell that option (he pays above BS FV)
      Mark 14 in market_trades as buyer → follow on 5400 (both buying underpriced)
      Mark 22 in market_trades as seller → reinforces sell signal (he is short vol)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
from math import log, sqrt, exp, erf
import json

# ── Maths ────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / 1.4142135623730951))

def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) * 0.3989422804014327

def bs_call(S: float, K: int, T: float, sigma: float) -> float:
    """Black-Scholes European call price. Returns intrinsic value if T<=0."""
    if T <= 1e-9:
        return max(S - K, 0.0)
    if sigma <= 1e-9:
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

def implied_vol(price: float, S: float, K: int, T: float,
                init: float = 0.267) -> float:
    """Newton-Raphson implied vol. Returns init if it fails to converge."""
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

# ── Constants ────────────────────────────────────────────────────────────────

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMBOLS = {K: f"VEV_{K}" for K in STRIKES}
VEV_SYM = "VELVETFRUIT_EXTRACT"
OPT_LIMIT     = 250   # exchange limit is 300; leave 50 buffer
OPT_AGGR_SIZE = 30    # aggressive take size

# TTE: Round 4 starts at TTE = 4 days.
# Each simulation "day" spans timestamps 0 → 999_900.
# Within the sim, TTE decreases linearly from 4 to ~3 days.
TTE_START_DAYS = 4
TICKS_PER_DAY  = 1_000_000        # 10000 steps of 100

# Counterparties
MARK_01 = "Mark 01"   # smart buyer (buys below mid per EDA) → follow on 5400
MARK_22 = "Mark 22"   # systematic option seller → reinforces sell on 5300/5500

# ── Trader ───────────────────────────────────────────────────────────────────

class Trader:

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_spot(state: TradingState) -> Optional[float]:
        depth = state.order_depths.get(VEV_SYM)
        if depth is None:
            return None
        bids = depth.buy_orders
        asks = depth.sell_orders
        if bids and asks:
            return (max(bids) + min(asks)) / 2.0
        return None

    @staticmethod
    def _tte(timestamp: int) -> float:
        """Time-to-expiry in years, decreasing from 4/252 at t=0."""
        tte_days = TTE_START_DAYS - timestamp / TICKS_PER_DAY
        return max(tte_days, 1e-9) / 252.0

    @staticmethod
    def _get_iv_from_atm(state: TradingState, S: float, T: float,
                         cached_iv: float) -> float:
        """Extract implied vol from VEV_5000 book mid; fall back to cached."""
        depth = state.order_depths.get(SYMBOLS[5000])
        if depth is None:
            return cached_iv
        bids = depth.buy_orders
        asks = depth.sell_orders
        if not bids or not asks:
            return cached_iv
        mid_5k = (max(bids) + min(asks)) / 2.0
        if mid_5k <= 0:
            return cached_iv
        iv = implied_vol(mid_5k, S, 5000, T, init=cached_iv)
        # Sanity: keep in plausible range
        if 0.05 < iv < 3.0:
            return iv
        return cached_iv

    # ------------------------------------------------------------------
    # per-option orders
    # ------------------------------------------------------------------

    def _orders_for_strike(
        self,
        K: int,
        depth: OrderDepth,
        position: int,
        S: float,
        T: float,
        sigma: float,
        mark01_buying: bool,
        mark22_selling: bool,
    ) -> List[Order]:
        sym = SYMBOLS[K]
        orders: List[Order] = []

        bids     = depth.buy_orders
        asks     = depth.sell_orders
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None

        rem_buy  = OPT_LIMIT - position
        rem_sell = OPT_LIMIT + position

        # ── Deep OTM (6000/6500): BS ≈ 0, sell any positive bid ──────────
        if K in (6000, 6500):
            if best_bid and best_bid > 0 and rem_sell > 0:
                vol = min(bids[best_bid], OPT_AGGR_SIZE, rem_sell)
                orders.append(Order(sym, best_bid, -vol))
            return orders

        # ── Only trade the three EDA-confirmed edge strikes ───────────────
        if K not in (5300, 5400, 5500):
            return orders

        fv = bs_call(S, K, T, sigma)

        if K == 5400:
            # EDA: ask < BS_FV 83% of time → buy aggressively when ask is cheap
            if best_ask is not None and best_ask < fv + 1.0 and rem_buy > 0:
                vol = min(-asks[best_ask], OPT_AGGR_SIZE, rem_buy)
                orders.append(Order(sym, best_ask, vol))
                rem_buy -= vol
            # Mark 01 is a smart buyer (EDA: buys below mid) → follow on 5400 only
            if mark01_buying and rem_buy > 0 and best_ask is not None:
                vol = min(-asks[best_ask], OPT_AGGR_SIZE // 2, rem_buy)
                orders.append(Order(sym, best_ask, vol))

        else:  # 5300 or 5500
            # EDA: bid > BS_FV 55% of time → sell aggressively when bid is elevated
            if best_bid is not None and best_bid > fv - 1.0 and rem_sell > 0:
                vol = min(bids[best_bid], OPT_AGGR_SIZE, rem_sell)
                orders.append(Order(sym, best_bid, -vol))
                rem_sell -= vol
            # Mark 22 is a systematic option seller → reinforce sell signal
            if mark22_selling and rem_sell > 0 and best_bid is not None:
                vol = min(bids[best_bid], OPT_AGGR_SIZE // 2, rem_sell)
                orders.append(Order(sym, best_bid, -vol))

        return orders

    # ------------------------------------------------------------------
    # main run
    # ------------------------------------------------------------------

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # Load persisted state
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}
        cached_iv = saved.get("iv", 0.267)

        # Spot price and time-to-expiry
        S = self._get_spot(state)
        if S is None:
            return result, 0, state.traderData or ""

        T = self._tte(state.timestamp)

        # Calibrate IV from ATM option
        sigma = self._get_iv_from_atm(state, S, T, cached_iv)

        # ── Counterparty signals from last tick's trades ──────────────────
        mark01_buying_sym: set  = set()
        mark22_selling_sym: set = set()

        for sym, trades in state.market_trades.items():
            if not sym.startswith("VEV_"):
                continue
            for t in trades:
                if t.buyer == MARK_01:
                    mark01_buying_sym.add(sym)
                if t.seller == MARK_22:
                    mark22_selling_sym.add(sym)

        # ── Generate orders for each option ──────────────────────────────
        for K in STRIKES:
            sym = SYMBOLS[K]
            if sym not in state.order_depths:
                continue
            depth    = state.order_depths[sym]
            position = state.position.get(sym, 0)

            orders = self._orders_for_strike(
                K, depth, position, S, T, sigma,
                mark01_buying  = sym in mark01_buying_sym,
                mark22_selling = sym in mark22_selling_sym,
            )
            if orders:
                result[sym] = orders

        new_state = json.dumps({"iv": sigma})
        return result, 0, new_state
