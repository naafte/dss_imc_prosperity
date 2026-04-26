from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional, Tuple
import math
import jsonpickle

# ─── Constants ────────────────────────────────────────────────────────────────

UNDERLYING    = "VELVETFRUIT_EXTRACT"
VEV_STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_SYM       = {k: f"VEV_{k}" for k in VEV_STRIKES}

LIM_UNDER     = 200
LIM_VOUCHER   = 300

TTE_AT_START  = 5           # days to expiry at start of Round 3 actual submission
TICKS_PER_DAY = 1_000_000   # timestamps run 0 … 999_900 per day

S_FALLBACK    = 5250.0      # fallback spot if order book is empty
DEFAULT_SIGMA = 0.0127      # Fallback if IV cannot be calculated

# ── Trading Parameters ───────────────────────────────────────────────────────
SPREAD_PAIRS: List[Tuple[int, int]] = [
    (5000, 5100), (5100, 5200), (5200, 5300), (5300, 5400), (5400, 5500),
    (5000, 5200), (5100, 5300), (5200, 5400), (5300, 5500),
]
MIN_EDGE      = 1.0   # Minimum edge in ticks to consider a trade
CONVICTION_MULTIPLIER = 2.0  # Scales volume based on edge: volume = edge * multiplier

SKIP_MM  = {4000, 4500, 6000, 6500}
MM_HALF  = 1    
ARB_EDGE = 1.0  

BFLY_SPACING: Dict[int, int] = {
    4500: 500, 5000: 500, 5100: 100, 5200: 100, 
    5300: 100, 5400: 100, 6000: 500,
}


# ─── Black-Scholes primitives ─────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x * 0.7071067811865476)

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9:
        return max(S - K, 0.0)
    if S <= 0 or K <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    d2  = d1 - sigma * sqT
    return S * _ncdf(d1) - K * _ncdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9:
        return 1.0 if S > K else 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    return _ncdf(d1)

def implied_volatility(S: float, K: float, T: float, market_price: float) -> float:
    """Calculates IV using the bisection method."""
    if market_price <= max(S - K, 0.0) or T <= 1e-9:
        return 0.0001 # Intrinsic or expired
    
    low, high = 0.0001, 0.1 # Daily vol bounds
    for _ in range(20):
        mid = (low + high) / 2
        price = bs_call(S, K, T, mid)
        if price < market_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# ─── Trader ───────────────────────────────────────────────────────────────────

class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):
        try:
            saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        result: Dict[str, List[Order]] = {}

        tte = max(TTE_AT_START - state.timestamp / TICKS_PER_DAY, 1e-9)

        S = self._mid(state, UNDERLYING)
        if S is None:
            S = S_FALLBACK

        # ── 1. Calculate Volatility Structure & Outliers ─────────────────────
        market_ivs = {}
        for k in VEV_STRIKES:
            sym = VEV_SYM[k]
            mid_price = self._mid(state, sym)
            if mid_price is not None:
                market_ivs[k] = implied_volatility(S, k, tte, mid_price)
        
        # Smooth the curve to find the "fair" structure the market is aiming for
        # This highlights the outliers the hints refer to.
        smoothed_ivs = {}
        available_strikes = sorted(market_ivs.keys())
        for i, k in enumerate(available_strikes):
            if i == 0:
                smoothed_ivs[k] = (market_ivs[k] + market_ivs[available_strikes[i+1]]) / 2
            elif i == len(available_strikes) - 1:
                smoothed_ivs[k] = (market_ivs[k] + market_ivs[available_strikes[i-1]]) / 2
            else:
                smoothed_ivs[k] = (market_ivs[available_strikes[i-1]] + market_ivs[k] + market_ivs[available_strikes[i+1]]) / 3

        # ── 2. Spread trading (delta-internalising, conviction scaled) ───────
        spread_orders = self._trade_spreads(state, S, tte, smoothed_ivs)

        # ── 3. Individual option market making ───────────────────────────────
        total_option_delta = 0.0
        indiv_orders: Dict[str, List[Order]] = {}

        for strike in VEV_STRIKES:
            sym   = VEV_SYM[strike]
            sigma = smoothed_ivs.get(strike, DEFAULT_SIGMA)
            
            if sym not in state.order_depths:
                continue
            
            pos   = state.position.get(sym, 0)
            fv    = bs_call(S, strike, tte, sigma)
            delta = bs_delta(S, strike, tte, sigma)
            total_option_delta += pos * delta

            od    = state.order_depths[sym]
            ords: List[Order] = []

            if strike not in SKIP_MM:
                ords += self._market_make(sym, od, fv, pos, LIM_VOUCHER)

            indiv_orders[sym] = ords

        # ── 4. Strike-to-strike arbitrage ────────────────────────────────────
        for sym, ords in self._strike_arb(state).items():
            indiv_orders.setdefault(sym, []).extend(ords)

        # ── Merge spread and individual orders ───────────────────────────────
        all_option_orders: Dict[str, List[Order]] = {}
        for sym in set(list(spread_orders) + list(indiv_orders)):
            combined = spread_orders.get(sym, []) + indiv_orders.get(sym, [])
            pos      = state.position.get(sym, 0)
            clipped  = self._clip(combined, pos, LIM_VOUCHER)
            if clipped:
                all_option_orders[sym] = clipped

        result.update(all_option_orders)

        # ── 5. Delta hedge residual with VELVETFRUIT_EXTRACT ─────────────────
        result[UNDERLYING] = self._delta_hedge(state, S, total_option_delta)

        traderData = jsonpickle.encode(saved)
        return result, 0, traderData

    # ─── Spread trading ───────────────────────────────────────────────────────

    def _trade_spreads(self, state: TradingState, S: float,
                        tte: float, smoothed_ivs: Dict[int, float]) -> Dict[str, List[Order]]:
        orders: Dict[str, List[Order]] = {}

        for k_long, k_short in SPREAD_PAIRS:
            s_long  = VEV_SYM[k_long]
            s_short = VEV_SYM[k_short]

            if s_long  not in state.order_depths: continue
            if s_short not in state.order_depths: continue

            od_l = state.order_depths[s_long]
            od_s = state.order_depths[s_short]

            if not od_l.sell_orders or not od_s.buy_orders: continue
            if not od_l.buy_orders  or not od_s.sell_orders: continue

            pos_l = state.position.get(s_long,  0)
            pos_s = state.position.get(s_short, 0)

            # Calculate fair value spread based on the structurally smoothed IV curve
            sigma_long = smoothed_ivs.get(k_long, DEFAULT_SIGMA)
            sigma_short = smoothed_ivs.get(k_short, DEFAULT_SIGMA)
            
            fv_spread = (bs_call(S, k_long,  tte, sigma_long) -
                         bs_call(S, k_short, tte, sigma_short))

            best_ask_l  = min(od_l.sell_orders)
            best_bid_s  = max(od_s.buy_orders)
            best_bid_l  = max(od_l.buy_orders)
            best_ask_s  = min(od_s.sell_orders)

            # ── Buy spread ───────────────────────────────────────────────────
            mkt_cost = best_ask_l - best_bid_s
            buy_edge = fv_spread - mkt_cost
            if buy_edge > MIN_EDGE:
                # Conviction-based sizing: Bigger gap -> Bigger Position
                conviction_size = int(buy_edge * CONVICTION_MULTIPLIER)
                room = min(LIM_VOUCHER - pos_l, LIM_VOUCHER + pos_s, conviction_size)
                if room > 0:
                    orders.setdefault(s_long,  []).append(Order(s_long,  best_ask_l,  room))
                    orders.setdefault(s_short, []).append(Order(s_short, best_bid_s, -room))

            # ── Sell spread ──────────────────────────────────────────────────
            mkt_recv = best_bid_l - best_ask_s
            sell_edge = mkt_recv - fv_spread
            if sell_edge > MIN_EDGE:
                # Conviction-based sizing: Bigger gap -> Bigger Position
                conviction_size = int(sell_edge * CONVICTION_MULTIPLIER)
                room = min(LIM_VOUCHER + pos_l, LIM_VOUCHER - pos_s, conviction_size)
                if room > 0:
                    orders.setdefault(s_long,  []).append(Order(s_long,  best_bid_l, -room))
                    orders.setdefault(s_short, []).append(Order(s_short, best_ask_s,  room))

        return orders

    # ─── Individual option market making ─────────────────────────────────────

    def _market_make(self, sym: str, od: OrderDepth, fv: float,
                      pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        skew      = pos / limit   

        bid_price = int(math.floor(fv - MM_HALF - skew))
        ask_price = int(math.ceil(fv  + MM_HALF - skew))

        if bid_price >= ask_price:
            return orders

        best_ask = min(od.sell_orders) if od.sell_orders else ask_price + 2
        best_bid = max(od.buy_orders) if od.buy_orders else bid_price - 2

        # Scale MM quote sizes based on how far market deviates from our FV
        buy_edge = fv - best_ask
        sell_edge = best_bid - fv
        
        base_size = 3
        dynamic_bid_size = int(base_size + max(0, buy_edge * CONVICTION_MULTIPLIER))
        dynamic_ask_size = int(base_size + max(0, sell_edge * CONVICTION_MULTIPLIER))

        buy_room  = limit - pos
        sell_room = limit + pos

        if buy_room  > 0:
            orders.append(Order(sym, bid_price,  min(dynamic_bid_size, buy_room)))
        if sell_room > 0:
            orders.append(Order(sym, ask_price, -min(dynamic_ask_size, sell_room)))

        return orders

    # ─── Strike-to-strike arbitrage ───────────────────────────────────────────
    def _strike_arb(self, state: TradingState) -> Dict[str, List[Order]]:
        # Function logic remains identical to your original implementation
        arb: Dict[str, List[Order]] = {}
        bb: Dict[int, int] = {}
        ba: Dict[int, int] = {}
        pm: Dict[int, int] = {}
        for k in VEV_STRIKES:
            sym = VEV_SYM[k]
            if sym not in state.order_depths: continue
            od = state.order_depths[sym]
            if od.buy_orders and od.sell_orders:
                bb[k] = max(od.buy_orders)
                ba[k] = min(od.sell_orders)
            pm[k] = state.position.get(sym, 0)

        avail = sorted(set(bb) & set(ba))

        for i in range(len(avail) - 1):
            k1, k2 = avail[i], avail[i + 1]
            profit = bb[k2] - ba[k1]
            if profit <= ARB_EDGE: continue
            room = min(LIM_VOUCHER - pm.get(k1, 0), LIM_VOUCHER + pm.get(k2, 0), 5)
            if room <= 0: continue
            s1, s2 = VEV_SYM[k1], VEV_SYM[k2]
            arb.setdefault(s1, []).append(Order(s1, ba[k1],  room))
            arb.setdefault(s2, []).append(Order(s2, bb[k2], -room))

        for k2, ds in BFLY_SPACING.items():
            k1, k3 = k2 - ds, k2 + ds
            if any(k not in bb for k in [k1, k2, k3]): continue
            if any(k not in ba for k in [k1, k2, k3]): continue
            cost = ba[k1] + ba[k3] - 2 * bb[k2]
            if cost >= -ARB_EDGE: continue
            room = min(LIM_VOUCHER - pm.get(k1, 0), LIM_VOUCHER - pm.get(k3, 0), (LIM_VOUCHER + pm.get(k2, 0)) // 2, 3)
            if room <= 0: continue
            s1, s2, s3 = VEV_SYM[k1], VEV_SYM[k2], VEV_SYM[k3]
            arb.setdefault(s1, []).append(Order(s1, ba[k1],       room))
            arb.setdefault(s2, []).append(Order(s2, bb[k2], -2 * room))
            arb.setdefault(s3, []).append(Order(s3, ba[k3],       room))
        return arb

    # ─── Delta hedge (residual only) ──────────────────────────────────────────
    def _delta_hedge(self, state: TradingState, S: float, total_option_delta: float) -> List[Order]:
        # Function logic remains identical to your original implementation
        od  = state.order_depths.get(UNDERLYING)
        if od is None: return []
        pos = state.position.get(UNDERLYING, 0)

        target    = int(round(max(-LIM_UNDER, min(LIM_UNDER, -total_option_delta))))
        delta_qty = target - pos
        if abs(delta_qty) < 1: return []

        orders: List[Order] = []
        if delta_qty > 0:
            rem = delta_qty
            for ask in sorted(od.sell_orders):
                if rem <= 0: break
                vol = min(rem, -od.sell_orders[ask])
                if vol > 0:
                    orders.append(Order(UNDERLYING, ask, vol))
                    rem -= vol
            if rem > 0:
                best_ask = min(od.sell_orders) if od.sell_orders else int(S) + 1
                orders.append(Order(UNDERLYING, best_ask - 1, rem))
        else:
            rem = -delta_qty
            for bid in sorted(od.buy_orders, reverse=True):
                if rem <= 0: break
                vol = min(rem, od.buy_orders[bid])
                if vol > 0:
                    orders.append(Order(UNDERLYING, bid, -vol))
                    rem -= vol
            if rem > 0:
                best_bid = max(od.buy_orders) if od.buy_orders else int(S) - 1
                orders.append(Order(UNDERLYING, best_bid + 1, -rem))

        return self._clip(orders, pos, LIM_UNDER)

    # ─── Utilities ────────────────────────────────────────────────────────────
    def _mid(self, state: TradingState, sym: str) -> Optional[float]:
        od = state.order_depths.get(sym)
        if not od or not od.buy_orders or not od.sell_orders: return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

    def _clip(self, orders: List[Order], pos: int, limit: int) -> List[Order]:
        out: List[Order] = []
        cur = pos
        for o in orders:
            if o.quantity > 0:
                room = limit - cur
                if room <= 0: continue
                qty = min(o.quantity, room)
            else:
                room = limit + cur
                if room <= 0: continue
                qty = max(o.quantity, -room)
            if qty != 0:
                out.append(Order(o.symbol, o.price, qty))
                cur += qty
        return out