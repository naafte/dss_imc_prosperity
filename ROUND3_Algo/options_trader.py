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

# Black-Scholes: sigma is daily vol, T is in days.
# Calibrated: VEV_5200 near-ATM at TTE≈8 → price≈95 → sigma_daily ≈ 0.01
SIGMA         = 0.01
TTE_AT_START  = 5           # days to expiry at start of Round 3 actual submission
TICKS_PER_DAY = 1_000_000   # timestamps run 0 … 999_900 per day

S_FALLBACK    = 5250.0      # fallback spot if order book is empty

# ── Spread trading (primary strategy) ────────────────────────────────────────
# Each entry: (buy_strike, sell_strike).  Net delta = delta(buy) - delta(sell),
# which is far smaller than holding either leg alone.
# Focus on the vol-rich near-ATM range; skip deep-ITM (delta≈1, no vol edge).
SPREAD_PAIRS: List[Tuple[int, int]] = [
    (5000, 5100),   # net delta ~0.08
    (5100, 5200),   # net delta ~0.23
    (5200, 5300),   # net delta ~0.33  ← most vega
    (5300, 5400),   # net delta ~0.23
    (5400, 5500),   # net delta ~0.09
    (5000, 5200),   # net delta ~0.32  wider spreads for extra coverage
    (5100, 5300),   # net delta ~0.57
    (5200, 5400),   # net delta ~0.56
    (5300, 5500),   # net delta ~0.32
]
SPREAD_EDGE = 1.0   # min BS edge (price ticks) to aggressively take a spread
SPREAD_SIZE = 10    # max units per spread trade

# ── Individual option strategies (residual / secondary) ───────────────────────
# Skip market-making on deep-ITM strikes; delta≈1 eats the entire hedge budget.
SKIP_MM     = {4000, 4500}
SKIP_VOL    = {4000, 4500, 6000, 6500}   # also skip flat/worthless OTM

MM_HALF     = 2     # passive quote half-spread (price ticks)
MM_SIZE     = 5     # reduced size for individual legs (spreads handle the bulk)
TAKE_EDGE   = 1.5   # min BS mispricing to aggressively take an individual option
VOL_THRESH  = 0.20  # IV must deviate >20% from SIGMA to vol-trade
ARB_EDGE    = 1.0   # min profit per unit for strike arb

# Equal-spacing map for butterfly arb: K2 → ds  (K1=K2-ds, K3=K2+ds)
BFLY_SPACING: Dict[int, int] = {
    4500: 500,  # 4000–4500–5000
    5000: 500,  # 4500–5000–5500
    5100: 100,  # 5000–5100–5200
    5200: 100,  # 5100–5200–5300
    5300: 100,  # 5200–5300–5400
    5400: 100,  # 5300–5400–5500
    6000: 500,  # 5500–6000–6500
}


# ─── Black-Scholes primitives ─────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x * 0.7071067811865476)

def _npdf(x: float) -> float:
    return 0.3989422804014327 * math.exp(-0.5 * x * x)

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

def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 0:
        return 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    return S * _npdf(d1) * sqT

def implied_vol(mkt: float, S: float, K: float, T: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if mkt < intrinsic - 0.5 or mkt <= 0.25 or T < 1e-6:
        return None
    sigma = SIGMA
    for _ in range(60):
        px   = bs_call(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        if vega < 1e-8:
            break
        step  = (px - mkt) / vega
        sigma = max(1e-4, min(sigma - step, 5.0))
        if abs(step) < 1e-7:
            return sigma
    return sigma


# ─── Trader ───────────────────────────────────────────────────────────────────

class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):
        # ── Restore persisted state ──────────────────────────────────────────
        try:
            saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        result: Dict[str, List[Order]] = {}

        # ── Time to expiry (days) ────────────────────────────────────────────
        tte = max(TTE_AT_START - state.timestamp / TICKS_PER_DAY, 1e-9)

        # ── Underlying spot ──────────────────────────────────────────────────
        S = self._mid(state, UNDERLYING)
        if S is None:
            S = S_FALLBACK

        # ── 1. Spread trading (primary, delta-internalising) ─────────────────
        # Trades pairs of options simultaneously so net delta is minimised
        # before touching VELVETFRUIT_EXTRACT.
        spread_orders = self._trade_spreads(state, S, tte)

        # ── 2. Individual option strategies (secondary) ───────────────────────
        total_option_delta = 0.0
        indiv_orders: Dict[str, List[Order]] = {}

        for strike in VEV_STRIKES:
            sym = VEV_SYM[strike]
            if sym not in state.order_depths:
                continue
            pos   = state.position.get(sym, 0)
            fv    = bs_call(S, strike, tte, SIGMA)
            delta = bs_delta(S, strike, tte, SIGMA)
            total_option_delta += pos * delta

            od    = state.order_depths[sym]
            ords: List[Order] = []

            # a) Aggressive: take individual mis-priced options
            ords += self._take_mispriced(sym, od, fv, pos, LIM_VOUCHER)

            # b) Passive: market-make (skip deep-ITM; no vol edge, delta≈1)
            if strike not in SKIP_MM:
                ords += self._market_make(sym, od, fv, pos, LIM_VOUCHER)

            # c) Vol trading: buy cheap / sell rich implied vol
            if strike not in SKIP_VOL:
                ords += self._vol_trade(sym, od, S, strike, tte, pos, LIM_VOUCHER)

            indiv_orders[sym] = ords

        # ── 3. Strike-to-strike arbitrage ────────────────────────────────────
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

        # ── 4. Delta hedge residual with VELVETFRUIT_EXTRACT ─────────────────
        result[UNDERLYING] = self._delta_hedge(state, S, total_option_delta)

        # ── Persist state ────────────────────────────────────────────────────
        traderData = jsonpickle.encode(saved)
        return result, 0, traderData

    # ─── Spread trading ───────────────────────────────────────────────────────

    def _trade_spreads(self, state: TradingState, S: float,
                        tte: float) -> Dict[str, List[Order]]:
        """
        Trade vertical call spreads (buy K_low, sell K_high) as a unit.

        Both legs execute simultaneously against the existing book, so the
        delta of the pair is always much smaller than either leg alone.
        No single-leg risk is accumulated here.

        Edge check:
          Buy spread: pay ask[K_low] - bid[K_high].
          BS fair value: bs_call(K_low) - bs_call(K_high).
          If BS_fv - mkt_cost > SPREAD_EDGE  → buy the spread.
          If mkt_cost - BS_fv > SPREAD_EDGE  → sell the spread.
        """
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

            fv_spread = bs_call(S, k_long, tte, SIGMA) - bs_call(S, k_short, tte, SIGMA)

            best_ask_l  = min(od_l.sell_orders)
            best_bid_s  = max(od_s.buy_orders)
            best_bid_l  = max(od_l.buy_orders)
            best_ask_s  = min(od_s.sell_orders)

            # ── Buy spread ───────────────────────────────────────────────────
            mkt_cost = best_ask_l - best_bid_s
            if fv_spread - mkt_cost > SPREAD_EDGE:
                room = min(
                    LIM_VOUCHER - pos_l,   # room to go longer on K_long
                    LIM_VOUCHER + pos_s,   # room to go shorter on K_short
                    SPREAD_SIZE,
                )
                if room > 0:
                    orders.setdefault(s_long,  []).append(Order(s_long,  best_ask_l,  room))
                    orders.setdefault(s_short, []).append(Order(s_short, best_bid_s, -room))

            # ── Sell spread ──────────────────────────────────────────────────
            mkt_recv = best_bid_l - best_ask_s
            if mkt_recv - fv_spread > SPREAD_EDGE:
                room = min(
                    LIM_VOUCHER + pos_l,   # room to go shorter on K_long
                    LIM_VOUCHER - pos_s,   # room to go longer on K_short
                    SPREAD_SIZE,
                )
                if room > 0:
                    orders.setdefault(s_long,  []).append(Order(s_long,  best_bid_l, -room))
                    orders.setdefault(s_short, []).append(Order(s_short, best_ask_s,  room))

        return orders

    # ─── Individual option strategies ─────────────────────────────────────────

    def _take_mispriced(self, sym: str, od: OrderDepth, fv: float,
                         pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_room  =  limit - pos
        sell_room =  limit + pos

        for ask in sorted(od.sell_orders):
            if ask >= fv - TAKE_EDGE or buy_room <= 0:
                break
            vol = min(-od.sell_orders[ask], buy_room, 20)
            if vol > 0:
                orders.append(Order(sym, ask, vol))
                buy_room -= vol

        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= fv + TAKE_EDGE or sell_room <= 0:
                break
            vol = min(od.buy_orders[bid], sell_room, 20)
            if vol > 0:
                orders.append(Order(sym, bid, -vol))
                sell_room -= vol

        return orders

    def _market_make(self, sym: str, od: OrderDepth, fv: float,
                      pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        skew      = pos / limit   # positive = long → lean to sell

        bid_price = int(math.floor(fv - MM_HALF - skew))
        ask_price = int(math.ceil(fv  + MM_HALF - skew))

        if bid_price >= ask_price:
            return orders

        buy_room  = limit - pos
        sell_room = limit + pos
        skew_adj  = int(abs(skew) * MM_SIZE)

        bid_size  = max(1, MM_SIZE - skew_adj if skew > 0 else MM_SIZE + skew_adj)
        ask_size  = max(1, MM_SIZE + skew_adj if skew > 0 else MM_SIZE - skew_adj)

        if buy_room  > 0:
            orders.append(Order(sym, bid_price,  min(bid_size,  buy_room)))
        if sell_room > 0:
            orders.append(Order(sym, ask_price, -min(ask_size, sell_room)))

        return orders

    def _vol_trade(self, sym: str, od: OrderDepth, S: float, K: int,
                    tte: float, pos: int, limit: int) -> List[Order]:
        if tte < 0.01 or not od.buy_orders or not od.sell_orders:
            return []
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid      = (best_bid + best_ask) / 2.0

        iv = implied_vol(mid, S, K, tte)
        if iv is None:
            return []

        buy_room  =  limit - pos
        sell_room =  limit + pos
        orders: List[Order] = []

        if iv < SIGMA * (1 - VOL_THRESH) and buy_room > 0:
            orders.append(Order(sym, best_ask, min(5, buy_room)))
        elif iv > SIGMA * (1 + VOL_THRESH) and sell_room > 0:
            orders.append(Order(sym, best_bid, -min(5, sell_room)))

        return orders

    # ─── Strike-to-strike arbitrage ───────────────────────────────────────────

    def _strike_arb(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Model-free arbitrage.

        Monotonicity:  C(K1) >= C(K2) for K1 < K2.
        Butterfly:     C(K1) + C(K3) - 2*C(K2) >= 0 on equal-spaced triples.

        Both structures are inherently low-delta by construction.
        """
        arb: Dict[str, List[Order]] = {}

        bb: Dict[int, int] = {}
        ba: Dict[int, int] = {}
        pm: Dict[int, int] = {}
        for k in VEV_STRIKES:
            sym = VEV_SYM[k]
            if sym not in state.order_depths:
                continue
            od = state.order_depths[sym]
            if od.buy_orders and od.sell_orders:
                bb[k] = max(od.buy_orders)
                ba[k] = min(od.sell_orders)
            pm[k] = state.position.get(sym, 0)

        avail = sorted(set(bb) & set(ba))

        # Monotonicity
        for i in range(len(avail) - 1):
            k1, k2 = avail[i], avail[i + 1]
            profit = bb[k2] - ba[k1]
            if profit <= ARB_EDGE:
                continue
            room = min(LIM_VOUCHER - pm.get(k1, 0),
                       LIM_VOUCHER + pm.get(k2, 0), 5)
            if room <= 0:
                continue
            s1, s2 = VEV_SYM[k1], VEV_SYM[k2]
            arb.setdefault(s1, []).append(Order(s1, ba[k1],  room))
            arb.setdefault(s2, []).append(Order(s2, bb[k2], -room))

        # Butterfly
        for k2, ds in BFLY_SPACING.items():
            k1, k3 = k2 - ds, k2 + ds
            if any(k not in bb for k in [k1, k2, k3]):
                continue
            if any(k not in ba for k in [k1, k2, k3]):
                continue
            cost = ba[k1] + ba[k3] - 2 * bb[k2]
            if cost >= -ARB_EDGE:
                continue
            room = min(
                LIM_VOUCHER - pm.get(k1, 0),
                LIM_VOUCHER - pm.get(k3, 0),
                (LIM_VOUCHER + pm.get(k2, 0)) // 2,
                3,
            )
            if room <= 0:
                continue
            s1, s2, s3 = VEV_SYM[k1], VEV_SYM[k2], VEV_SYM[k3]
            arb.setdefault(s1, []).append(Order(s1, ba[k1],       room))
            arb.setdefault(s2, []).append(Order(s2, bb[k2], -2 * room))
            arb.setdefault(s3, []).append(Order(s3, ba[k3],       room))

        return arb

    # ─── Delta hedge (residual only) ──────────────────────────────────────────

    def _delta_hedge(self, state: TradingState, S: float,
                     total_option_delta: float) -> List[Order]:
        """
        Hedge whatever net delta remains after spread trades have
        internalised most of the exposure.  VELVETFRUIT_EXTRACT is
        used only for this residual.
        """
        od  = state.order_depths.get(UNDERLYING)
        if od is None:
            return []
        pos = state.position.get(UNDERLYING, 0)

        target    = int(round(max(-LIM_UNDER, min(LIM_UNDER, -total_option_delta))))
        delta_qty = target - pos
        if abs(delta_qty) < 1:
            return []

        orders: List[Order] = []

        if delta_qty > 0:
            rem = delta_qty
            for ask in sorted(od.sell_orders):
                if rem <= 0:
                    break
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
                if rem <= 0:
                    break
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
        if not od or not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

    def _clip(self, orders: List[Order], pos: int, limit: int) -> List[Order]:
        """Enforce position limits; earlier orders get capacity priority."""
        out: List[Order] = []
        cur = pos
        for o in orders:
            if o.quantity > 0:
                room = limit - cur
                if room <= 0:
                    continue
                qty = min(o.quantity, room)
            else:
                room = limit + cur
                if room <= 0:
                    continue
                qty = max(o.quantity, -room)
            if qty != 0:
                out.append(Order(o.symbol, o.price, qty))
                cur += qty
        return out
