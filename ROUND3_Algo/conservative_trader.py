from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional
import math

# ─── Constants ────────────────────────────────────────────────────────────────

UNDERLYING    = "VELVETFRUIT_EXTRACT"
VEV_STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_SYM       = {k: f"VEV_{k}" for k in VEV_STRIKES}

LIM_UNDER     = 200
LIM_VOUCHER   = 300

TTE_AT_START  = 5
TICKS_PER_DAY = 1_000_000
S_FALLBACK    = 5250.0

# Small lot sizes keep exposure low and avoid exhausting the book
ARB_SIZE  = 5   # max units per spread arb trade
BFLY_SIZE = 3   # max units per butterfly arb trade
INTR_SIZE = 5   # max units per intrinsic arb trade

# Butterfly wing spacings (center_strike -> spacing from centre to each wing)
BFLY_SPACING: Dict[int, int] = {
    4500: 500, 5000: 500, 5100: 100, 5200: 100,
    5300: 100, 5400: 100, 6000: 500,
}


# ─── Trader ───────────────────────────────────────────────────────────────────

class Trader:
    """
    Conservative arbitrage-only trader.

    Three strategies, all with mathematically guaranteed non-negative payoffs:

    1. Intrinsic value arb — buy any call trading below its intrinsic value
       max(S-K, 0). At expiry the option settles at ≥ intrinsic, locking profit.

    2. Call-spread dominance — call prices must be non-increasing in strike
       (C(K1) ≥ C(K2) for K1 < K2). If ask[K1] < bid[K2], buying K1 and selling
       K2 costs negative net cash; the spread payoff is always ≥ 0, so any credit
       received is pure profit regardless of where the underlying ends up.

    3. Butterfly convexity — call prices must be convex in strike
       (C(K1)+C(K3) ≥ 2*C(K2)). If ask[K1]+ask[K3] < 2*bid[K2], buying the
       butterfly (long K1, short 2*K2, long K3) has negative cost and a payoff
       that is always ≥ 0.

    No delta hedging needed: every position is fully hedged by its own structure.
    """

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # ── Snapshot spot price ───────────────────────────────────────────────
        S = self._mid(state, UNDERLYING)
        if S is None:
            S = S_FALLBACK

        # ── Build per-strike market snapshot ─────────────────────────────────
        bb:  Dict[int, int] = {}   # best bid price
        ba:  Dict[int, int] = {}   # best ask price
        bv:  Dict[int, int] = {}   # available bid volume (positive)
        av:  Dict[int, int] = {}   # available ask volume (positive)

        for k in VEV_STRIKES:
            sym = VEV_SYM[k]
            if sym not in state.order_depths:
                continue
            od = state.order_depths[sym]
            if od.buy_orders:
                bb[k] = max(od.buy_orders)
                bv[k] = od.buy_orders[bb[k]]          # positive
            if od.sell_orders:
                ba[k] = min(od.sell_orders)
                av[k] = -od.sell_orders[ba[k]]         # store as positive

        # Track running positions to respect limits across strategies
        cur: Dict[int, int] = {k: state.position.get(VEV_SYM[k], 0) for k in VEV_STRIKES}
        pending: Dict[str, List[Order]] = {}

        def place(k: int, price: int, qty: int) -> None:
            sym = VEV_SYM[k]
            cur[k] += qty
            pending.setdefault(sym, []).append(Order(sym, price, qty))

        # ── Strategy 1: Intrinsic value arbitrage ─────────────────────────────
        # Buy call if ask < max(S-K, 0). Payoff at expiry ≥ intrinsic ≥ ask paid.
        for k in VEV_STRIKES:
            if k not in ba:
                continue
            intrinsic = S - k
            if intrinsic < 1.0:       # only act on clearly ITM strikes
                continue
            if ba[k] < intrinsic - 0.5:
                room = LIM_VOUCHER - cur[k]
                vol  = min(INTR_SIZE, av.get(k, 0), room)
                if vol > 0:
                    place(k, ba[k], vol)

        # ── Strategy 2: Adjacent call-spread dominance arbitrage ───────────────
        # Buy lower-strike K1 at ask, sell higher-strike K2 at bid.
        # Profit = bid[K2] - ask[K1] > 0; spread payoff in [0, K2-K1] always ≥ 0.
        avail = sorted(set(bb) & set(ba))
        for i in range(len(avail) - 1):
            k1, k2 = avail[i], avail[i + 1]
            if ba[k1] >= bb[k2]:
                continue                         # no arb
            profit_per_unit = bb[k2] - ba[k1]
            if profit_per_unit <= 0:
                continue
            room1 = LIM_VOUCHER - cur[k1]
            room2 = LIM_VOUCHER + cur[k2]        # selling k2 so we need short room
            vol   = min(ARB_SIZE, av.get(k1, 0), bv.get(k2, 0), room1, room2)
            if vol > 0:
                place(k1, ba[k1],  vol)
                place(k2, bb[k2], -vol)

        # ── Strategy 3: Butterfly convexity arbitrage ─────────────────────────
        # Long K1 + K3 wings at ask, short 2*K2 body at bid.
        # Cost = ask[K1] + ask[K3] - 2*bid[K2]; must be < 0 for arb.
        # Butterfly payoff is always ≥ 0 (convexity of option payoff).
        for k2, ds in BFLY_SPACING.items():
            k1, k3 = k2 - ds, k2 + ds
            if k1 not in ba or k3 not in ba or k2 not in bb:
                continue
            net_cost = ba[k1] + ba[k3] - 2 * bb[k2]
            if net_cost >= 0:
                continue                         # no arb
            room1 = LIM_VOUCHER - cur[k1]
            room2 = LIM_VOUCHER + cur[k2]        # selling 2*vol at k2
            room3 = LIM_VOUCHER - cur[k3]
            vol   = min(BFLY_SIZE,
                        av.get(k1, 0),
                        bv.get(k2, 0) // 2,
                        av.get(k3, 0),
                        room1,
                        room2 // 2,
                        room3)
            if vol > 0:
                place(k1, ba[k1],        vol)
                place(k2, bb[k2], -2 * vol)
                place(k3, ba[k3],        vol)

        result.update(pending)
        return result, 0, ""

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _mid(self, state: TradingState, sym: str) -> Optional[float]:
        od = state.order_depths.get(sym)
        if not od or not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
