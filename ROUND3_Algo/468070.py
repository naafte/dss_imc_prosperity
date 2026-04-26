from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import jsonpickle
import math


class Trader:
    """
    Round 3 active volatility-smile strategy.

    Fix from prior conservative version:
    - The old version required BOTH a large IV residual and a large mid-price residual,
      then required a nearby opposite-side pair. That was too strict, so it often sent no orders.
    - This version trades only when the EXECUTABLE price has edge:
          buy only if fair_value > best_ask + buffer
          sell only if best_bid > fair_value + buffer
      That is much better than trading from mid-price residuals.
    - It still scales volume by conviction and caps exposure hard.
    """

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    VOUCHERS = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    POSITION_LIMITS = {
        "VELVETFRUIT_EXTRACT": 400,
        "VEV_4000": 100,
        "VEV_4500": 100,
        "VEV_5000": 100,
        "VEV_5100": 100,
        "VEV_5200": 100,
        "VEV_5300": 100,
        "VEV_5400": 100,
        "VEV_5500": 100,
        "VEV_6000": 80,
        "VEV_6500": 60,
    }

    FIT_PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000"]

    DEFAULT_VOL = 0.23
    TICKS_PER_DAY = 1_000_000
    TOTAL_DAYS = 7

    # Less frozen, still not reckless.
    MAX_SPREAD_TO_USE = 25
    MIN_IV_EDGE = 0.010          # 1 vol point minimum residual
    MIN_EXEC_EDGE = 0.75         # executable edge after crossing bid/ask
    EDGE_BUFFER_FRACTION = 0.20  # require edge to clear part of spread too

    BASE_LOT = 2
    MAX_LEG_SIZE = 14
    FULL_SIGNAL_IV = 0.055
    MAX_TRADES_PER_TICK = 4

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_call(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)

    def implied_vol(self, price: float, S: float, K: float, T: float) -> Optional[float]:
        intrinsic = max(S - K, 0.0)
        if price <= intrinsic + 1e-4:
            return None
        lo, hi = 1e-5, 3.0
        for _ in range(45):
            mid = (lo + hi) / 2.0
            if self.bs_call(S, K, T, mid) < price:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def best_bid_ask(self, depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bid = max(depth.buy_orders) if depth.buy_orders else None
        ask = min(depth.sell_orders) if depth.sell_orders else None
        return bid, ask

    def mid_price(self, depth: OrderDepth) -> Optional[float]:
        bid, ask = self.best_bid_ask(depth)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def current_T(self, state: TradingState) -> float:
        day = state.timestamp // self.TICKS_PER_DAY
        days_left = max(1, self.TOTAL_DAYS - day)
        return days_left / 365.0

    def position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def room_buy(self, state: TradingState, product: str) -> int:
        return max(0, self.POSITION_LIMITS[product] - self.position(state, product))

    def room_sell(self, state: TradingState, product: str) -> int:
        return max(0, self.POSITION_LIMITS[product] + self.position(state, product))

    def solve_3x3(self, A, b):
        M = [A[i] + [b[i]] for i in range(3)]
        for col in range(3):
            pivot = max(range(col, 3), key=lambda r: abs(M[r][col]))
            M[col], M[pivot] = M[pivot], M[col]
            if abs(M[col][col]) < 1e-10:
                return [self.DEFAULT_VOL, 0.0, 0.0]
            div = M[col][col]
            for j in range(col, 4):
                M[col][j] /= div
            for r in range(3):
                if r == col:
                    continue
                factor = M[r][col]
                for j in range(col, 4):
                    M[r][j] -= factor * M[col][j]
        return [M[i][3] for i in range(3)]

    def fit_quadratic(self, points: List[Tuple[float, float]]):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        n = len(points)
        sx = sum(xs)
        sx2 = sum(x*x for x in xs)
        sx3 = sum(x*x*x for x in xs)
        sx4 = sum(x*x*x*x for x in xs)
        sy = sum(ys)
        sxy = sum(x*y for x, y in zip(xs, ys))
        sx2y = sum(x*x*y for x, y in zip(xs, ys))
        A = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]]
        b = [sy, sxy, sx2y]
        return self.solve_3x3(A, b)

    def add_order(self, result: Dict[str, List[Order]], product: str, price: int, qty: int):
        if qty == 0:
            return
        if product not in result:
            result[product] = []
        result[product].append(Order(product, price, qty))

    def size_from_signal(self, iv_edge: float, exec_edge: float) -> int:
        iv_strength = (abs(iv_edge) - self.MIN_IV_EDGE) / max(1e-9, self.FULL_SIGNAL_IV - self.MIN_IV_EDGE)
        px_strength = exec_edge / 8.0
        strength = max(0.0, min(1.0, 0.65 * iv_strength + 0.35 * px_strength))
        return max(self.BASE_LOT, int(self.BASE_LOT + strength * (self.MAX_LEG_SIZE - self.BASE_LOT)))

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        trader_data = {}
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
            except Exception:
                trader_data = {}

        if self.UNDERLYING not in state.order_depths:
            return result, conversions, jsonpickle.encode(trader_data)

        S = self.mid_price(state.order_depths[self.UNDERLYING])
        if S is None:
            return result, conversions, jsonpickle.encode(trader_data)

        T = self.current_T(state)
        rows = []
        fit_points = []

        for product, K in sorted(self.VOUCHERS.items(), key=lambda kv: kv[1]):
            if product not in state.order_depths:
                continue
            depth = state.order_depths[product]
            bid, ask = self.best_bid_ask(depth)
            if bid is None or ask is None:
                continue
            spread = ask - bid
            if spread < 0 or spread > self.MAX_SPREAD_TO_USE:
                continue
            mid = (bid + ask) / 2.0
            iv = self.implied_vol(mid, S, K, T)
            if iv is None or not (0.04 <= iv <= 1.50):
                continue
            m = math.log(K / S)
            row = {"product": product, "K": K, "bid": bid, "ask": ask, "spread": spread, "mid": mid, "iv": iv, "m": m}
            rows.append(row)
            if product in self.FIT_PRODUCTS:
                fit_points.append((m, iv))

        if len(fit_points) < 4:
            return result, conversions, jsonpickle.encode(trader_data)

        a, b, c = self.fit_quadratic(fit_points)

        candidates = []
        for r in rows:
            fair_iv = max(0.01, a + b * r["m"] + c * r["m"] * r["m"])
            fair_price = self.bs_call(S, r["K"], T, fair_iv)
            iv_edge = r["iv"] - fair_iv

            # Dynamic buffer: small absolute edge plus a fraction of the spread.
            buffer = max(self.MIN_EXEC_EDGE, self.EDGE_BUFFER_FRACTION * r["spread"])

            # Rich: executable sell edge exists at current bid.
            sell_edge = r["bid"] - fair_price
            if iv_edge > self.MIN_IV_EDGE and sell_edge > buffer:
                candidates.append({**r, "side": "SELL", "iv_edge": iv_edge, "exec_edge": sell_edge})

            # Cheap: executable buy edge exists at current ask.
            buy_edge = fair_price - r["ask"]
            if iv_edge < -self.MIN_IV_EDGE and buy_edge > buffer:
                candidates.append({**r, "side": "BUY", "iv_edge": iv_edge, "exec_edge": buy_edge})

        # Trade strongest executable edges only.
        candidates.sort(key=lambda x: (x["exec_edge"], abs(x["iv_edge"])), reverse=True)

        trades = 0
        for r in candidates:
            if trades >= self.MAX_TRADES_PER_TICK:
                break
            product = r["product"]
            depth = state.order_depths[product]
            desired = self.size_from_signal(r["iv_edge"], r["exec_edge"])

            if r["side"] == "BUY":
                book_qty = abs(depth.sell_orders.get(r["ask"], 0))
                qty = min(desired, book_qty, self.room_buy(state, product))
                if qty > 0:
                    self.add_order(result, product, r["ask"], qty)
                    trades += 1
            else:
                book_qty = abs(depth.buy_orders.get(r["bid"], 0))
                qty = min(desired, book_qty, self.room_sell(state, product))
                if qty > 0:
                    self.add_order(result, product, r["bid"], -qty)
                    trades += 1

        trader_data["last_debug"] = {
            "S": S,
            "T": T,
            "fit": [a, b, c],
            "num_rows": len(rows),
            "num_candidates": len(candidates),
        }
        return result, conversions, jsonpickle.encode(trader_data)