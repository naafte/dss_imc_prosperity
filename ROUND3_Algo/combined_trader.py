from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import jsonpickle
import math


class Trader:
    # --- Options / VEV constants ---
    UNDERLYING = "VELVETFRUIT_EXTRACT"

    VOUCHERS = {
        "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
        "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }

    POSITION_LIMITS = {
        "VELVETFRUIT_EXTRACT": 400, "VEV_4000": 100, "VEV_4500": 100, "VEV_5000": 100,
        "VEV_5100": 100, "VEV_5200": 100, "VEV_5300": 100, "VEV_5400": 100,
        "VEV_5500": 100, "VEV_6000": 80, "VEV_6500": 60,
        "HYDROGEL_PACK": 30,
    }

    FIT_PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000"]

    DEFAULT_VOL = 0.23
    TICKS_PER_DAY = 1_000_000
    TOTAL_DAYS = 7

    MAX_SPREAD_TO_USE = 25
    MIN_IV_EDGE = 0.010
    MIN_EXEC_EDGE = 0.75
    DELTA_THRESHOLD = 8.0

    BASE_LOT = 2
    MAX_TRADES_PER_TICK = 5

    # --- Hydrogel constants ---
    HP_PRODUCT   = "HYDROGEL_PACK"
    HP_LIMIT     = 30
    HP_MM_SIZE   = 5
    HP_AGGR_EDGE = 5
    HP_AGGR_SIZE = 15

    # --- VFE spot constants ---
    VFE_FV        = 5_250
    VFE_LIMIT     = 400   # matches POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
    VFE_MM_HALF   = 8
    VFE_MM_SIZE   = 5
    VFE_AGGR_EDGE = 20
    VFE_AGGR_SIZE = 5

    # =========================================================================
    # Math helpers (options)
    # =========================================================================

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_d1(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))

    def bs_call(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        d1 = self.bs_d1(S, K, T, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)

    def bs_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        return self.norm_cdf(self.bs_d1(S, K, T, sigma))

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

    # =========================================================================
    # WLS smile fitting
    # =========================================================================

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

    def fit_quadratic_wls(self, points: List[Tuple[float, float, float]]):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ws = [p[2] for p in points]
        sw   = sum(ws)
        sx   = sum(w * x for x, w in zip(xs, ws))
        sx2  = sum(w * x * x for x, w in zip(xs, ws))
        sx3  = sum(w * x ** 3 for x, w in zip(xs, ws))
        sx4  = sum(w * x ** 4 for x, w in zip(xs, ws))
        sy   = sum(w * y for y, w in zip(ys, ws))
        sxy  = sum(w * x * y for x, y, w in zip(xs, ys, ws))
        sx2y = sum(w * x * x * y for x, y, w in zip(xs, ys, ws))
        return self.solve_3x3(
            [[sw, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]],
            [sy, sxy, sx2y],
        )

    # =========================================================================
    # Order-book helpers
    # =========================================================================

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

    def room_buy(self, state: TradingState, product: str, extra_bought: int = 0) -> int:
        limit = self.POSITION_LIMITS[product]
        return max(0, limit - self.position(state, product) - extra_bought)

    def room_sell(self, state: TradingState, product: str, extra_sold: int = 0) -> int:
        limit = self.POSITION_LIMITS[product]
        return max(0, limit + self.position(state, product) - extra_sold)

    def add_order(self, result: Dict[str, List[Order]], product: str, price: int, qty: int):
        if qty == 0:
            return
        if product not in result:
            result[product] = []
        result[product].append(Order(product, price, qty))

    # =========================================================================
    # Options + delta-hedge logic
    # =========================================================================

    def _options_orders(self, state: TradingState, result: Dict[str, List[Order]]):
        """Populate result with VEV_* voucher orders and VELVETFRUIT_EXTRACT delta hedge."""
        if self.UNDERLYING not in state.order_depths:
            return

        S = self.mid_price(state.order_depths[self.UNDERLYING])
        if S is None:
            return
        T = self.current_T(state)

        rows, fit_points = [], []

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
            rows.append({"product": product, "K": K, "bid": bid, "ask": ask,
                         "spread": spread, "mid": mid, "iv": iv, "m": m})

            if product in self.FIT_PRODUCTS:
                weight = 1.0 / max(1.0, float(spread))
                fit_points.append((m, iv, weight))

        if len(fit_points) < 4:
            return

        a, b, c = self.fit_quadratic_wls(fit_points)

        # Passive quoting on vouchers
        for r in rows:
            product = r["product"]
            fair_iv = max(0.01, a + b * r["m"] + c * r["m"] * r["m"])
            fair_price = self.bs_call(S, r["K"], T, fair_iv)
            required_edge = self.MIN_EXEC_EDGE

            room_buy = self.room_buy(state, product)
            if room_buy > 0:
                max_buy_price = math.floor(fair_price - required_edge)
                target_bid = min(max_buy_price, r["bid"] + 1)
                final_bid = min(target_bid, r["ask"] - 1)
                qty = min(10, room_buy)
                self.add_order(result, product, int(final_bid), qty)

            room_sell = self.room_sell(state, product)
            if room_sell > 0:
                min_sell_price = math.ceil(fair_price + required_edge)
                target_ask = max(min_sell_price, r["ask"] - 1)
                final_ask = max(target_ask, r["bid"] + 1)
                qty = min(10, room_sell)
                self.add_order(result, product, int(final_ask), -qty)

        # Delta hedge via underlying
        portfolio_delta = 0.0
        for product, K in self.VOUCHERS.items():
            pos = self.position(state, product)
            if pos != 0:
                m = math.log(K / S)
                fair_iv = max(0.01, a + b * m + c * m * m)
                portfolio_delta += pos * self.bs_delta(S, K, T, fair_iv)

        underlying_pos = self.position(state, self.UNDERLYING)
        net_portfolio_delta = portfolio_delta + underlying_pos

        if abs(net_portfolio_delta) > self.DELTA_THRESHOLD:
            hedge_qty = -int(round(net_portfolio_delta))
            if hedge_qty != 0:
                depth = state.order_depths[self.UNDERLYING]
                if hedge_qty > 0:
                    qty = min(hedge_qty, self.room_buy(state, self.UNDERLYING))
                    best_ask = min(depth.sell_orders) if depth.sell_orders else None
                    if best_ask and qty > 0:
                        self.add_order(result, self.UNDERLYING, best_ask, qty)
                elif hedge_qty < 0:
                    qty = max(hedge_qty, -self.room_sell(state, self.UNDERLYING))
                    best_bid = max(depth.buy_orders) if depth.buy_orders else None
                    if best_bid and qty < 0:
                        self.add_order(result, self.UNDERLYING, best_bid, qty)

    # =========================================================================
    # VFE spot market-making
    # =========================================================================

    def _vfe_orders(self, state: TradingState, result: Dict[str, List[Order]]):
        """Add spot VFE orders on top of any existing delta-hedge orders."""
        if self.UNDERLYING not in state.order_depths:
            return

        depth = state.order_depths[self.UNDERLYING]
        position = self.position(state, self.UNDERLYING)

        # Account for hedge orders already queued this tick
        already_bought = sum(o.quantity for o in result.get(self.UNDERLYING, []) if o.quantity > 0)
        already_sold   = sum(-o.quantity for o in result.get(self.UNDERLYING, []) if o.quantity < 0)

        rem_buy  = self.room_buy(state, self.UNDERLYING, extra_bought=already_bought)
        rem_sell = self.room_sell(state, self.UNDERLYING, extra_sold=already_sold)

        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None

        # Aggressive taking vs fair value
        if best_ask is not None and best_ask < self.VFE_FV - self.VFE_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.VFE_AGGR_SIZE, rem_buy)
            self.add_order(result, self.UNDERLYING, best_ask, vol)
            rem_buy -= vol

        if best_bid is not None and best_bid > self.VFE_FV + self.VFE_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.VFE_AGGR_SIZE, rem_sell)
            self.add_order(result, self.UNDERLYING, best_bid, -vol)
            rem_sell -= vol

        # Passive limit orders with inventory skew
        skew   = int(position / self.VFE_LIMIT * self.VFE_MM_HALF * 2)
        my_bid = self.VFE_FV - self.VFE_MM_HALF - skew
        my_ask = self.VFE_FV + self.VFE_MM_HALF - skew

        if position > self.VFE_LIMIT // 3 and best_ask is not None:
            my_ask = min(my_ask, best_ask)
        if position < -(self.VFE_LIMIT // 3) and best_bid is not None:
            my_bid = max(my_bid, best_bid)

        if rem_buy > 0 and (best_ask is None or my_bid < best_ask):
            self.add_order(result, self.UNDERLYING, my_bid, min(self.VFE_MM_SIZE, rem_buy))
        if rem_sell > 0 and (best_bid is None or my_ask > best_bid):
            self.add_order(result, self.UNDERLYING, my_ask, -min(self.VFE_MM_SIZE, rem_sell))

    # =========================================================================
    # Hydrogel market-making
    # =========================================================================

    def _hp_orders(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        depth = state.order_depths[self.HP_PRODUCT]
        position = self.position(state, self.HP_PRODUCT)

        buy_prices  = sorted(depth.buy_orders.keys(),  reverse=True)
        sell_prices = sorted(depth.sell_orders.keys())

        best_bid = buy_prices[0]  if buy_prices  else None
        best_ask = sell_prices[0] if sell_prices else None

        mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 10_000

        rem_buy  = self.HP_LIMIT - position
        rem_sell = self.HP_LIMIT + position

        # Aggressive taking
        if best_ask is not None and best_ask <= mid_price - self.HP_AGGR_EDGE and rem_buy > 0:
            vol = min(-depth.sell_orders[best_ask], self.HP_AGGR_SIZE, rem_buy)
            orders.append(Order(self.HP_PRODUCT, best_ask, vol))
            rem_buy -= vol

        if best_bid is not None and best_bid >= mid_price + self.HP_AGGR_EDGE and rem_sell > 0:
            vol = min(depth.buy_orders[best_bid], self.HP_AGGR_SIZE, rem_sell)
            orders.append(Order(self.HP_PRODUCT, best_bid, -vol))
            rem_sell -= vol

        # Competitive market making
        if best_bid is not None:
            my_bid = best_bid + 1 - (1 if position > 10 else 0)
        else:
            my_bid = int(mid_price - 5)

        if best_ask is not None:
            my_ask = best_ask - 1 + (1 if position < -10 else 0)
        else:
            my_ask = int(mid_price + 5)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        if rem_buy > 0:
            orders.append(Order(self.HP_PRODUCT, int(my_bid), min(self.HP_MM_SIZE, rem_buy)))
        if rem_sell > 0:
            orders.append(Order(self.HP_PRODUCT, int(my_ask), -min(self.HP_MM_SIZE, rem_sell)))

        return orders

    # =========================================================================
    # Entry point
    # =========================================================================

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # 1. Options smile + delta hedge (populates VEV_* and VELVETFRUIT_EXTRACT)
        self._options_orders(state, result)

        # 2. VFE spot market-making (respects any hedge orders already queued)
        self._vfe_orders(state, result)

        # 3. Hydrogel market-making
        if self.HP_PRODUCT in state.order_depths:
            result[self.HP_PRODUCT] = self._hp_orders(state)

        return result, 0, ""
