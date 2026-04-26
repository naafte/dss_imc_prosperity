from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import jsonpickle
import math

class Trader:
    """
    Round 3 Volatility-Smile Strategy - Competitive Market Maker
    
    Final Recalibration:
    1. Competitive Quoting: Joins best bid/ask on 1-tick spreads.
    2. High-Frequency Hedging: Tight 5.0 Delta deadband to prevent drawdown spikes.
    3. WLS-Fit Accuracy: Maintains precise fair values using Weighted Least Squares.
    4. Tapered Sizing: Reduces lot sizes as position limits are approached.
    """

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
    }

    FIT_PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000"]

    DEFAULT_VOL = 0.23
    TICKS_PER_DAY = 1_000_000
    TOTAL_DAYS = 7

    MAX_SPREAD_TO_USE = 25
    MIN_IV_EDGE = 0.010          
    MIN_EXEC_EDGE = 0.51         # Recalibrated to capture 1-tick spread profit
    DELTA_THRESHOLD = 5.0        # Tightened to prevent ~200 pnl drawdowns

    BASE_LOT = 10                
    MAX_TRADES_PER_TICK = 6

    # --- Math Helpers ---

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_d1(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0: return 0.0
        return (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))

    def bs_call(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0: return max(S - K, 0.0)
        d1 = self.bs_d1(S, K, T, sigma)
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)

    def bs_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0: return 1.0 if S > K else 0.0
        return self.norm_cdf(self.bs_d1(S, K, T, sigma))

    def implied_vol(self, price: float, S: float, K: float, T: float) -> Optional[float]:
        intrinsic = max(S - K, 0.0)
        if price <= intrinsic + 1e-4: return None
        lo, hi = 1e-5, 3.0
        for _ in range(45):
            mid = (lo + hi) / 2.0
            if self.bs_call(S, K, T, mid) < price: lo = mid
            else: hi = mid
        return (lo + hi) / 2.0

    # --- Matrix Fitting (WLS) ---

    def solve_3x3(self, A, b):
        M = [A[i] + [b[i]] for i in range(3)]
        for col in range(3):
            pivot = max(range(col, 3), key=lambda r: abs(M[r][col]))
            M[col], M[pivot] = M[pivot], M[col]
            if abs(M[col][col]) < 1e-10: return [self.DEFAULT_VOL, 0.0, 0.0]
            div = M[col][col]
            for j in range(col, 4): M[col][j] /= div
            for r in range(3):
                if r == col: continue
                factor = M[r][col]
                for j in range(col, 4): M[r][j] -= factor * M[col][j]
        return [M[i][3] for i in range(3)]

    def fit_quadratic_wls(self, points: List[Tuple[float, float, float]]):
        xs, ys, ws = [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]
        sw, sx, sx2, sx3, sx4 = sum(ws), sum(w*x for x, w in zip(xs, ws)), sum(w*x*x for x, w in zip(xs, ws)), sum(w*x**3 for x, w in zip(xs, ws)), sum(w*x**4 for x, w in zip(xs, ws))
        sy, sxy, sx2y = sum(w*y for y, w in zip(ys, ws)), sum(w*x*y for x, y, w in zip(xs, ys, ws)), sum(w*x*x*y for x, y, w in zip(xs, ys, ws))
        return self.solve_3x3([[sw, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]], [sy, sxy, sx2y])

    # --- State Helpers ---

    def best_bid_ask(self, depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bid = max(depth.buy_orders) if depth.buy_orders else None
        ask = min(depth.sell_orders) if depth.sell_orders else None
        return bid, ask

    def mid_price(self, depth: OrderDepth) -> Optional[float]:
        bid, ask = self.best_bid_ask(depth)
        if bid is None or ask is None: return None
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

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions, trader_data = 0, ""

        if self.UNDERLYING not in state.order_depths:
            return result, conversions, trader_data

        u_depth = state.order_depths[self.UNDERLYING]
        u_bid, u_ask = self.best_bid_ask(u_depth)
        if u_bid is None or u_ask is None: return result, conversions, trader_data
        
        # Protective Throttling (allows trading in standard market conditions)
        if (u_ask - u_bid) > 10: return result, conversions, trader_data
        
        S = (u_bid + u_ask) / 2.0
        T = self.current_T(state)
        
        rows, fit_points = [], []
        for product, K in sorted(self.VOUCHERS.items(), key=lambda kv: kv[1]):
            if product not in state.order_depths: continue
            depth = state.order_depths[product]
            bid, ask = self.best_bid_ask(depth)
            if bid is None or ask is None: continue
            spread = ask - bid
            if spread < 0 or spread > self.MAX_SPREAD_TO_USE: continue
            mid = (bid + ask) / 2.0
            iv = self.implied_vol(mid, S, K, T)
            if iv is None or not (0.04 <= iv <= 1.50): continue
            m = math.log(K / S)
            rows.append({"product": product, "K": K, "bid": bid, "ask": ask, "spread": spread, "mid": mid, "iv": iv, "m": m})
            if product in self.FIT_PRODUCTS:
                weight = 1.0 / max(1.0, float(spread))
                fit_points.append((m, iv, weight))

        if len(fit_points) < 4: return result, conversions, trader_data
        a, b, c = self.fit_quadratic_wls(fit_points)

        # 3. Competitive Market Making
        for r in rows:
            product, pos, limit = r["product"], self.position(state, r["product"]), self.POSITION_LIMITS[r["product"]]
            fair_iv = max(0.01, a + b * r["m"] + c * r["m"] * r["m"])
            fair_price = self.bs_call(S, r["K"], T, fair_iv)
            
            # Tapered Lot Sizing
            b_size = int(self.BASE_LOT * (1 - max(0, pos) / limit))
            s_size = int(self.BASE_LOT * (1 - max(0, -pos) / limit))

            # PASSIVE BUY (Competitive Price Improvement)
            rb = self.room_buy(state, product)
            if rb > 0 and b_size > 0:
                p = min(math.floor(fair_price - self.MIN_EXEC_EDGE), r["ask"] - 1)
                # Join the bid or improve it if fair value allows
                target_p = max(p, r["bid"] + 1) if p > r["bid"] else p
                result.setdefault(product, []).append(Order(product, int(target_p), min(b_size, rb)))

            # PASSIVE SELL (Competitive Price Improvement)
            rs = self.room_sell(state, product)
            if rs > 0 and s_size > 0:
                p = max(math.ceil(fair_price + self.MIN_EXEC_EDGE), r["bid"] + 1)
                target_p = min(p, r["ask"] - 1) if p < r["ask"] else p
                result.setdefault(product, []).append(Order(product, int(target_p), -min(s_size, rs)))

        # 4. Aggressive Portfolio Delta Hedging
        portfolio_delta = 0.0
        for p, K in self.VOUCHERS.items():
            pos = self.position(state, p)
            if pos != 0:
                m = math.log(K / S)
                fair_iv = max(0.01, a + b * m + c * m * m)
                portfolio_delta += (pos * self.bs_delta(S, K, T, fair_iv))
        
        u_pos = self.position(state, self.UNDERLYING)
        net_delta = portfolio_delta + u_pos
        
        # High-frequency hedging: neutralize as soon as we breach threshold
        if abs(net_delta) > self.DELTA_THRESHOLD:
            h_qty = -int(round(net_delta))
            if h_qty > 0:
                rb = self.room_buy(state, self.UNDERLYING)
                if rb > 0: result.setdefault(self.UNDERLYING, []).append(Order(self.UNDERLYING, int(u_ask), min(h_qty, rb)))
            elif h_qty < 0:
                rs = self.room_sell(state, self.UNDERLYING)
                if rs > 0: result.setdefault(self.UNDERLYING, []).append(Order(self.UNDERLYING, int(u_bid), max(h_qty, -rs)))

        return result, conversions, trader_data