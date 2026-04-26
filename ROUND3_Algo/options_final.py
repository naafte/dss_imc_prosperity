from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import jsonpickle
import math

class Trader:
    """
    Round 3 Volatility-Smile Strategy - Final Optimized Version
    
    Optimizations:
    1. Passive Quoting: Captures the spread by making markets instead of taking them.
    2. WLS Fitting: Anchors the volatility smile to the most liquid strikes.
    3. Delta Deadband: Neutralizes directional risk efficiently via the underlying.
    4. Execution Logic: Disabled aggressive inventory penalties to maximize PnL.
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
    MIN_EXEC_EDGE = 0.75         # Margin added to fair price for market making
    DELTA_THRESHOLD = 8.0        # Neutralizes risk once delta breaches this deadband

    BASE_LOT = 2
    MAX_TRADES_PER_TICK = 5

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
        sw, sx, sx2 = sum(ws), sum(w*x for x, w in zip(xs, ws)), sum(w*x*x for x, w in zip(xs, ws))
        sx3, sx4 = sum(w*x**3 for x, w in zip(xs, ws)), sum(w*x**4 for x, w in zip(xs, ws))
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

    def add_order(self, result: Dict[str, List[Order]], product: str, price: int, qty: int):
        if qty == 0: return
        if product not in result: result[product] = []
        result[product].append(Order(product, price, qty))

    # --- Execution Loop ---

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        trader_data = ""

        if self.UNDERLYING not in state.order_depths:
            return result, conversions, trader_data

        S = self.mid_price(state.order_depths[self.UNDERLYING])
        if S is None: return result, conversions, trader_data
        T = self.current_T(state)
        
        rows, fit_points = [], []

        # 1. Collect market data and IVs
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

        # 2. Smooth the smile using WLS
        a, b, c = self.fit_quadratic_wls(fit_points)

        # 3. Aggressive Passive Quoting (Market Making)
        for r in rows:
            product = r["product"]
            current_pos = self.position(state, product)
            
            fair_iv = max(0.01, a + b * r["m"] + c * r["m"] * r["m"])
            fair_price = self.bs_call(S, r["K"], T, fair_iv)
            
            # THE EDGE: Minimum profit we must make per contract
            # 0.75 is safe, but 0.50 might get more fills if -7 persists
            required_edge = self.MIN_EXEC_EDGE 

            # --- AGGRESSIVE PASSIVE BID ---
            room_buy = self.room_buy(state, product)
            if room_buy > 0:
                # Calculate the highest price we are willing to pay
                max_buy_price = math.floor(fair_price - required_edge)
                
                # Aim to join the best bid or be 1 tick better, 
                # but never exceed our max_buy_price or the market ask
                target_bid = min(max_buy_price, r["bid"] + 1)
                final_bid = min(target_bid, r["ask"] - 1)
                
                # Increase volume: quote larger lots when we have room
                qty = min(10, room_buy) 
                self.add_order(result, product, int(final_bid), qty)

            # --- AGGRESSIVE PASSIVE ASK ---
            room_sell = self.room_sell(state, product)
            if room_sell > 0:
                # Calculate the lowest price we are willing to accept
                min_sell_price = math.ceil(fair_price + required_edge)
                
                # Aim to join the best ask or be 1 tick better
                target_ask = max(min_sell_price, r["ask"] - 1)
                final_ask = max(target_ask, r["bid"] + 1)
                
                qty = min(10, room_sell)
                self.add_order(result, product, int(final_ask), -qty)

        # 4. Delta Hedging existing inventory via the underlying
        portfolio_delta = 0.0
        for product, K in self.VOUCHERS.items():
            pos = self.position(state, product)
            if pos != 0:
                m = math.log(K / S)
                fair_iv = max(0.01, a + b * m + c * m * m)
                portfolio_delta += (pos * self.bs_delta(S, K, T, fair_iv))
                
        underlying_pos = self.position(state, self.UNDERLYING)
        net_portfolio_delta = portfolio_delta + underlying_pos
        
        # Cross the spread on the underlying only if net exposure exceeds deadband
        if abs(net_portfolio_delta) > self.DELTA_THRESHOLD:
            hedge_qty = -int(round(net_portfolio_delta))
            if hedge_qty != 0:
                depth = state.order_depths[self.UNDERLYING]
                if hedge_qty > 0:
                    qty = min(hedge_qty, self.room_buy(state, self.UNDERLYING))
                    best_ask = min(depth.sell_orders) if depth.sell_orders else None
                    if best_ask and qty > 0: self.add_order(result, self.UNDERLYING, best_ask, qty)
                elif hedge_qty < 0:
                    qty = max(hedge_qty, -self.room_sell(state, self.UNDERLYING))
                    best_bid = max(depth.buy_orders) if depth.buy_orders else None
                    if best_bid and qty < 0: self.add_order(result, self.UNDERLYING, best_bid, qty)

        return result, conversions, trader_data