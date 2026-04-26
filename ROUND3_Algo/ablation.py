import pandas as pd
import glob
import math
import jsonpickle
from typing import Dict, List, Optional, Tuple

# =========================================================================
# 1. MOCK DATAMODEL
# =========================================================================

class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class TradingState:
    def __init__(self, traderData: str, timestamp: int, listings: dict, 
                 order_depths: Dict[str, OrderDepth], own_trades: dict, 
                 market_trades: dict, position: Dict[str, int], observations: dict):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

# =========================================================================
# 2. THE UPGRADED TRADER CLASS (Passive Quoting + Delta Deadband)
# =========================================================================

class Trader:
    def __init__(self, use_wls=True, use_inventory_penalty=True, 
                 use_delta_hedge=True, delta_threshold=8.0, risk_aversion=0.005):
        
        self.use_wls = use_wls
        self.use_inventory_penalty = use_inventory_penalty
        self.use_delta_hedge = use_delta_hedge
        self.delta_threshold = delta_threshold
        self.risk_aversion = risk_aversion

        self.UNDERLYING = "VELVETFRUIT_EXTRACT"
        self.VOUCHERS = {
            "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100, 
            "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500, 
            "VEV_6000": 6000, "VEV_6500": 6500
        }
        self.POSITION_LIMITS = {
            "VELVETFRUIT_EXTRACT": 400, "VEV_4000": 100, "VEV_4500": 100, "VEV_5000": 100, 
            "VEV_5100": 100, "VEV_5200": 100, "VEV_5300": 100, "VEV_5400": 100, 
            "VEV_5500": 100, "VEV_6000": 80, "VEV_6500": 60
        }
        self.FIT_PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000"]
        
        self.DEFAULT_VOL = 0.23
        self.TICKS_PER_DAY = 1_000_000
        self.TOTAL_DAYS = 7
        self.MAX_SPREAD_TO_USE = 25
        self.MIN_EXEC_EDGE = 0.75         
        self.BASE_LOT = 2

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

    def fit_quadratic(self, points: List[Tuple[float, float]]):
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        n = len(points)
        sx, sx2, sx3, sx4 = sum(xs), sum(x*x for x in xs), sum(x**3 for x in xs), sum(x**4 for x in xs)
        sy, sxy, sx2y = sum(ys), sum(x*y for x, y in zip(xs, ys)), sum(x*x*y for x, y in zip(xs, ys))
        return self.solve_3x3([[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]], [sy, sxy, sx2y])

    def fit_quadratic_wls(self, points: List[Tuple[float, float, float]]):
        xs, ys, ws = [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]
        sw, sx, sx2 = sum(ws), sum(w*x for x, w in zip(xs, ws)), sum(w*x*x for x, w in zip(xs, ws))
        sx3, sx4 = sum(w*x**3 for x, w in zip(xs, ws)), sum(w*x**4 for x, w in zip(xs, ws))
        sy, sxy, sx2y = sum(w*y for y, w in zip(ys, ws)), sum(w*x*y for x, y, w in zip(xs, ys, ws)), sum(w*x*x*y for x, y, w in zip(xs, ys, ws))
        return self.solve_3x3([[sw, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]], [sy, sxy, sx2y])

    def best_bid_ask(self, depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        return max(depth.buy_orders) if depth.buy_orders else None, min(depth.sell_orders) if depth.sell_orders else None

    def mid_price(self, depth: OrderDepth) -> Optional[float]:
        bid, ask = self.best_bid_ask(depth)
        if bid is None or ask is None: return None
        return (bid + ask) / 2.0

    def current_T(self, state: TradingState) -> float:
        day = state.timestamp // self.TICKS_PER_DAY
        return max(1, self.TOTAL_DAYS - day) / 365.0

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

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        trader_data = ""
        if self.UNDERLYING not in state.order_depths: return result, 0, trader_data

        S = self.mid_price(state.order_depths[self.UNDERLYING])
        if S is None: return result, 0, trader_data
        T = self.current_T(state)
        
        rows, fit_points_wls, fit_points_ols = [], [], []

        # 1. Gather Implied Volatilities
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
                fit_points_wls.append((m, iv, weight))
                fit_points_ols.append((m, iv))

        if len(fit_points_ols) < 4: return result, 0, trader_data

        # 2. Fit the Smile
        if self.use_wls: a, b, c = self.fit_quadratic_wls(fit_points_wls)
        else: a, b, c = self.fit_quadratic(fit_points_ols)

        # 3. Passive Quoting Market Making
        for r in rows:
            product = r["product"]
            current_pos = self.position(state, product)
            
            fair_iv = max(0.01, a + b * r["m"] + c * r["m"] * r["m"])
            fair_price = self.bs_call(S, r["K"], T, fair_iv)
            
            # Base quotes demanding our minimum edge
            my_bid = math.floor(fair_price - self.MIN_EXEC_EDGE)
            my_ask = math.ceil(fair_price + self.MIN_EXEC_EDGE)

            # PASSIVE BUYING
            room_to_buy = self.room_buy(state, product)
            if room_to_buy > 0:
                bid_price = my_bid
                # Inventory skew: if short, pay 1 tick more to cover
                if self.use_inventory_penalty and current_pos < 0:
                    bid_price += 1 
                
                # Prevent crossing the spread to avoid fees
                bid_price = min(bid_price, r["ask"] - 1)
                
                qty = min(self.BASE_LOT, room_to_buy)
                self.add_order(result, product, bid_price, qty)

            # PASSIVE SELLING
            room_to_sell = self.room_sell(state, product)
            if room_to_sell > 0:
                ask_price = my_ask
                # Inventory skew: if long, ask 1 tick less to dump
                if self.use_inventory_penalty and current_pos > 0:
                    ask_price -= 1 
                
                # Prevent crossing the spread to avoid fees
                ask_price = max(ask_price, r["bid"] + 1)
                
                qty = min(self.BASE_LOT, room_to_sell)
                self.add_order(result, product, ask_price, -qty)

        # 4. Strictly Delta Hedge EXISTING Inventory (Do not pre-hedge resting limits!)
        if self.use_delta_hedge:
            portfolio_delta = 0.0
            for product, K in self.VOUCHERS.items():
                pos = self.position(state, product)
                if pos != 0:
                    m = math.log(K / S)
                    fair_iv = max(0.01, a + b * m + c * m * m)
                    portfolio_delta += (pos * self.bs_delta(S, K, T, fair_iv))
                    
            net_portfolio_delta = portfolio_delta + self.position(state, self.UNDERLYING)
            
            if abs(net_portfolio_delta) > self.delta_threshold:
                hedge_qty = -int(round(net_portfolio_delta))
                if hedge_qty != 0:
                    depth = state.order_depths[self.UNDERLYING]
                    if hedge_qty > 0:
                        qty = min(hedge_qty, self.room_buy(state, self.UNDERLYING))
                        ask = min(depth.sell_orders) if depth.sell_orders else None
                        if ask and qty > 0: self.add_order(result, self.UNDERLYING, ask, qty)
                    elif hedge_qty < 0:
                        qty = max(hedge_qty, -self.room_sell(state, self.UNDERLYING))
                        bid = max(depth.buy_orders) if depth.buy_orders else None
                        if bid and qty < 0: self.add_order(result, self.UNDERLYING, bid, qty)

        return result, 0, trader_data

# =========================================================================
# 3. LOCAL SIMULATOR (Modified to simulate Passive Fills)
# =========================================================================

def simulate_day(trader: Trader, df: pd.DataFrame) -> float:
    position = {}
    cash = 0.0
    grouped = df.groupby('timestamp')
    
    for timestamp, group in grouped:
        order_depths = {}
        mid_prices = {}
        
        for _, row in group.iterrows():
            product = row['product']
            od = OrderDepth()
            
            if pd.notna(row['bid_price_1']): od.buy_orders[int(row['bid_price_1'])] = int(row['bid_volume_1'])
            if pd.notna(row['ask_price_1']): od.sell_orders[int(row['ask_price_1'])] = -abs(int(row['ask_volume_1']))
            
            order_depths[product] = od
            mid_prices[product] = row['mid_price']
            if product not in position: position[product] = 0

        state = TradingState(
            traderData="", timestamp=timestamp, listings={}, 
            order_depths=order_depths, own_trades={}, market_trades={}, 
            position=position.copy(), observations={}
        )
        
        orders, _, _ = trader.run(state)
        
        for product, product_orders in orders.items():
            od = order_depths[product]
            for order in product_orders:
                qty_to_fill = order.quantity
                
                if qty_to_fill > 0: 
                    # Simulating Passive Limit Buys:
                    # If we are bidding at or above the market's best bid, assume we get filled by retail flow
                    best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
                    if best_bid and order.price >= best_bid:
                        fill_qty = min(qty_to_fill, 5) # Assume we get a conservative fill of 5 lots
                        cash -= (fill_qty * order.price)
                        position[product] += fill_qty
                        
                elif qty_to_fill < 0: 
                    # Simulating Passive Limit Sells:
                    best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
                    if best_ask and order.price <= best_ask:
                        fill_qty = min(abs(qty_to_fill), 5)
                        cash += (fill_qty * order.price)
                        position[product] -= fill_qty

    # Mark to Market EOD
    mtm_pnl = cash
    for product, qty in position.items():
        if qty != 0 and product in mid_prices:
            mtm_pnl += (qty * mid_prices[product])
            
    return mtm_pnl

if __name__ == "__main__":
    
    # Notice we tightened delta_threshold for WLS
    ablation_configs = {
        "Baseline (Original)": {
            "use_wls": False, "use_inventory_penalty": False, 
            "use_delta_hedge": False, "delta_threshold": 0.0, "risk_aversion": 0.0
        },
        "Baseline + WLS": {
            "use_wls": True, "use_inventory_penalty": False, 
            "use_delta_hedge": False, "delta_threshold": 0.0, "risk_aversion": 0.0
        },
        "Baseline + Inventory Penalty": {
            "use_wls": False, "use_inventory_penalty": True, 
            "use_delta_hedge": False, "delta_threshold": 0.0, "risk_aversion": 0.005
        },
        "Baseline + Deadband Hedging": {
            "use_wls": False, "use_inventory_penalty": False, 
            "use_delta_hedge": True, "delta_threshold": 15.0, "risk_aversion": 0.0
        },
        "All Upgrades Combined (Passive)": {
            "use_wls": True, "use_inventory_penalty": True, 
            "use_delta_hedge": True, "delta_threshold": 8.0, "risk_aversion": 0.005
        }
    }

    data_files = sorted(glob.glob("prices_round_3_day_*.csv"))
    if not data_files:
        print("ERROR: No 'prices_round_3_day_*.csv' files found in the current directory.")
        exit()

    results = {}

    for config_name, params in ablation_configs.items():
        print(f"\n--- Running Ablation: {config_name} ---")
        trader = Trader(**params)
        total_pnl = 0.0
        
        for file in data_files:
            df = pd.read_csv(file, sep=';')
            day_pnl = simulate_day(trader, df) 
            total_pnl += day_pnl
            print(f"    Day PnL: {day_pnl:,.2f}")
            
        results[config_name] = total_pnl

    print("\n" + "="*50)
    print("NEW ABLATION STUDY RESULTS (PASSIVE QUOTING)")
    print("="*50)
    for config, final_pnl in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{config:<35} | {final_pnl:>15,.2f}")