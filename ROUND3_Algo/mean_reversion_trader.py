from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle
import statistics

# Products and their position limits
PRODUCTS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
}

# Per-product tuning
# HYDROGEL_PACK: mean ~9991, std ~32, spread ~16
# VELVETFRUIT_EXTRACT: mean ~5250, std ~16, spread narrow
WINDOW = 100          # rolling window for mean/std
Z_AGGRESSIVE = 1.5   # z-score threshold to trade aggressively
Z_PASSIVE = 0.5      # z-score threshold to start passive quoting
PASSIVE_SIZE = 15     # max passive quote size per side


def _mid(od: OrderDepth):
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0


def _take_orders(od: OrderDepth, fair: float, pos: int, limit: int) -> List[Order]:
    """Take all orders that cross our fair value estimate."""
    orders = []
    buy_cap = limit - pos
    sell_cap = limit + pos

    # Buy any ask strictly below fair value
    if buy_cap > 0:
        for ask in sorted(od.sell_orders):
            if ask >= fair:
                break
            vol = min(-od.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order("", ask, vol))
                buy_cap -= vol
            if buy_cap <= 0:
                break

    # Sell any bid strictly above fair value
    if sell_cap > 0:
        for bid in sorted(od.buy_orders, reverse=True):
            if bid <= fair:
                break
            vol = min(od.buy_orders[bid], sell_cap)
            if vol > 0:
                orders.append(Order("", bid, -vol))
                sell_cap -= vol
            if sell_cap <= 0:
                break

    return orders, buy_cap, sell_cap


def _trade_product(product: str, od: OrderDepth, pos: int, hist: List[float]) -> List[Order]:
    limit = LIMITS[product]
    orders: List[Order] = []

    mid = _mid(od)
    if mid is None:
        return orders

    hist.append(mid)
    if len(hist) > WINDOW:
        del hist[:-WINDOW]

    if len(hist) < 20:
        return orders

    mean = statistics.mean(hist)
    std = max(statistics.stdev(hist), 0.5)
    z = (mid - mean) / std

    fair = mean  # fair value is the rolling mean

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    buy_cap = limit - pos
    sell_cap = limit + pos

    # ── 1. Take all orders that beat fair value ──────────────────────────────
    # Buy asks below fair
    for ask in sorted(od.sell_orders):
        if ask >= fair or buy_cap <= 0:
            break
        vol = min(-od.sell_orders[ask], buy_cap)
        if vol > 0:
            orders.append(Order(product, ask, vol))
            buy_cap -= vol

    # Sell bids above fair
    for bid in sorted(od.buy_orders, reverse=True):
        if bid <= fair or sell_cap <= 0:
            break
        vol = min(od.buy_orders[bid], sell_cap)
        if vol > 0:
            orders.append(Order(product, bid, -vol))
            sell_cap -= vol

    # ── 2. Aggressive mean-reversion on extreme z-score ─────────────────────
    if z < -Z_AGGRESSIVE and buy_cap > 0:
        # Price is very low → buy more aggressively (up to fair value)
        for ask in sorted(od.sell_orders):
            if ask > fair or buy_cap <= 0:
                break
            vol = min(-od.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order(product, ask, vol))
                buy_cap -= vol

    elif z > Z_AGGRESSIVE and sell_cap > 0:
        # Price is very high → sell more aggressively (down to fair value)
        for bid in sorted(od.buy_orders, reverse=True):
            if bid < fair or sell_cap <= 0:
                break
            vol = min(od.buy_orders[bid], sell_cap)
            if vol > 0:
                orders.append(Order(product, bid, -vol))
                sell_cap -= vol

    # ── 3. Passive quoting around fair value ────────────────────────────────
    # Post on the side that benefits from mean reversion
    fair_int = round(fair)

    if z < -Z_PASSIVE and buy_cap > 0:
        # Price below mean → passively bid just inside/at best ask
        passive_bid = min(best_ask - 1, fair_int)
        vol = min(buy_cap, PASSIVE_SIZE)
        if vol > 0 and passive_bid >= best_bid:
            orders.append(Order(product, passive_bid, vol))

    if z > Z_PASSIVE and sell_cap > 0:
        # Price above mean → passively ask just inside/at best bid
        passive_ask = max(best_bid + 1, fair_int)
        vol = min(sell_cap, PASSIVE_SIZE)
        if vol > 0 and passive_ask <= best_ask:
            orders.append(Order(product, passive_ask, -vol))

    return orders


class Trader:

    def run(self, state: TradingState):
        # Restore persisted history
        if state.traderData:
            try:
                ts = jsonpickle.decode(state.traderData)
            except Exception:
                ts = {}
        else:
            ts = {}

        result: Dict[str, List[Order]] = {}

        for product in PRODUCTS:
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            hist = ts.setdefault(f"{product}_hist", [])
            result[product] = _trade_product(product, od, pos, hist)

        traderData = jsonpickle.encode(ts)
        return result, 0, traderData
