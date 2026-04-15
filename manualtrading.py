bids2 = {20: 43000, 19: 17000, 18: 6000, 17: 5000, 16: 10000, 15: 5000, 14: 10000, 13:7000}
asks2 = {12: 20000, 13: 25000, 14: 35000, 15: 6000, 16: 5000, 17: 0, 18: 10000, 19:12000}

bids1 = {30: 30000, 29: 5000, 28: 12000, 27: 28000}
asks1 = {28: 40000, 31: 20000, 32: 20000, 33: 30000}


class Optimizer:
    def __init__(self, bids, asks, trade_fee, sell_price):
        self.bids = bids
        self.asks = asks
        self.trade_fee = trade_fee
        self.sell_price = sell_price

        self.bid_vals = list(self.bids.keys())
        self.ask_vals = list(self.asks.keys())
        self.bid_vals.sort(reverse=True)
        self.ask_vals.sort()

    def clearing_price(self,price, volume):
        best_vol, best_cv = 0, 0
        for cv in range(min(min(self.ask_vals), min(self.bid_vals)), max(max(self.ask_vals), max(self.bid_vals)) + 1):
            bid_count = 0
            if price >= cv:
                bid_count += volume
            
            for bid, vol in self.bids.items():
                if bid >= cv:
                    bid_count += vol

            ask_count = 0
            for ask, vol in self.asks.items():
                if ask <= cv:
                    ask_count += vol

            if min(bid_count, ask_count) >= best_vol:
                best_vol, best_cv = min(bid_count, ask_count), cv
        return best_vol, best_cv

    def profit(self, price, volume):
        total_vol, cv = self.clearing_price(price, volume)
        for bid in self.bid_vals:
            if bid >= price:
                total_vol -= self.bids[bid]
                if total_vol <= 0:
                    break
        actual_vol = min(total_vol, volume)
        return actual_vol * (self.sell_price - cv - self.trade_fee*2)


    def optimize_profit(self):
        best = 0
        best_price = 0
        best_volume = 0
        for price in range(min(min(self.bid_vals), min(self.ask_vals)) - 1, max(max(self.bid_vals), max(self.ask_vals)) + 1):
            for volume in range(1, min(sum(self.bids.values()), sum(self.asks.values())) + 1):
                total_profit = self.profit(price, volume)
                if total_profit > best:
                    best = total_profit
                    best_price = price
                    best_volume = volume
        return best, best_price, best_volume

optimizer1 = Optimizer(bids1, asks1, 0, 30)
print(optimizer1.optimize_profit())

optimizer2 = Optimizer(bids2, asks2, 0.05, 20)
print(optimizer2.optimize_profit())