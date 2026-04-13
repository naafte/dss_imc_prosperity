import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prices1 = pd.read_csv("C:\\Users\\sab06\\IMC Prosperity Tutorial\\prices_round_0_day_-1.csv", sep=";")
prices2 = pd.read_csv("C:\\Users\\sab06\\IMC Prosperity Tutorial\\prices_round_0_day_-2.csv", sep=";")
trades1 = pd.read_csv("C:\\Users\\sab06\\IMC Prosperity Tutorial\\trades_round_0_day_-1.csv", sep=";")
trades2 = pd.read_csv("C:\\Users\\sab06\\IMC Prosperity Tutorial\\trades_round_0_day_-2.csv", sep=";")

# mean_prices1 = [prices1["bid_price_1"].mean(), 
#                 prices1["bid_price_2"].mean(), 
#                 prices1["ask_price_1"].mean(), 
#                 prices1["ask_price_2"].mean()]
# mean_prices2 = [prices2["bid_price_1"].mean(), 
#                 prices2["bid_price_2"].mean(), 
#                 prices2["ask_price_1"].mean(), 
#                 prices2["ask_price_2"].mean()]
# median_prices1 = [prices1["bid_price_1"].median(), 
#                 prices1["bid_price_2"].median(), 
#                 prices1["ask_price_1"].median(), 
#                 prices1["ask_price_2"].median()]
# median_prices2 = [prices2["bid_price_1"].median(), 
#                 prices2["bid_price_2"].median(), 
#                 prices2["ask_price_1"].median(), 
#                 prices2["ask_price_2"].median()]
# print(mean_prices1)
# print(mean_prices2)
# print(median_prices1)
# print(median_prices2)


# Filter products
prices1_tomatoes = prices1[prices1["product"] == "TOMATOES"].sort_values("timestamp")
prices1_emeralds = prices1[prices1["product"] == "EMERALDS"].sort_values("timestamp")

step = 20

# Sampled data
tomatoes_sample = prices1_tomatoes.iloc[::step]
emeralds_sample = prices1_emeralds.iloc[::step]

# Means (use FULL data)
t_bid_mean = prices1_tomatoes["bid_price_1"].mean()
t_ask_mean = prices1_tomatoes["ask_price_1"].mean()

e_bid_mean = prices1_emeralds["bid_price_1"].mean()
e_ask_mean = prices1_emeralds["ask_price_1"].mean()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- TOMATOES ---
axes[0].plot(tomatoes_sample["timestamp"], tomatoes_sample["bid_price_1"], label="bid")
axes[0].plot(tomatoes_sample["timestamp"], tomatoes_sample["ask_price_1"], label="ask")

# Mean lines
axes[0].axhline(t_bid_mean, linestyle="--", label="bid mean")
axes[0].axhline(t_ask_mean, linestyle="--", label="ask mean")

axes[0].set_title("TOMATOES (sampled)")
axes[0].legend()

# --- EMERALDS ---
axes[1].plot(emeralds_sample["timestamp"], emeralds_sample["bid_price_1"], label="bid")
axes[1].plot(emeralds_sample["timestamp"], emeralds_sample["ask_price_1"], label="ask")

# Mean lines
axes[1].axhline(e_bid_mean, linestyle="--", label="bid mean")
axes[1].axhline(e_ask_mean, linestyle="--", label="ask mean")

axes[1].set_title("EMERALDS (sampled)")
axes[1].legend()

axes[1].set_xlabel("timestamp")

plt.tight_layout()
plt.show()