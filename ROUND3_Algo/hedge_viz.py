"""
Visualise how much VELVETFRUIT_EXTRACT the delta hedge consumes over time.

Two views:
  1. Per-strike delta over time (how each option's hedge requirement evolves)
  2. Simulated aggregate hedge position, using a lightweight replay of the
     market-making/taking logic to estimate realistic option holdings at every tick.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import warnings
warnings.filterwarnings("ignore")

BASE = "C:/Users/sab06/dss_imc_prosperity/ROUND3_Algo"

# ── Parameters (must match trader.py) ────────────────────────────────────────
SIGMA         = 0.01
TTE_AT_START  = 5          # TTE at timestamp=0 for the actual Round 3 submission
TICKS_PER_DAY = 1_000_000
VEV_STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
LIM_VOUCHER   = 300
LIM_UNDER     = 200
TAKE_EDGE     = 1.5
MM_HALF       = 2
MM_SIZE       = 8

# ── BS functions ──────────────────────────────────────────────────────────────
def ncdf(x):
    return 0.5 * math.erfc(-x * 0.7071067811865476)

def bs_call(S, K, T, sigma):
    if T <= 1e-9: return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    return S * ncdf(d1) - K * ncdf(d1 - sigma * sqT)

def bs_delta(S, K, T, sigma):
    if T <= 1e-9: return 1.0 if S > K else 0.0
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    return ncdf(d1)

# ── Load data ─────────────────────────────────────────────────────────────────
dfs = []
for day in [0, 1, 2]:
    df = pd.read_csv(f"{BASE}/prices_round_3_day_{day}.csv", sep=";")
    dfs.append(df)
prices = pd.concat(dfs, ignore_index=True)
prices["global_ts"] = prices["day"] * TICKS_PER_DAY + prices["timestamp"]

# Underlying price series
under = (
    prices[prices["product"] == "VELVETFRUIT_EXTRACT"]
    .sort_values("global_ts")[["global_ts", "mid_price"]]
    .set_index("global_ts")
    .rename(columns={"mid_price": "S"})
)

# Option mid prices keyed by (global_ts, strike)
opt = prices[prices["product"].str.startswith("VEV_")].copy()
opt["strike"] = opt["product"].str.split("_").str[1].astype(int)
opt_pivot = opt.pivot_table(index="global_ts", columns="strike",
                             values="mid_price", aggfunc="first")

# Align indices
ts_common = under.index.intersection(opt_pivot.index)
S_series  = under.loc[ts_common, "S"]
opt_pivot = opt_pivot.loc[ts_common]

# ── TTE at each tick (using historical TTE: day0 starts at TTE=8) ─────────────
# historical_day = global_ts // TICKS_PER_DAY
# TTE_hist = 8 - historical_day - (global_ts % TICKS_PER_DAY) / TICKS_PER_DAY
tte_series = np.maximum(
    8 - ts_common // TICKS_PER_DAY - (ts_common % TICKS_PER_DAY) / TICKS_PER_DAY,
    1e-9
)

# ── 1. Per-strike delta over time ─────────────────────────────────────────────
print("Computing per-strike deltas...")
delta_df = pd.DataFrame(index=ts_common)
for k in VEV_STRIKES:
    if k not in opt_pivot.columns:
        continue
    delta_df[k] = [
        bs_delta(S_series.iloc[i], k, tte_series[i], SIGMA)
        for i in range(len(ts_common))
    ]

# ── 2. Simulate option positions via lightweight replay ───────────────────────
# Logic: at each tick, if market ask < BS_fv - TAKE_EDGE → buy (up to cap).
#        if market bid > BS_fv + TAKE_EDGE → sell (up to cap).
#        We track running position per strike and compute total hedge needed.
print("Simulating option positions...")
positions = {k: 0 for k in VEV_STRIKES}
hedge_ts   = []
hedge_vals = []

for i, ts in enumerate(ts_common):
    S   = float(S_series.iloc[i])
    tte = float(tte_series[i])

    total_delta = 0.0
    for k in VEV_STRIKES:
        if k not in opt_pivot.columns:
            continue
        mid = opt_pivot.loc[ts, k]
        if pd.isna(mid):
            continue
        fv  = bs_call(S, k, tte, SIGMA)
        pos = positions[k]

        # Approximate bid/ask from mid (half-tick spread is usually 0.5–1)
        approx_ask = mid + 0.5
        approx_bid = mid - 0.5

        # Take mis-priced
        if approx_ask < fv - TAKE_EDGE and pos < LIM_VOUCHER:
            buy = min(5, LIM_VOUCHER - pos)
            positions[k] += buy
        elif approx_bid > fv + TAKE_EDGE and pos > -LIM_VOUCHER:
            sell = min(5, LIM_VOUCHER + pos)
            positions[k] -= sell

        d = bs_delta(S, k, tte, SIGMA)
        total_delta += positions[k] * d

    # Hedge = -total_delta, clamped to ±LIM_UNDER
    hedge = max(-LIM_UNDER, min(LIM_UNDER, -total_delta))
    hedge_ts.append(ts)
    hedge_vals.append(hedge)

hedge_series = pd.Series(hedge_vals, index=hedge_ts)

# ── Colours ───────────────────────────────────────────────────────────────────
pal = {k: cm.plasma(i / (len(VEV_STRIKES) - 1)) for i, k in enumerate(VEV_STRIKES)}

# ── Figure 1: per-strike delta over time ──────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(14, 6))
fig1.patch.set_facecolor("#0f1117")
ax1.set_facecolor("#1a1d2e")
ax1.tick_params(colors="white")
ax1.spines[:].set_color("#3a3d50")

for k in VEV_STRIKES:
    if k not in delta_df.columns:
        continue
    ax1.plot(delta_df.index, delta_df[k], linewidth=0.8,
             color=pal[k], label=f"K={k}")

for d in [1, 2]:
    ax1.axvline(d * TICKS_PER_DAY, color="#555577", linestyle="--",
                linewidth=0.8, alpha=0.7)

ax1.set_xlabel("Global Timestamp", color="white", fontsize=11)
ax1.set_ylabel("BS Delta (per 1 unit of option)", color="white", fontsize=11)
ax1.set_title("BS Delta per Strike over Time",
              color="white", fontsize=13, fontweight="bold")
leg = ax1.legend(facecolor="#252836", edgecolor="#3a3d50", labelcolor="white",
                 fontsize=8, loc="center right", ncol=2)
fig1.tight_layout()
fig1.savefig(f"{BASE}/hedge_delta_per_strike.png", dpi=150,
             bbox_inches="tight", facecolor=fig1.get_facecolor())
print("Saved: hedge_delta_per_strike.png")

# ── Figure 2: simulated aggregate hedge position over time ────────────────────
fig2, axes = plt.subplots(3, 1, figsize=(14, 10),
                           gridspec_kw={"hspace": 0.35})
fig2.patch.set_facecolor("#0f1117")

# Panel A: hedge position
ax_h = axes[0]
ax_h.set_facecolor("#1a1d2e")
ax_h.tick_params(colors="white")
ax_h.spines[:].set_color("#3a3d50")

ax_h.plot(hedge_series.index, hedge_series.values,
          color="#00d4ff", linewidth=0.9, label="Hedge position")
ax_h.axhline(0,    color="#888", linestyle="--", linewidth=0.6, alpha=0.6)
ax_h.axhline( LIM_UNDER, color="#e74c3c", linestyle=":", linewidth=0.8,
              alpha=0.7, label=f"+{LIM_UNDER} limit")
ax_h.axhline(-LIM_UNDER, color="#e74c3c", linestyle=":", linewidth=0.8, alpha=0.7)
for d in [1, 2]:
    ax_h.axvline(d * TICKS_PER_DAY, color="#555577", linestyle="--",
                 linewidth=0.8, alpha=0.7)
ax_h.set_ylabel("VELVETFRUIT_EXTRACT\nhedge position", color="white", fontsize=10)
ax_h.set_title("Simulated Delta Hedge Position over Time",
               color="white", fontsize=12, fontweight="bold")
ax_h.legend(facecolor="#252836", edgecolor="#3a3d50", labelcolor="white",
            fontsize=9, loc="upper right")

# Panel B: underlying price (context)
ax_s = axes[1]
ax_s.set_facecolor("#1a1d2e")
ax_s.tick_params(colors="white")
ax_s.spines[:].set_color("#3a3d50")
ax_s.plot(S_series.index, S_series.values,
          color="#2ecc71", linewidth=0.8)
ax_s.axhline(5250, color="#888", linestyle="--", linewidth=0.6, alpha=0.6)
for d in [1, 2]:
    ax_s.axvline(d * TICKS_PER_DAY, color="#555577", linestyle="--",
                 linewidth=0.8, alpha=0.7)
ax_s.set_ylabel("VELVETFRUIT_EXTRACT\nmid price", color="white", fontsize=10)
ax_s.set_title("Underlying Price (for context)", color="white", fontsize=11)

# Panel C: per-strike simulated positions
ax_p = axes[2]
ax_p.set_facecolor("#1a1d2e")
ax_p.tick_params(colors="white")
ax_p.spines[:].set_color("#3a3d50")

# Re-run simulation recording per-strike positions at each tick
pos_history = {k: [] for k in VEV_STRIKES}
positions2  = {k: 0 for k in VEV_STRIKES}

for i, ts in enumerate(ts_common):
    S   = float(S_series.iloc[i])
    tte = float(tte_series[i])
    for k in VEV_STRIKES:
        if k not in opt_pivot.columns:
            pos_history[k].append(0)
            continue
        mid = opt_pivot.loc[ts, k]
        if pd.isna(mid):
            pos_history[k].append(positions2[k])
            continue
        fv  = bs_call(S, k, tte, SIGMA)
        pos = positions2[k]
        approx_ask = mid + 0.5
        approx_bid = mid - 0.5
        if approx_ask < fv - TAKE_EDGE and pos < LIM_VOUCHER:
            positions2[k] += min(5, LIM_VOUCHER - pos)
        elif approx_bid > fv + TAKE_EDGE and pos > -LIM_VOUCHER:
            positions2[k] -= min(5, LIM_VOUCHER + pos)
        pos_history[k].append(positions2[k])

for k in VEV_STRIKES:
    ax_p.plot(ts_common, pos_history[k], linewidth=0.7,
              color=pal[k], label=f"K={k}", alpha=0.85)

for d in [1, 2]:
    ax_p.axvline(d * TICKS_PER_DAY, color="#555577", linestyle="--",
                 linewidth=0.8, alpha=0.7)
ax_p.axhline(0, color="#888", linestyle="--", linewidth=0.5, alpha=0.5)
ax_p.set_ylabel("Option position (units)", color="white", fontsize=10)
ax_p.set_xlabel("Global Timestamp", color="white", fontsize=11)
ax_p.set_title("Simulated Option Positions by Strike", color="white", fontsize=11)
ax_p.legend(facecolor="#252836", edgecolor="#3a3d50", labelcolor="white",
            fontsize=7, loc="upper right", ncol=2)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d50")
    ax.yaxis.label.set_color("white")

fig2.savefig(f"{BASE}/hedge_simulation.png", dpi=150,
             bbox_inches="tight", facecolor=fig2.get_facecolor())
print("Saved: hedge_simulation.png")

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\nHedge position summary:")
print(f"  Mean   : {hedge_series.mean():.1f}")
print(f"  Std    : {hedge_series.std():.1f}")
print(f"  Min    : {hedge_series.min():.1f}")
print(f"  Max    : {hedge_series.max():.1f}")
print(f"  % at ±limit: {(hedge_series.abs() >= LIM_UNDER).mean()*100:.1f}%")

plt.show()
print("\nDone.")
