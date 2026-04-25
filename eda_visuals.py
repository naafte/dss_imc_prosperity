import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy import stats

base = r"C:\Users\sab06\IMC Prosperity\ROUND2\ROUND2_Algo"
fig_dir = r"C:\Users\sab06\IMC Prosperity\ROUND2\figures"
os.makedirs(fig_dir, exist_ok=True)

dfs = []
for day in [-1, 0, 1]:
    path = os.path.join(base, f"prices_round_2_day_{day}.csv")
    df = pd.read_csv(path, sep=';')
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df = df[df['mid_price'] > 0].copy()
df = df.sort_values(['product', 'day', 'timestamp']).reset_index(drop=True)

PEPPER = df[df['product'] == 'INTARIAN_PEPPER_ROOT'].copy()
OSMIUM = df[df['product'] == 'ASH_COATED_OSMIUM'].copy()

day_colors = {-1: '#e74c3c', 0: '#2ecc71', 1: '#3498db'}
day_labels = {-1: 'Day -1', 0: 'Day 0', 1: 'Day 1'}

def add_features(sub):
    sub = sub.copy()
    sub['ret'] = sub.groupby('day')['mid_price'].pct_change()
    sub['spread'] = sub['ask_price_1'] - sub['bid_price_1']
    return sub

PEPPER = add_features(PEPPER)
OSMIUM = add_features(OSMIUM)

# ── Figure 1: Mid-price over time ──────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Mid-Price Over Time", fontsize=15, fontweight='bold')

for day in [-1, 0, 1]:
    s = PEPPER[PEPPER['day'] == day]
    axes[0].plot(s['timestamp'], s['mid_price'], color=day_colors[day], lw=0.7, label=day_labels[day])
axes[0].set_title("INTARIAN_PEPPER_ROOT  —  Strong Upward Trend (+1000/day)", fontsize=11)
axes[0].set_ylabel("Mid Price")
axes[0].legend()
axes[0].grid(alpha=0.3)

for day in [-1, 0, 1]:
    s = OSMIUM[OSMIUM['day'] == day]
    axes[1].plot(s['timestamp'], s['mid_price'], color=day_colors[day], lw=0.7, label=day_labels[day])
axes[1].set_title("ASH_COATED_OSMIUM  —  Stationary, Range-Bound (~10,000)", fontsize=11)
axes[1].set_ylabel("Mid Price")
axes[1].set_xlabel("Timestamp")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'eda_1_midprice.png'), dpi=140, bbox_inches='tight')
plt.close()
print("saved fig 1")

# ── Figure 2: Spread distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Bid-Ask Spread Distribution", fontsize=14, fontweight='bold')

for ax, sub, title, c in zip(axes,
        [PEPPER, OSMIUM],
        ['INTARIAN_PEPPER_ROOT', 'ASH_COATED_OSMIUM'],
        ['#2980b9', '#e67e22']):
    spread = sub['spread'].dropna()
    ax.hist(spread, bins=25, color=c, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.axvline(spread.mean(), color='red', ls='--', lw=1.5, label=f"Mean = {spread.mean():.1f}")
    ax.axvline(spread.median(), color='black', ls=':', lw=1.5, label=f"Median = {spread.median():.1f}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Spread")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'eda_2_spread.png'), dpi=140, bbox_inches='tight')
plt.close()
print("saved fig 2")

# ── Figure 3: Return distributions ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Tick-Level Return Distributions", fontsize=14, fontweight='bold')

for ax, sub, title, c in zip(axes,
        [PEPPER, OSMIUM],
        ['INTARIAN_PEPPER_ROOT', 'ASH_COATED_OSMIUM'],
        ['#2980b9', '#e67e22']):
    ret = sub['ret'].dropna()
    ret = ret[(ret > ret.quantile(0.001)) & (ret < ret.quantile(0.999))]
    ax.hist(ret, bins=70, color=c, alpha=0.75, edgecolor='white', linewidth=0.3, density=True)
    x = np.linspace(ret.min(), ret.max(), 300)
    ax.plot(x, stats.norm.pdf(x, ret.mean(), ret.std()), 'k--', lw=1.5, label='Normal fit')
    kurt = ret.kurt()
    ax.set_title(f"{title}\nExcess Kurtosis = {kurt:.2f}", fontsize=10)
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'eda_3_returns.png'), dpi=140, bbox_inches='tight')
plt.close()
print("saved fig 3")

# ── Figure 4: Rolling volatility ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Rolling Price Volatility (window=100 ticks)", fontsize=14, fontweight='bold')

for ax, sub, title in zip(axes,
        [PEPPER, OSMIUM],
        ['INTARIAN_PEPPER_ROOT', 'ASH_COATED_OSMIUM']):
    for day in [-1, 0, 1]:
        s = sub[sub['day'] == day].copy().reset_index(drop=True)
        rv = s['mid_price'].rolling(100).std()
        ax.plot(s['timestamp'], rv, color=day_colors[day], lw=0.8, label=day_labels[day], alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price Std")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'eda_4_volatility.png'), dpi=140, bbox_inches='tight')
plt.close()
print("saved fig 4")

# ── Figure 5: Autocorrelation of returns ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Return Autocorrelation (lags 1-30)", fontsize=14, fontweight='bold')

for ax, sub, title, c in zip(axes,
        [PEPPER, OSMIUM],
        ['INTARIAN_PEPPER_ROOT', 'ASH_COATED_OSMIUM'],
        ['#2980b9', '#e67e22']):
    ret = sub['ret'].dropna()
    lags = list(range(1, 31))
    acs = [ret.autocorr(lag=l) for l in lags]
    bar_colors = [c if a < 0 else '#e74c3c' for a in acs]
    ax.bar(lags, acs, color=bar_colors, alpha=0.8, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    conf = 1.96 / np.sqrt(len(ret))
    ax.axhline(conf, color='gray', ls='--', lw=1, label=f'95% CI (+/-{conf:.3f})')
    ax.axhline(-conf, color='gray', ls='--', lw=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'eda_5_autocorr.png'), dpi=140, bbox_inches='tight')
plt.close()
print("saved fig 5")

# ── Figure 6: Order book depth ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Average Order Book Depth (Bid & Ask Levels)", fontsize=14, fontweight='bold')

for ax, sub, title in zip(axes,
        [PEPPER, OSMIUM],
        ['INTARIAN_PEPPER_ROOT', 'ASH_COATED_OSMIUM']):
    bid_vols = [sub[f'bid_volume_{i}'].mean() for i in [1, 2, 3]]
    ask_vols = [sub[f'ask_volume_{i}'].mean() for i in [1, 2, 3]]
    levels = ['Level 1', 'Level 2', 'Level 3']
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, bid_vols, w, label='Bid', color='#27ae60', alpha=0.8, edgecolor='white')
    ax.bar(x + w/2, ask_vols, w, label='Ask', color='#e74c3c', alpha=0.8, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Avg Volume")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'eda_6_depth.png'), dpi=140, bbox_inches='tight')
plt.close()
print("saved fig 6")

# ── Figure 7: Mid-price histogram per day ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Mid-Price Distribution Per Day", fontsize=14, fontweight='bold')

for row, (sub, pname) in enumerate(zip([PEPPER, OSMIUM], ['PEPPER_ROOT', 'ASH_OSMIUM'])):
    for col, day in enumerate([-1, 0, 1]):
        ax = axes[row][col]
        s = sub[sub['day'] == day]['mid_price']
        c = day_colors[day]
        ax.hist(s, bins=40, color=c, alpha=0.8, edgecolor='white', linewidth=0.4)
        ax.axvline(s.mean(), color='black', ls='--', lw=1.2, label=f'mean={s.mean():.0f}')
        ax.set_title(f"{pname} | Day {day}", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("Count")
        if row == 1:
            ax.set_xlabel("Mid Price")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'eda_7_price_hist.png'), dpi=140, bbox_inches='tight')
plt.close()
print("saved fig 7")

print("\nAll figures saved to:", fig_dir)
