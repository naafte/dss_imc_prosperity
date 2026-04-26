import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

BASE = "C:/Users/sab06/dss_imc_prosperity/ROUND3_Algo"

# ── Load & concatenate all price data ────────────────────────────────────────
dfs = []
for day in [0, 1, 2]:
    df = pd.read_csv(f"{BASE}/prices_round_3_day_{day}.csv", sep=";")
    dfs.append(df)
prices = pd.concat(dfs, ignore_index=True)

# Global timestamp: each day has 1_000_000 ticks
prices["global_ts"] = prices["day"] * 1_000_000 + prices["timestamp"]

# ── Asset class split ─────────────────────────────────────────────────────────
VEV_products   = sorted([p for p in prices["product"].unique() if p.startswith("VEV_")])
UNDER_products = [p for p in prices["product"].unique() if not p.startswith("VEV_")]

vev   = prices[prices["product"].isin(VEV_products)].copy()
under = prices[prices["product"].isin(UNDER_products)].copy()

# ── Print summary stats ───────────────────────────────────────────────────────
for label, df, products in [
    ("UNDERLYING ASSETS", under, UNDER_products),
    ("VEV OPTIONS",       vev,   VEV_products),
]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Products : {products}")
    print(f"  Rows     : {len(df):,}")
    print(f"  Days     : {sorted(df['day'].unique())}")
    print()
    for prod in products:
        s = df[df["product"] == prod]["mid_price"]
        print(f"  {prod:<25}  mean={s.mean():.2f}  std={s.std():.2f}"
              f"  min={s.min():.2f}  max={s.max():.2f}  skew={s.skew():.3f}")

# ── Rolling volatility (100-tick window, per product) ────────────────────────
ROLL = 100

def add_rolling_vol(df, window=ROLL):
    out = []
    for prod, g in df.sort_values("global_ts").groupby("product"):
        g = g.copy()
        g["ret"]     = g["mid_price"].pct_change()
        g["roll_vol"] = g["ret"].rolling(window, min_periods=10).std() * np.sqrt(window)
        out.append(g)
    return pd.concat(out)

vev   = add_rolling_vol(vev)
under = add_rolling_vol(under)

# ── Colour palettes ───────────────────────────────────────────────────────────
import matplotlib.cm as cm

vev_colors   = {p: cm.plasma(i / max(len(VEV_products)-1, 1))
                for i, p in enumerate(VEV_products)}
under_colors = {"VELVETFRUIT_EXTRACT": "#2ecc71", "HYDROGEL_PACK": "#e74c3c"}

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 — Underlying Assets: price + volatility
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                            gridspec_kw={"hspace": 0.08})
fig1.patch.set_facecolor("#0f1117")
for ax in axes1:
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.spines[:].set_color("#3a3d50")

ax_p, ax_v = axes1

for prod in UNDER_products:
    g = under[under["product"] == prod].sort_values("global_ts")
    c = under_colors.get(prod, "white")
    ax_p.plot(g["global_ts"], g["mid_price"],  linewidth=0.9, color=c, label=prod)
    ax_v.plot(g["global_ts"], g["roll_vol"],   linewidth=0.9, color=c, alpha=0.85, label=prod)

ax_p.set_ylabel("Mid Price", fontsize=11, color="white")
ax_v.set_ylabel(f"Rolling Volatility\n(window={ROLL})", fontsize=11, color="white")
ax_v.set_xlabel("Global Timestamp", fontsize=11, color="white")

for ax in axes1:
    leg = ax.legend(facecolor="#252836", edgecolor="#3a3d50", labelcolor="white",
                    fontsize=9, loc="upper right")

# day boundary lines
for d in [1, 2]:
    for ax in axes1:
        ax.axvline(d * 1_000_000, color="#555577", linestyle="--", linewidth=0.8, alpha=0.7)

fig1.suptitle("Underlying Assets — Price & Volatility over Time",
              color="white", fontsize=14, y=0.97, fontweight="bold")
for ax in axes1:
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d50")

plt.savefig(f"{BASE}/eda_underlying.png", dpi=150, bbox_inches="tight",
            facecolor=fig1.get_facecolor())
print(f"\nSaved: eda_underlying.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 — VEV Options: price over time (all strikes on one panel)
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                            gridspec_kw={"hspace": 0.08})
fig2.patch.set_facecolor("#0f1117")
for ax in axes2:
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.spines[:].set_color("#3a3d50")

ax_p2, ax_v2 = axes2

for prod in VEV_products:
    g = vev[vev["product"] == prod].sort_values("global_ts")
    c = vev_colors[prod]
    ax_p2.plot(g["global_ts"], g["mid_price"],  linewidth=0.8, color=c, label=prod)
    ax_v2.plot(g["global_ts"], g["roll_vol"],   linewidth=0.8, color=c, alpha=0.85, label=prod)

ax_p2.set_ylabel("Mid Price", fontsize=11, color="white")
ax_v2.set_ylabel(f"Rolling Volatility\n(window={ROLL})", fontsize=11, color="white")
ax_v2.set_xlabel("Global Timestamp", fontsize=11, color="white")

for ax in axes2:
    leg = ax.legend(facecolor="#252836", edgecolor="#3a3d50", labelcolor="white",
                    fontsize=8, loc="upper right", ncol=2)

for d in [1, 2]:
    for ax in axes2:
        ax.axvline(d * 1_000_000, color="#555577", linestyle="--", linewidth=0.8, alpha=0.7)

fig2.suptitle("VEV Options — Price & Volatility over Time",
              color="white", fontsize=14, y=0.97, fontweight="bold")

plt.savefig(f"{BASE}/eda_vev_options.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
print(f"Saved: eda_vev_options.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 — VEV Options: price grid (one subplot per strike)
# ─────────────────────────────────────────────────────────────────────────────
n = len(VEV_products)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig3, axes3 = plt.subplots(nrows, ncols, figsize=(16, nrows * 3),
                            sharex=True, gridspec_kw={"hspace": 0.4, "wspace": 0.3})
fig3.patch.set_facecolor("#0f1117")
axes3_flat = axes3.flatten()

for i, prod in enumerate(VEV_products):
    ax = axes3_flat[i]
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#3a3d50")

    g = vev[vev["product"] == prod].sort_values("global_ts")
    c = vev_colors[prod]

    ax.plot(g["global_ts"], g["mid_price"], linewidth=0.7, color=c)
    ax.fill_between(g["global_ts"], g["mid_price"], alpha=0.15, color=c)

    strike = int(prod.split("_")[1])
    ax.set_title(f"Strike {strike}", color="white", fontsize=9, pad=3)
    ax.yaxis.set_major_locator(MaxNLocator(4))

    for d in [1, 2]:
        ax.axvline(d * 1_000_000, color="#555577", linestyle="--", linewidth=0.6, alpha=0.7)

    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d50")

# hide unused subplots
for j in range(i + 1, len(axes3_flat)):
    axes3_flat[j].set_visible(False)

fig3.suptitle("VEV Options — Mid Price by Strike",
              color="white", fontsize=14, y=1.01, fontweight="bold")

plt.savefig(f"{BASE}/eda_vev_grid.png", dpi=150, bbox_inches="tight",
            facecolor=fig3.get_facecolor())
print(f"Saved: eda_vev_grid.png")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 — Volatility smile (avg vol per strike, per day)
# ─────────────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 5))
fig4.patch.set_facecolor("#0f1117")
ax4.set_facecolor("#1a1d2e")
ax4.tick_params(colors="white")
ax4.spines[:].set_color("#3a3d50")

day_colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}

for day in [0, 1, 2]:
    strikes, vols = [], []
    for prod in VEV_products:
        g = vev[(vev["product"] == prod) & (vev["day"] == day)]
        if len(g) > 0:
            avg_vol = g["roll_vol"].dropna().mean()
            if not np.isnan(avg_vol):
                strikes.append(int(prod.split("_")[1]))
                vols.append(avg_vol)
    if strikes:
        ax4.plot(strikes, vols, "o-", color=day_colors[day], linewidth=1.8,
                 markersize=6, label=f"Day {day}")

ax4.set_xlabel("Strike Price", fontsize=11, color="white")
ax4.set_ylabel("Avg Rolling Volatility", fontsize=11, color="white")
ax4.set_title("Implied Volatility Smile — VEV Options by Day",
              color="white", fontsize=13, fontweight="bold")
ax4.legend(facecolor="#252836", edgecolor="#3a3d50", labelcolor="white", fontsize=10)
ax4.xaxis.label.set_color("white")

plt.savefig(f"{BASE}/eda_vol_smile.png", dpi=150, bbox_inches="tight",
            facecolor=fig4.get_facecolor())
print(f"Saved: eda_vol_smile.png")

plt.show()
print("\nEDA complete.")
