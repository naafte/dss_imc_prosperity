import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Problem Setup ──────────────────────────────────────────────────────────────
FAIR_PRICE = 920
RESERVE_MIN, RESERVE_MAX, STEP = 670, 920, 5
reserves = np.arange(RESERVE_MIN, RESERVE_MAX + STEP, STEP)   # 51 values
N = len(reserves)   # number of counterparties (equal weight)

# ── Profit functions ───────────────────────────────────────────────────────────
def profit_b2(b2, b1, avg_b2):
    """Expected total profit from second bid across all counterparties."""
    # Counterparties eligible for b2: those with b1 < reserve <= b2
    eligible = reserves[(reserves > b1) & (reserves <= b2)]
    n_eligible = len(eligible)
    if n_eligible == 0:
        return 0.0
    margin = FAIR_PRICE - b2
    if margin <= 0:
        return 0.0
    if b2 > avg_b2:
        return n_eligible * margin
    else:
        # Penalised PNL per trade
        penalised = (FAIR_PRICE - avg_b2)**3 / (FAIR_PRICE - b2)**2
        return n_eligible * penalised

# ── Grid of b2 values ──────────────────────────────────────────────────────────
b2_vals = np.arange(RESERVE_MIN + STEP, FAIR_PRICE + STEP, STEP)  # 675..920

b1 = 750   # assumed first bid (conservative)
avg_b2_scenarios = {
    "avg_b2 = 860 (low)": 860,
    "avg_b2 = 880 (mid)": 880,
    "avg_b2 = 900 (high)": 900,
}

# ── Summary statistics for each scenario ──────────────────────────────────────
print("=" * 65)
print(f"Reserve prices: uniform on [{RESERVE_MIN}, {RESERVE_MAX}] step {STEP}  (N={N})")
print(f"Fair price: {FAIR_PRICE}   Assumed b1: {b1}")
print("=" * 65)

optimal_b2s = {}
for label, avg_b2 in avg_b2_scenarios.items():
    profits = [profit_b2(b2, b1, avg_b2) for b2 in b2_vals]
    best_idx = np.argmax(profits)
    best_b2 = b2_vals[best_idx]
    best_profit = profits[best_idx]
    optimal_b2s[label] = (best_b2, best_profit, profits)
    print(f"\n{label}")
    print(f"  Optimal b2   : {best_b2}")
    print(f"  Expected PNL : {best_profit:.2f}")
    # break-even: profit at avg_b2 threshold
    be = profit_b2(avg_b2 + STEP, b1, avg_b2)
    print(f"  PNL at b2=avg_b2+5 ({avg_b2+STEP}): {be:.2f}  (just above threshold)")

# ── Reserve price distribution stats ──────────────────────────────────────────
print("\n" + "=" * 65)
print("Reserve-price distribution (uniform 670-920 step 5):")
print(f"  Mean   : {reserves.mean():.1f}")
print(f"  Median : {np.median(reserves):.1f}")
print(f"  Std    : {reserves.std():.1f}")
print(f"  Count  : {N}")
print("=" * 65)

# ── Sensitivity: optimal b2 vs avg_b2 assumption ──────────────────────────────
avg_b2_range = np.arange(RESERVE_MIN + STEP, FAIR_PRICE, STEP)
opt_b2_curve = []
for avg in avg_b2_range:
    profits = [profit_b2(b2, b1, avg) for b2 in b2_vals]
    opt_b2_curve.append(b2_vals[np.argmax(profits)])

# ── Plotting ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("IMC Prosperity – Second Bid Analysis\n"
             f"Fair price = {FAIR_PRICE}, b1 = {b1}, Reserves uniform [{RESERVE_MIN}–{RESERVE_MAX}] step {STEP}",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

colors = ["#2196F3", "#FF5722", "#4CAF50"]

# Panel 1 – Reserve price distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(reserves, np.ones(N) / N, width=4, color="#9C27B0", alpha=0.8, edgecolor="white")
ax1.set_title("Reserve Price Distribution (uniform)")
ax1.set_xlabel("Reserve Price")
ax1.set_ylabel("Probability")
ax1.axvline(reserves.mean(), color="red", ls="--", label=f"Mean={reserves.mean():.0f}")
ax1.legend(fontsize=9)

# Panel 2 – Expected PNL vs b2 for each avg_b2 scenario
ax2 = fig.add_subplot(gs[0, 1])
for (label, avg_b2), color in zip(avg_b2_scenarios.items(), colors):
    best_b2, best_profit, profits = optimal_b2s[label]
    ax2.plot(b2_vals, profits, color=color, label=label, lw=2)
    ax2.axvline(avg_b2, color=color, ls=":", alpha=0.5)
    ax2.scatter([best_b2], [best_profit], color=color, s=80, zorder=5)
    ax2.annotate(f"  b2*={best_b2}", (best_b2, best_profit), fontsize=8, color=color)
ax2.set_title("Expected PNL from 2nd Bid vs b2")
ax2.set_xlabel("Second Bid (b2)")
ax2.set_ylabel("Expected PNL")
ax2.legend(fontsize=8)
ax2.axhline(0, color="black", lw=0.5)

# Panel 3 – Penalised vs unpenalised PNL per trade (avg_b2=880)
ax3 = fig.add_subplot(gs[1, 0])
avg_b2_ref = 880
full_margins = np.where(b2_vals <= FAIR_PRICE, FAIR_PRICE - b2_vals, 0).astype(float)
penalised = np.where(
    b2_vals < avg_b2_ref,
    (FAIR_PRICE - avg_b2_ref)**3 / np.maximum(FAIR_PRICE - b2_vals, 1)**2,
    full_margins
)
ax3.plot(b2_vals, full_margins, "--", color="gray", label="Full margin (920−b2)", lw=1.5)
ax3.plot(b2_vals, penalised, color="#FF5722", label=f"Effective PNL/trade (avg_b2={avg_b2_ref})", lw=2)
ax3.axvline(avg_b2_ref, color="navy", ls=":", label=f"avg_b2={avg_b2_ref}")
ax3.fill_between(b2_vals, penalised, full_margins,
                 where=(b2_vals < avg_b2_ref), alpha=0.15, color="red", label="Penalty region")
ax3.set_title("Per-Trade PNL: Penalised vs Full (avg_b2=880)")
ax3.set_xlabel("b2")
ax3.set_ylabel("PNL per trade")
ax3.legend(fontsize=8)

# Panel 4 – Optimal b2 sensitivity to avg_b2 assumption
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(avg_b2_range, opt_b2_curve, color="#2196F3", lw=2)
ax4.plot(avg_b2_range, avg_b2_range, "k--", lw=1, label="b2 = avg_b2 (diagonal)")
ax4.fill_between(avg_b2_range, opt_b2_curve, avg_b2_range,
                 where=(np.array(opt_b2_curve) >= avg_b2_range),
                 alpha=0.15, color="green", label="opt b2 > avg_b2 (safe zone)")
ax4.set_title("Optimal b2 vs Assumed avg_b2")
ax4.set_xlabel("Assumed avg_b2")
ax4.set_ylabel("Optimal b2")
ax4.legend(fontsize=8)
# Mark the three scenarios
for (label, avg_b2), color in zip(avg_b2_scenarios.items(), colors):
    best_b2 = optimal_b2s[label][0]
    ax4.scatter([avg_b2], [best_b2], color=color, s=80, zorder=5)

plt.savefig(
    "C:/Users/sab06/IMC Prosperity/ROUND3/ROUND3_Manual/bid2_analysis.png",
    dpi=150, bbox_inches="tight"
)
print("\nPlot saved -> bid2_analysis.png")
plt.show()
