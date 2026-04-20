import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

LOG101 = np.log(101)

def compute_f(b):
    """Optimal Research × Scale product for remaining budget b (percent, 0-100)."""
    if b <= 0:
        return 0.0
    # Optimality condition: b = x_r + (1+x_r)*ln(1+x_r)
    def eq(x_r):
        return x_r + (1 + x_r) * np.log(1 + x_r) - b
    # eq(0) = 0 - b < 0; eq(b) = b + (1+b)*ln(1+b) - b = (1+b)*ln(1+b) > 0
    x_r = brentq(eq, 0, b)
    x_s = b - x_r
    R = 200_000 * np.log(1 + x_r) / LOG101
    S = 7 * x_s / 100
    return R * S

baseline = compute_f(100) * 0.1  # f(100) × 0.1 — PnL numerator when speed=0

speed_pcts = np.arange(1, 101)
breakeven = np.array([baseline / compute_f(100 - s) if (100 - s) > 0 else np.inf
                      for s in speed_pcts])

# Clip for plotting — values above 0.9 mean "never worth it"
breakeven_clipped = np.clip(breakeven, 0, 1.05)

fig, ax = plt.subplots(figsize=(10, 6))

# Shade region where investing 0 always beats this speed allocation
never_worth = breakeven > 0.9
ax.fill_between(speed_pcts, 0.9, breakeven_clipped,
                where=never_worth, color='#ff4444', alpha=0.25,
                label='Breakeven > 0.9 — investing 0 always wins here')
ax.fill_between(speed_pcts, 0, np.minimum(breakeven_clipped, 0.9),
                color='#4488ff', alpha=0.15,
                label='Speed investment worth it if multiplier ≥ breakeven')

ax.plot(speed_pcts, breakeven_clipped, color='#1144cc', linewidth=2.2,
        label='Breakeven multiplier')
ax.axhline(0.9, color='red',   linestyle='--', linewidth=1.4, label='Max multiplier (0.9)')
ax.axhline(0.1, color='green', linestyle='--', linewidth=1.4, label='Min multiplier (0.1)')

# Annotate where breakeven crosses 0.9
cross_idx = np.argmax(breakeven > 0.9)
cross_s = speed_pcts[cross_idx]
ax.axvline(cross_s, color='orange', linestyle=':', linewidth=1.5)
ax.annotate(f'Speed ≥ {cross_s}%\nnever worthwhile',
            xy=(cross_s, 0.9), xytext=(cross_s + 3, 0.75),
            arrowprops=dict(arrowstyle='->', color='orange'),
            fontsize=10, color='darkorange')

ax.set_xlabel('Speed Investment (%)', fontsize=12)
ax.set_ylabel('Breakeven Multiplier', fontsize=12)
ax.set_title('Breakeven Speed Multiplier vs Speed Investment\n'
             'Invest 0 in speed if your expected multiplier falls below this curve',
             fontsize=12)
ax.set_xlim(1, 100)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('breakeven_multipliers.png', dpi=150, bbox_inches='tight')
print(f"Saved. Crossover at speed = {cross_s}%")
print(f"Baseline PnL (speed=0): {baseline - 50000:.0f}")

# Print a few key values
for s in [10, 20, 40, 60, 75, 80, 90]:
    bm = baseline / compute_f(100 - s) if (100 - s) > 0 else float('inf')
    print(f"  Speed={s:3d}%  breakeven multiplier = {bm:.3f}  {'(never worth it)' if bm > 0.9 else ''}")
