import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N_TEAMS = 6000  # other teams (I am +1); ~6000 per Directions
BUDGET  = 50_000

# ── Core formulas ─────────────────────────────────────────────────────────────

R_ARR = np.array([
    200_000 * np.log(1 + r) / np.log(101) if r > 0 else 0.0
    for r in range(101)
])
S_ARR = np.array([7 * s / 100 for s in range(101)])

def speed_mult_vec(other_speeds):
    """Return array[101]: my speed multiplier for each integer v in 0..100."""
    sm = np.empty(101)
    for v in range(101):
        n_above = int(np.sum(other_speeds > v))
        rank    = n_above + 1
        N_total = len(other_speeds) + 1
        sm[v]   = 0.9 - 0.8 * (rank - 1) / (N_total - 1)
    return sm

def optimize(sm_arr):
    """
    Fast optimizer: Scale is linear in s, so for fixed (r, v) the optimal s
    is always a boundary value — reduces search from O(100^3) to O(100^2).
    """
    best_pnl = -np.inf
    best_r = best_s = best_v = 0
    r_idx = np.arange(101)

    for v in range(101):
        sm = sm_arr[v]
        if sm <= 0:
            continue
        max_rs = 100 - v
        if max_rs < 0:
            continue
        r_range = r_idx[:max_rs + 1]
        coeff  = R_ARR[r_range] * sm * (7 / 100) - 500
        s_opt  = np.clip(np.where(coeff > 0, max_rs - r_range, 0), 0, max_rs - r_range)
        gross  = R_ARR[r_range] * S_ARR[s_opt] * sm
        budget = BUDGET * (r_range + s_opt + v) / 100
        net    = gross - budget
        best_idx = int(np.argmax(net))
        if net[best_idx] > best_pnl:
            best_pnl = float(net[best_idx])
            best_r   = int(r_range[best_idx])
            best_s   = int(s_opt[best_idx])
            best_v   = v

    return best_r, best_s, best_v, best_pnl

# ── Distribution generators ───────────────────────────────────────────────────

def add_offbyone_spikes(arr, schelling_pts, spike_frac=0.05):
    arr = arr.copy()
    for pt in schelling_pts:
        idxs = np.where(arr == pt)[0]
        n_spike = int(len(idxs) * spike_frac)
        arr[idxs[:n_spike]] = min(pt + 1, 100)
    return arr

def gen_dist1():
    """GPT Bimodal: peaks at 5-20% and 50-80%, sparse 61-81%, very few 81-100%."""
    n_peak1  = int(N_TEAMS * 0.28)
    n_dead   = int(N_TEAMS * 0.05)
    n_peak2  = int(N_TEAMS * 0.52)
    n_sparse = int(N_TEAMS * 0.10)
    n_tail   = N_TEAMS - n_peak1 - n_dead - n_peak2 - n_sparse
    s_peak1  = np.clip(np.random.normal(12, 4, n_peak1), 5, 20)
    s_dead   = np.random.uniform(20, 50, n_dead)
    s_peak2  = np.clip(np.random.normal(56, 5, n_peak2), 50, 80)
    s_sparse = np.random.uniform(61, 81, n_sparse)
    s_tail   = np.random.uniform(81, 100, n_tail)
    raw = np.concatenate([s_peak1, s_dead, s_peak2, s_sparse, s_tail])
    return np.clip(np.round(raw).astype(int), 0, 100)

def gen_dist2():
    """Normal distribution centered at 33.33%, std~15, truncated to [0,100]."""
    raw = np.random.normal(loc=33.33, scale=15, size=N_TEAMS * 2)
    raw = raw[(raw >= 0) & (raw <= 100)][:N_TEAMS]
    arr = np.clip(np.round(raw).astype(int), 0, 100)
    return add_offbyone_spikes(arr, [33], spike_frac=0.10)

def gen_dist3():
    """Gemini: small 0-25%, very large 26-36%, large 37-46%, medium 47-58%, small 59%+."""
    segments = [
        (0,  26, 0.10),
        (26, 37, 0.35),
        (37, 47, 0.28),
        (47, 59, 0.23),
        (59, 100, 0.04),
    ]
    parts = []
    total = 0
    for i, (lo, hi, frac) in enumerate(segments):
        n = int(N_TEAMS * frac) if i < len(segments) - 1 else N_TEAMS - total
        total += n
        parts.append(np.random.uniform(lo, hi, n))
    arr = np.clip(np.round(np.concatenate(parts)).astype(int), 0, 100)
    return add_offbyone_spikes(arr, [25, 36, 46], spike_frac=0.08)

def gen_dist4():
    """Claude: ~20% at [0,5], ~45% at [10,25], ~22% at [25,40], ~8% outliers above 40%."""
    n1 = int(N_TEAMS * 0.20)   # 0-5%
    n2 = int(N_TEAMS * 0.05)   # 5-10% implied near-zero
    n3 = int(N_TEAMS * 0.45)   # 10-25%
    n4 = int(N_TEAMS * 0.22)   # 25-40%
    n5 = N_TEAMS - n1 - n2 - n3 - n4  # 40+%
    raw = np.concatenate([
        np.random.uniform(0, 5, n1),
        np.random.uniform(5, 10, n2),
        np.random.uniform(10, 25, n3),
        np.random.uniform(25, 40, n4),
        np.clip(np.random.exponential(8, n5) + 40, 40, 100),
    ])
    arr = np.clip(np.round(raw).astype(int), 0, 100)
    return add_offbyone_spikes(arr, [0], spike_frac=0.15)

COLORS = ['#1976D2', '#E64A19', '#AD1457', '#6A1B9A', '#C62828']

d1 = gen_dist1()
d2 = gen_dist2()
d3 = gen_dist3()
d4 = gen_dist4()

def gen_dist5(d1, d2, d3, d4):
    """KDE of Dists 1-4; Normal (Dist 2) weighted at half the others."""
    pooled = np.concatenate([d1, d2[:N_TEAMS // 2], d3, d4]).astype(float)
    kde    = gaussian_kde(pooled, bw_method='scott')
    samples = []
    while len(samples) < N_TEAMS:
        raw = kde.resample(N_TEAMS * 2)[0]
        raw = raw[(raw >= 0) & (raw <= 100)]
        samples.extend(raw.tolist())
    arr = np.clip(np.round(np.array(samples[:N_TEAMS])).astype(int), 0, 100)
    return arr, kde

d5, kde5 = gen_dist5(d1, d2, d3, d4)

dists = [
    (d1, "Dist 1: GPT (Bimodal)"),
    (d2, "Dist 2: Normal (std=15)"),
    (d3, "Dist 3: Gemini"),
    (d4, "Dist 4: Claude"),
    (d5, "Dist 5: KDE Consensus"),
]

# ── Figure 1: Distribution visuals (2×3, last cell hidden) ────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Speed Investment Distributions  (~6 000 teams)", fontsize=16, fontweight='bold')
axes[1][2].set_visible(False)  # only 5 distributions

annot_data = [
    [(12, 0.80, "Peak 1\n5–20%"), (56, 0.80, "Peak 2\n50–80%"), (82, 0.45, "Very\nfew")],
    [(33, 0.88, "Peak ~33%\nstd=15")],
    [(12, 0.45, "Small"), (31, 0.88, "Very large\n26–36%"), (41, 0.70, "Large\n37–46%"),
     (52, 0.60, "Medium\n47–58%"), (75, 0.30, "Small")],
    [(2,  0.70, "~20%"), (17, 0.92, "~45%\nof teams"), (32, 0.65, "~22%"), (55, 0.40, "~8%\noutliers")],
    [(50, 0.80, "Smoothed consensus\n(Normal ½ weight)")],
]
vline_data = [
    [],
    [33],
    [26, 37, 47, 59],
    [5, 10, 25, 40],
    [],
]
offbyone_data = {
    0: [],
    1: [34],
    2: [26, 37],
    3: [1],
    4: [],
}

for idx, (dist, name) in enumerate(dists):
    ax = axes[idx // 3][idx % 3]
    counts, _, _ = ax.hist(dist, bins=range(0, 102), color=COLORS[idx],
                           alpha=0.72, edgecolor='none', density=True)
    ymax = max(counts)

    if idx == 4:
        x_smooth = np.linspace(0, 100, 500)
        kde_vals  = kde5(x_smooth)
        ax.plot(x_smooth, kde_vals, color='black', lw=2.0, label='KDE curve')
        ax.legend(fontsize=9)

    ax.set_title(name, fontweight='bold', fontsize=12, pad=8)
    ax.set_xlabel("Speed Investment (%)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(-1, 101)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    for x, y_frac, label in annot_data[idx]:
        ax.text(x, ymax * y_frac, label, fontsize=8, ha='center',
                bbox=dict(boxstyle='round', fc='white', alpha=0.75))
    for v in vline_data[idx]:
        ax.axvline(v, color='darkred', lw=1.0, linestyle=':', alpha=0.85)
        ax.text(v, ymax * 1.01, f"{v}%", fontsize=7, ha='center',
                color='darkred', rotation=90, va='bottom')
    for hv in offbyone_data.get(idx, []):
        ax.axvline(hv, color='gold', lw=1.2, linestyle='--', alpha=0.8)

plt.tight_layout()
plt.savefig("speed_distributions.png", dpi=150, bbox_inches='tight')
print("Saved: speed_distributions.png")

# ── Optimize ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("OPTIMAL ALLOCATIONS  (grid search, numpy-vectorised)")
print("=" * 72)

results = []
for dist, name in dists:
    sm_arr = speed_mult_vec(dist)
    r_opt, s_opt, v_opt, net_opt = optimize(sm_arr)

    sm_val      = sm_arr[v_opt]
    R_val       = R_ARR[r_opt]
    S_val       = S_ARR[s_opt]
    gross       = R_val * S_val * sm_val
    budget_used = BUDGET * (r_opt + s_opt + v_opt) / 100

    results.append(dict(name=name, r=r_opt, s=s_opt, v=v_opt,
                        total=r_opt+s_opt+v_opt, sm=sm_val,
                        R_val=R_val, S_val=S_val, gross=gross,
                        budget=budget_used, net=net_opt, sm_arr=sm_arr))

    print(f"\n{name}")
    print(f"  Research={r_opt}%  Scale={s_opt}%  Speed={v_opt}%  (Total={r_opt+s_opt+v_opt}%)")
    print(f"  Speed multiplier : {sm_val:.4f}")
    print(f"  Research output  : {R_val:>14,.0f}")
    print(f"  Scale output     : {S_val:>14.4f}")
    print(f"  Gross PnL        : {gross:>14,.0f}")
    print(f"  Budget used      : {budget_used:>14,.0f}")
    print(f"  Net PnL          : {net_opt:>14,.0f}")

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 90)
print("OPTIMAL ALLOCATION SUMMARY")
print("=" * 90)
print(f"{'Distribution':<28} {'R%':>4} {'S%':>4} {'V%':>4} {'Sum':>4} {'SpeedMult':>10} {'Gross PnL':>14} {'Net PnL':>14}")
print("-" * 90)
for r in results:
    print(f"{r['name']:<28} {r['r']:>4} {r['s']:>4} {r['v']:>4} {r['total']:>4} "
          f"{r['sm']:>10.4f} {r['gross']:>14,.0f} {r['net']:>14,.0f}")
print("=" * 90)

# ── Figure 2: PnL vs Speed at optimal R, S (2×3, last cell hidden) ───────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Net PnL vs Speed Investment  (R and S fixed at their optimal values)",
             fontsize=14, fontweight='bold')
axes[1][2].set_visible(False)

for idx, res in enumerate(results):
    ax = axes[idx // 3][idx % 3]
    r_opt, s_opt = res['r'], res['s']
    sm_arr = res['sm_arr']

    v_vals, pnl_vals = [], []
    for v in range(101):
        if r_opt + s_opt + v > 100:
            break
        p = R_ARR[r_opt] * S_ARR[s_opt] * sm_arr[v] - BUDGET * (r_opt + s_opt + v) / 100
        v_vals.append(v); pnl_vals.append(p)

    ax.plot(v_vals, pnl_vals, color=COLORS[idx], linewidth=2.2)
    ax.axvline(res['v'], color='red', linestyle='--', lw=1.5,
               label=f"Optimal V={res['v']}%  PnL={res['net']:,.0f}")
    ax.set_title(res['name'], fontweight='bold', fontsize=11, pad=6)
    ax.set_xlabel("Speed Investment (%)")
    ax.set_ylabel("Net PnL")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout()
plt.savefig("pnl_sensitivity.png", dpi=150, bbox_inches='tight')
print("\nSaved: pnl_sensitivity.png")
