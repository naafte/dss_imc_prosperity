import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N_TEAMS = 10000  # other teams (I am +1)
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
    Fast optimizer using the fact that S(s) = 7s/100 is linear in s.
    For any fixed (r, v), the optimal s is at a boundary (0 or max_rs-r).
    This reduces the search from O(100^3) to O(100^2).
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
        # PnL contribution from s is linear => optimum at boundary
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
# Note: peaks at round numbers (0%, 50%, etc.) also generate off-by-one spikes
# at the +1 value (1%, 51%) from players trying to edge above the Schelling point.

def add_offbyone_spikes(speeds, schelling_pts, spike_frac=0.05):
    """For each Schelling point, add a spike at +1 with spike_frac of its mass."""
    speeds = list(speeds)
    arr = np.array(speeds)
    for pt in schelling_pts:
        n_at_pt  = int(np.sum(arr == pt))
        n_spike  = int(n_at_pt * spike_frac)
        # Remove n_spike entries at pt, replace with pt+1
        idxs = np.where(arr == pt)[0][:n_spike]
        arr[idxs] = min(pt + 1, 100)
    return arr

def gen_dist1():
    """Exponential: very few at low %, rises exponentially, peak ~70%, sparse 70-100."""
    k = 0.08
    n_main, n_tail = int(N_TEAMS * 0.93), int(N_TEAMS * 0.07)
    U = np.random.uniform(0, 1, n_main)
    x_main = np.log(1 + U * (np.exp(k * 70) - 1)) / k
    x_tail = np.random.uniform(70, 100, n_tail)
    return np.clip(np.round(np.concatenate([x_main, x_tail])).astype(int), 0, 100)

def gen_dist2():
    """Uniform Low: uniform 0-50, slight drop 50-70, near-zero 70-100."""
    w = np.array([50.0, 14.0, 1.5]); w /= w.sum()
    n1, n2 = int(N_TEAMS * w[0]), int(N_TEAMS * w[1]); n3 = N_TEAMS - n1 - n2
    raw = np.concatenate([
        np.random.uniform(0, 50, n1),
        np.random.uniform(50, 70, n2),
        np.random.uniform(70, 100, n3),
    ])
    speeds = np.clip(np.round(raw).astype(int), 0, 100)
    # Off-by-one spike at 1 (from players edging above 0)
    return add_offbyone_spikes(speeds, [0], spike_frac=0.15)

def gen_dist3():
    """Uniform Race: lower uniform 0-50, rises 50-70, near-zero 70-100."""
    # 50-70 region is slightly denser than 0-50
    w = np.array([50.0, 28.0, 1.5]); w /= w.sum()
    n1, n2 = int(N_TEAMS * w[0]), int(N_TEAMS * w[1]); n3 = N_TEAMS - n1 - n2
    raw = np.concatenate([
        np.random.uniform(0, 50, n1),
        np.random.uniform(50, 70, n2),
        np.random.uniform(70, 100, n3),
    ])
    speeds = np.clip(np.round(raw).astype(int), 0, 100)
    return add_offbyone_spikes(speeds, [0, 50], spike_frac=0.12)

def gen_dist4():
    """Normal distribution centered at 33.33%, std~10, truncated to [0,100]."""
    raw = np.random.normal(loc=33.33, scale=10, size=N_TEAMS * 2)
    raw = raw[(raw >= 0) & (raw <= 100)][:N_TEAMS]
    arr = np.clip(np.round(raw).astype(int), 0, 100)
    return add_offbyone_spikes(arr, [33], spike_frac=0.10)

def gen_dist5():
    """Normal distribution centered at 33.33%, std~15, truncated to [0,100]."""
    raw = np.random.normal(loc=33.33, scale=15, size=N_TEAMS * 2)
    raw = raw[(raw >= 0) & (raw <= 100)][:N_TEAMS]
    arr = np.clip(np.round(raw).astype(int), 0, 100)
    return add_offbyone_spikes(arr, [33], spike_frac=0.10)

def gen_dist6():
    """Bimodal/Gemini: oblivious peak 0-10, dead zone 11-32, war-zone bulge 33-58.
    Schelling spikes at 33, 40, 50, 55 plus off-by-one edges at 34, 41, 51, 56."""
    speeds = []
    n_ob   = int(N_TEAMS * 0.08)
    p_ob   = np.array([0.40] + [0.067] * 9); p_ob /= p_ob.sum()
    speeds.extend(np.random.choice(range(0, 10), n_ob, p=p_ob))

    n_dead = int(N_TEAMS * 0.03)
    speeds.extend(np.random.randint(11, 33, n_dead))

    n_war  = N_TEAMS - n_ob - n_dead
    schelling = {33: 0.22, 40: 0.28, 50: 0.35, 55: 0.15}
    n_sch  = int(n_war * 0.32)
    pts, wts = zip(*schelling.items())
    speeds.extend(np.random.choice(pts, n_sch, p=list(wts)))

    n_cont = n_war - n_sch
    raw    = np.random.beta(2, 5, n_cont) * 26 + 33
    speeds.extend(np.round(raw).astype(int))

    arr = np.clip(np.array(speeds, dtype=int), 0, 100)
    return add_offbyone_spikes(arr, [0, 33, 40, 50, 55], spike_frac=0.10)

def gen_dist7():
    """Noisy/Claude: spike at 0, mass 20-40, exponential tail to 60+."""
    n_zero = int(N_TEAMS * 0.09)
    n_mid  = int(N_TEAMS * 0.57)
    n_tail = N_TEAMS - n_zero - n_mid
    s_mid  = np.random.beta(2, 2, n_mid) * 20 + 20
    s_tail = np.clip(np.random.exponential(12, n_tail) + 40, 40, 100)
    raw    = np.concatenate([np.zeros(n_zero), s_mid, s_tail])
    arr    = np.clip(np.round(raw).astype(int), 0, 100)
    return add_offbyone_spikes(arr, [0], spike_frac=0.15)

def gen_dist8():
    """Race/GPT: heavy concentration 55-95%, inverse of typical distributions."""
    # From the table: 85-95=25%, 70-85=30%, 55-70=22%, 40-55=12%, 25-40=7%, <25=4%
    segments = [(85, 95, 0.25), (70, 85, 0.30), (55, 70, 0.22),
                (40, 55, 0.12), (25, 40, 0.07), (0,  25, 0.04)]
    parts = []
    total_so_far = 0
    for i, (lo, hi, frac) in enumerate(segments):
        n = int(N_TEAMS * frac) if i < len(segments) - 1 else N_TEAMS - total_so_far
        total_so_far += n
        parts.append(np.random.uniform(lo, hi, n))
    arr = np.clip(np.round(np.concatenate(parts)).astype(int), 0, 100)
    # Off-by-one near 50 and 0 (smaller effect here since most players are high)
    return add_offbyone_spikes(arr, [0, 50], spike_frac=0.10)

# ── Generate all eight, then derive the 9th ──────────────────────────────────

COLORS = ['#1976D2', '#388E3C', '#E64A19', '#AD1457', '#6A1B9A', '#7B1FA2', '#F57C00', '#00838F', '#C62828']

d1 = gen_dist1(); d2 = gen_dist2(); d3 = gen_dist3(); d4 = gen_dist4()
d5 = gen_dist5(); d6 = gen_dist6(); d7 = gen_dist7(); d8 = gen_dist8()

def gen_dist9(all_eight):
    """
    KDE of the combined speeds from all 8 distributions.
    Fit a gaussian_kde to the 80 000 pooled samples, then draw
    10 000 new integer samples from that smoothed distribution.
    """
    pooled = np.concatenate(all_eight).astype(float)
    kde    = gaussian_kde(pooled, bw_method='scott')
    samples = []
    while len(samples) < N_TEAMS:
        raw = kde.resample(N_TEAMS * 2)[0]
        raw = raw[(raw >= 0) & (raw <= 100)]
        samples.extend(raw.tolist())
    arr = np.clip(np.round(np.array(samples[:N_TEAMS])).astype(int), 0, 100)
    return arr, kde

d9, kde9 = gen_dist9([d1, d2, d3, d4, d5, d6, d7, d8])

dists = [
    (d1, "Dist 1: Exponential"),
    (d2, "Dist 2: Uniform (Low)"),
    (d3, "Dist 3: Uniform (Race)"),
    (d4, "Dist 4: Normal (std=10)"),
    (d5, "Dist 5: Normal (std=15)"),
    (d6, "Dist 6: Bimodal (Gemini)"),
    (d7, "Dist 7: Noisy (Claude)"),
    (d8, "Dist 8: Race (GPT)"),
    (d9, "Dist 9: KDE Consensus"),
]

# ── Figure 1: Distribution visuals (3×3, last cell empty) ────────────────────

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle("Speed Investment Distributions  (~10 000 teams)", fontsize=16, fontweight='bold')

annot_data = [
    [(10, 0.55, "Very few\nat low %"), (70, 0.92, "Peak ~70%")],
    [(25, 0.88, "Uniform 0-50%"), (59, 0.65, "Slight drop"), (83, 0.25, "Near-zero")],
    [(25, 0.70, "Lower flat\n0-50%"), (60, 0.88, "Rises\n50-70%"), (83, 0.25, "Near-zero")],
    [(33, 0.92, "Peak ~33%\nstd=10")],
    [(33, 0.92, "Peak ~33%\nstd=15")],
    [(5, 0.92, "Oblivious"), (21, 0.40, "Dead\nzone"), (44, 0.70, "War-zone")],
    [(0, 0.88, "0% cluster"), (30, 0.78, "Core 20-40%"), (62, 0.45, "Tail 60+")],
    [(15, 0.40, "<25%:\n~4%"), (47, 0.55, "40-55%:\n~12%"), (77, 0.88, "55-95%:\n~77%")],
    [(50, 0.92, "Smoothed\nconsensus")],
]
vline_data = [
    [68], [], [50], [33], [33], [33, 40, 50, 55], [], [55], [],
]
offbyone_data = {
    1: [1], 2: [1, 51], 3: [34], 4: [34], 5: [1, 34, 41, 51, 56], 6: [1], 7: [1, 51],
}

for idx, (dist, name) in enumerate(dists):
    ax = axes[idx // 3][idx % 3]
    counts, _, _ = ax.hist(dist, bins=range(0, 102), color=COLORS[idx],
                           alpha=0.72, edgecolor='none', density=True)
    ymax = max(counts)

    # Overlay KDE smooth curve on the 9th subplot
    if idx == 8:
        x_smooth = np.linspace(0, 100, 500)
        kde_vals  = kde9(x_smooth)
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

# All 9 cells used — nothing to hide

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

# ── Figure 2: PnL vs Speed at optimal R, S ───────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle("Net PnL vs Speed Investment  (R and S fixed at their optimal values)",
             fontsize=14, fontweight='bold')

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
