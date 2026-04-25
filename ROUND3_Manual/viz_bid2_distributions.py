import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 30_000

# ── Helpers ───────────────────────────────────────────────────────────────────
def norm(p):
    a = np.array(p, dtype=float)
    return a / a.sum()

def make_samples(bids, pcts, n=N):
    w = norm(pcts)
    counts = np.round(w * n).astype(int)
    return np.concatenate([np.full(c, b) for b, c in zip(bids, counts)])

def triangular_segment(lo, hi, pct, peak='left'):
    n = hi - lo + 1
    w = np.arange(n, 0, -1, dtype=float)
    w = w / w.sum() * pct
    bids = np.arange(lo, hi + 1)
    if peak == 'right':
        w = w[::-1]
    return bids.tolist(), w.tolist()

def uniform_segment(lo, hi, pct):
    n = hi - lo + 1
    w = np.full(n, pct / n)
    return np.arange(lo, hi + 1).tolist(), w.tolist()

def bell_segment(lo, hi, pct, peak=None, sharp=2):
    """Bell peaking at `peak` (default: midpoint). sharp=2 gives quadratic falloff."""
    if peak is None:
        peak = (lo + hi) // 2
    bids = np.arange(lo, hi + 1)
    dists = np.abs(bids - peak).astype(float)
    radius = max(peak - lo, hi - peak) + 1
    w = (radius - dists).clip(min=0) ** sharp
    w = w / w.sum() * pct
    return bids.tolist(), w.tolist()

# ── Distribution 1: GPT ───────────────────────────────────────────────────────
# Ranges: big bells at multiples of 10, mini-mini bells at multiples of 5.
gpt_b, gpt_p = [], []
def add(b, p): gpt_b.extend(b); gpt_p.extend(p)

# ≤820: big bell at 820, mini-mini at 815
b, p = bell_segment(815, 820, 3.0,  peak=820); add(b, p)
b, p = bell_segment(812, 818, 1.0,  peak=815); add(b, p)
# 821-840: big bells at 830, 840; mini-mini at 825, 835
b, p = bell_segment(826, 834, 1.8,  peak=830); add(b, p)
b, p = bell_segment(836, 840, 1.8,  peak=840); add(b, p)
b, p = bell_segment(823, 827, 1.2,  peak=825); add(b, p)
b, p = bell_segment(832, 838, 1.2,  peak=835); add(b, p)
# 841-850: big bell at 850, mini-mini at 845
b, p = bell_segment(844, 850, 6.0,  peak=850); add(b, p)
b, p = bell_segment(841, 847, 2.0,  peak=845); add(b, p)
# 851-857: main at 854, mini-mini at 855
b, p = bell_segment(851, 857, 7.5,  peak=854); add(b, p)
b, p = bell_segment(853, 857, 2.5,  peak=855); add(b, p)
add([858], [14.0])
add([859], [4.0])
add([860], [9.0])
add([861], [3.0])
add([862], [7.0])
add([863], [3.0])
add([864], [5.0])
add([865], [10.0])
add([866], [3.0])
add([867], [7.0])
add([868], [2.0])
add([869], [5.0])
add([870], [7.0])
b, p = bell_segment(871, 875, 7.0,  peak=873); add(b, p)           # 871-875
# 876-885: big bell at 880, mini-mini at 885
b, p = bell_segment(876, 884, 3.2,  peak=880); add(b, p)
b, p = bell_segment(882, 885, 0.8,  peak=885); add(b, p)
# 886+: big bell at 890
b, p = bell_segment(886, 894, 2.0,  peak=890); add(b, p)

gpt_bids = np.array(gpt_b); gpt_pct = np.array(gpt_p)

# ── Distribution 2: Gemini ────────────────────────────────────────────────────
gem_b, gem_p = [], []
def addg(b, p): gem_b.extend(b); gem_p.extend(p)

b, p = triangular_segment(856, 860, 20.0, peak='right'); addg(b, p)
addg([861], [30.0]); addg([862], [30.0])
addg([863], [12.0]); addg([864], [6.0]); addg([865], [2.0])
b, p = triangular_segment(866, 870, 20.0, peak='left'); addg(b, p)

gem_bids = np.array(gem_b); gem_pct = np.array(gem_p)

# ── Distribution 3: Claude ────────────────────────────────────────────────────
# Each listed bid is the bell peak; sharp quadratic falloff on both sides.
cla_b, cla_p = [], []
def addc(b, p): cla_b.extend(b); cla_p.extend(p)

b, p = triangular_segment(810, 819, 1.5, peak='right'); addc(b, p)  # <820: tail peaking at 819
b, p = bell_segment(817, 823, 1.5,  peak=820); addc(b, p)          # narrow: drops by ~822-823
b, p = bell_segment(822, 828, 0.4,  peak=825); addc(b, p)          # mini-mini at 825
b, p = bell_segment(825, 834, 2.0,  peak=830); addc(b, p)
b, p = bell_segment(832, 838, 0.4,  peak=835); addc(b, p)          # mini-mini at 835
b, p = bell_segment(835, 842, 3.0,  peak=840); addc(b, p)
b, p = bell_segment(843, 847, 3.0,  peak=845); addc(b, p)
b, p = bell_segment(847, 853, 8.0,  peak=850); addc(b, p)
b, p = bell_segment(852, 857, 18.0, peak=855); addc(b, p)
b, p = bell_segment(857, 862, 22.0, peak=860); addc(b, p)
b, p = bell_segment(862, 867, 14.0, peak=865); addc(b, p)
b, p = bell_segment(867, 872, 10.0, peak=870); addc(b, p)
b, p = bell_segment(872, 877, 6.0,  peak=875); addc(b, p)
b, p = bell_segment(877, 882, 5.0,  peak=880); addc(b, p)
b, p = bell_segment(882, 889, 3.0,  peak=885); addc(b, p)
b, p = bell_segment(889, 901, 2.0,  peak=895); addc(b, p)
b, p = triangular_segment(905, 919, 1.0, peak='left'); addc(b, p)   # 910+: tail

cla_bids = np.array(cla_b); cla_pct = np.array(cla_p)

# ── Generate samples ──────────────────────────────────────────────────────────
gpt_s = make_samples(gpt_bids, gpt_pct)
gem_s = make_samples(gem_bids, gem_pct)
cla_s = make_samples(cla_bids, cla_pct)
all_s = np.concatenate([gpt_s, gem_s, cla_s])

gpt_mean = np.dot(gpt_bids, norm(gpt_pct))
gem_mean = np.dot(gem_bids, norm(gem_pct))
cla_mean = np.dot(cla_bids, norm(cla_pct))
all_mean = np.mean(all_s)

# ── Style constants (ROUND2 palette) ─────────────────────────────────────────
COLORS = ['#1976D2', '#E64A19', '#AD1457', '#6A1B9A']
BIN_LO, BIN_HI = 808, 922
bins = range(BIN_LO, BIN_HI + 1)

# ── Figure (ROUND2 style: 2×3, last cell hidden) ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Second Bid Distributions — IMC Prosperity Round 3\n"
    "Bid 1 = 795  |  ~3 000 players  |  Whole-number bids only",
    fontsize=16, fontweight='bold'
)
axes[1][2].set_visible(False)

dists_data = [
    (gpt_s,  gpt_mean, "Dist 1: GPT",          COLORS[0]),
    (gem_s,  gem_mean, "Dist 2: Gemini",        COLORS[1]),
    (cla_s,  cla_mean, "Dist 3: Claude",        COLORS[2]),
]

annot_data = [
    [(858, 0.82, "Peak 858\n14%"), (865, 0.60, "Peak 865\n10%"), (820, 0.50, "Tail\n4%")],
    [(861.5, 0.85, "Peak 861–862\n~60%"), (866, 0.60, "Peak 866\n~20%"), (858, 0.45, "~20%\n856–860")],
    [(860, 0.90, "Peak ~860\n22%"), (855, 0.55, "Ramp 855\n18%"), (870, 0.65, "Tail\n10%")],
]

vline_data = [
    [858, 865],
    [861, 866],
    [860],
]

for idx, (samples, mean_val, name, color) in enumerate(dists_data):
    ax = axes[0][idx]
    counts, _, _ = ax.hist(samples, bins=bins, density=True, color=color,
                           alpha=0.72, edgecolor='none')
    ymax = max(counts)
    ax.axvline(mean_val, color='red', ls='--', lw=1.5, label=f'Mean={mean_val:.1f}')
    ax.legend(fontsize=9)
    ax.set_title(name, fontweight='bold', fontsize=12, pad=8)
    ax.set_xlabel("Second Bid", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim(BIN_LO, BIN_HI)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    for x_pos, y_frac, label in annot_data[idx]:
        ax.text(x_pos, ymax * y_frac, label, fontsize=8, ha='center',
                bbox=dict(boxstyle='round', fc='white', alpha=0.75))
    for v in vline_data[idx]:
        ax.axvline(v, color='darkred', lw=1.0, linestyle=':', alpha=0.85)
        ax.text(v, ymax * 1.01, f"{v}", fontsize=7, ha='center',
                color='darkred', rotation=90, va='bottom')

# ── Dist 4: Combined KDE only (ROUND2 Dist 5 style) ──────────────────────────
ax_kde = axes[1][0]
counts_all, _, _ = ax_kde.hist(all_s, bins=bins, density=True,
                                color='red', alpha=0.72, edgecolor='none',
                                label='Pooled histogram')
ymax_kde = max(counts_all)
x_kde = np.linspace(BIN_LO, BIN_HI, 600)
kde_all = gaussian_kde(all_s, bw_method='scott')
ax_kde.plot(x_kde, kde_all(x_kde), color='black', lw=2.0, label='KDE curve')
ax_kde.axvline(all_mean, color='red', ls='--', lw=1.5, label=f'Mean={all_mean:.1f}')
ax_kde.text(all_mean + 1, ymax_kde * 0.80, f"Combined\nmean={all_mean:.1f}",
            fontsize=8, ha='left',
            bbox=dict(boxstyle='round', fc='white', alpha=0.75))
ax_kde.set_title("Dist 4: KDE Consensus (All Sources)", fontweight='bold', fontsize=12, pad=8)
ax_kde.set_xlabel("Second Bid", fontsize=10)
ax_kde.set_ylabel("Density", fontsize=10)
ax_kde.set_xlim(BIN_LO, BIN_HI)
ax_kde.grid(axis='y', alpha=0.25, linestyle='--')
ax_kde.legend(fontsize=9)

# ── Stats panel (cell [1][1]) ─────────────────────────────────────────────────
ax_s = axes[1][1]
ax_s.axis('off')

def stats_for(s):
    p5, p25, p50, p75, p95 = np.percentile(s, [5, 25, 50, 75, 95])
    return dict(Mean=np.mean(s), Median=p50, Std=np.std(s),
                P5=p5, P25=p25, P75=p75, P95=p95,
                Skew=skew(s), Kurt=kurtosis(s))

sg = stats_for(gpt_s)
sm_g = stats_for(gem_s)
sc = stats_for(cla_s)
sa = stats_for(all_s)

lines = [
    "Summary Statistics",
    "─" * 38,
    f"{'':10s} {'GPT':>7} {'Gemini':>7} {'Claude':>7} {'All':>7}",
    "─" * 38,
    f"{'Mean':10s} {sg['Mean']:>7.1f} {sm_g['Mean']:>7.1f} {sc['Mean']:>7.1f} {sa['Mean']:>7.1f}",
    f"{'Median':10s} {sg['Median']:>7.1f} {sm_g['Median']:>7.1f} {sc['Median']:>7.1f} {sa['Median']:>7.1f}",
    f"{'Std':10s} {sg['Std']:>7.1f} {sm_g['Std']:>7.1f} {sc['Std']:>7.1f} {sa['Std']:>7.1f}",
    f"{'P5':10s} {sg['P5']:>7.1f} {sm_g['P5']:>7.1f} {sc['P5']:>7.1f} {sa['P5']:>7.1f}",
    f"{'P25':10s} {sg['P25']:>7.1f} {sm_g['P25']:>7.1f} {sc['P25']:>7.1f} {sa['P25']:>7.1f}",
    f"{'P75':10s} {sg['P75']:>7.1f} {sm_g['P75']:>7.1f} {sc['P75']:>7.1f} {sa['P75']:>7.1f}",
    f"{'P95':10s} {sg['P95']:>7.1f} {sm_g['P95']:>7.1f} {sc['P95']:>7.1f} {sa['P95']:>7.1f}",
    f"{'Skewness':10s} {sg['Skew']:>7.3f} {sm_g['Skew']:>7.3f} {sc['Skew']:>7.3f} {sa['Skew']:>7.3f}",
    f"{'Kurtosis':10s} {sg['Kurt']:>7.3f} {sm_g['Kurt']:>7.3f} {sc['Kurt']:>7.3f} {sa['Kurt']:>7.3f}",
    "",
    "Calculated Means",
    "─" * 38,
    f"  GPT     : {gpt_mean:.2f}",
    f"  Gemini  : {gem_mean:.2f}",
    f"  Claude  : {cla_mean:.2f}",
    f"  Combined: {all_mean:.2f}",
    "",
    "Recommendation",
    "─" * 38,
    f"  Penalty cancels at b2 = avg_b2",
    f"  Best b2 : 862–863",
    f"  Hedge   : 865–870",
]

ax_s.text(0.04, 0.98, "\n".join(lines), transform=ax_s.transAxes,
          fontsize=9, verticalalignment='top', family='monospace',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9',
                    edgecolor='#cccccc', linewidth=1))

plt.tight_layout()
out = "C:/Users/sab06/IMC Prosperity/ROUND3/ROUND3_Manual/bid2_distributions_viz.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved -> {out}")
print(f"\nMeans — GPT: {gpt_mean:.2f}  Gemini: {gem_mean:.2f}  Claude: {cla_mean:.2f}  Combined: {all_mean:.2f}")
for label, s in [("GPT", sg), ("Gemini", sm_g), ("Claude", sc), ("Combined", sa)]:
    print(f"{label:10s}  mean={s['Mean']:.1f}  median={s['Median']:.1f}  std={s['Std']:.1f}"
          f"  P25={s['P25']:.1f}  P75={s['P75']:.1f}  skew={s['Skew']:.3f}  kurt={s['Kurt']:.3f}")
plt.close()
