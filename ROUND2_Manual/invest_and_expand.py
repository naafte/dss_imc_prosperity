"""
Round 2 Manual Challenge: "Invest & Expand"
============================================

Budget: 50_000 XIRECs.
Allocate percentages (0-100) across three pillars (sum <= 100):
  - Research(x) = 200_000 * log(1+x) / log(1+100)       # logarithmic 0 -> 200_000
  - Scale(x)    = 7 * x / 100                            # linear      0 -> 7
  - Speed       = rank-based multiplier across players   # 0.9 (top) ... 0.1 (bottom)

PnL = Research(R) * Scale(Sc) * SpeedMult(Sp) - Budget_Used
Budget_Used = (R + Sc + Sp) / 100 * 50_000

Speed is the adversarial piece. We model opponent Speed bids as the user's
"ugly bimodal" distribution: oblivious peak (0-10), dead zone (10-25),
war-zone bulge (30-60, left-skewed), Schelling spikes at 33.3 / 40 / 50 / 55.

Run:  python invest_and_expand.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-blocking backend
import matplotlib.pyplot as plt
from dataclasses import dataclass

BUDGET = 50_000


# --------------------------------------------------------------------------- #
# Pillar functions
# --------------------------------------------------------------------------- #
def research(x: np.ndarray | float) -> np.ndarray | float:
    return 200_000.0 * np.log(1.0 + x) / np.log(1.0 + 100.0)


def scale(x: np.ndarray | float) -> np.ndarray | float:
    return 7.0 * x / 100.0


def speed_mult_from_rank(rank: np.ndarray, n: int) -> np.ndarray:
    """Rank 1 -> 0.9, rank n -> 0.1, linear between."""
    if n <= 1:
        return np.full_like(rank, 0.9, dtype=float)
    return 0.9 - (rank - 1) * (0.8 / (n - 1))


# --------------------------------------------------------------------------- #
# Opponent bid distribution (vectorized)
# --------------------------------------------------------------------------- #
def sample_opponent_speed_bids(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Vectorized mixture:
      - 15%  oblivious  : Beta(1.2, 4) * 10            in [0, 10]
      -  5%  dead zone  : Uniform(10, 25)
      - 55%  war bulge  : 30 + Beta(5, 2.2) * 30       in [30, 60], left-skewed
      - 25%  Schelling  : pick from {33.33, 40, 50, 55} with jitter
    """
    w = np.array([0.15, 0.05, 0.55, 0.25])
    comp = rng.choice(4, size=n, p=w)

    out = np.empty(n)

    m0 = comp == 0
    m1 = comp == 1
    m2 = comp == 2
    m3 = comp == 3

    out[m0] = rng.beta(1.2, 4.0, size=m0.sum()) * 10.0
    out[m1] = rng.uniform(10.0, 25.0, size=m1.sum())
    out[m2] = 30.0 + rng.beta(5.0, 2.2, size=m2.sum()) * 30.0

    spike_vals = np.array([33.333, 40.0, 50.0, 55.0])
    spike_probs = np.array([0.20, 0.25, 0.35, 0.20])
    spikes = rng.choice(spike_vals, size=m3.sum(), p=spike_probs)
    out[m3] = spikes + rng.normal(0.0, 0.3, size=m3.sum())

    return np.clip(out, 0.0, 100.0)


# --------------------------------------------------------------------------- #
# Speed multiplier: vectorized over many allocations, given a sorted opponent panel
# --------------------------------------------------------------------------- #
def speed_mult_many(my_bids: np.ndarray, sorted_opp_desc: np.ndarray) -> np.ndarray:
    """
    Given a sorted-descending opponent-bid array, compute the speed multiplier
    for every bid in `my_bids`. Ties share the same (best) rank.
    """
    n_opp = len(sorted_opp_desc)
    n_total = n_opp + 1
    # ascending copy for searchsorted
    sorted_asc = sorted_opp_desc[::-1]
    # number of opponents strictly greater than my_bid -> my rank = that + 1
    # searchsorted with side='right' on ascending array: index of first element > value
    # number of opponents > value = n_opp - idx
    idx_right = np.searchsorted(sorted_asc, my_bids, side="right")
    n_greater = n_opp - idx_right
    # Ties tied-with-me don't change my rank (shared), only strictly-greater do.
    my_rank = n_greater + 1
    return speed_mult_from_rank(my_rank, n_total)


# --------------------------------------------------------------------------- #
# Allocation + PnL
# --------------------------------------------------------------------------- #
@dataclass
class Allocation:
    research_pct: float
    scale_pct: float
    speed_pct: float

    @property
    def total(self) -> float:
        return self.research_pct + self.scale_pct + self.speed_pct

    def budget_used(self) -> float:
        return self.total / 100.0 * BUDGET

    def gross_prefix(self) -> float:
        return research(self.research_pct) * scale(self.scale_pct)


def simulate_pnl_for(
    allocs: list[Allocation],
    n_opponents: int = 400,
    n_sims: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """
    Returns a (len(allocs), n_sims) PnL array. Shares opponent panels across
    allocations so comparisons are apples-to-apples.
    """
    rng = np.random.default_rng(seed)
    speed_bids = np.array([a.speed_pct for a in allocs])
    prefix = np.array([a.gross_prefix() for a in allocs])
    used = np.array([a.budget_used() for a in allocs])

    pnl = np.empty((len(allocs), n_sims))
    for k in range(n_sims):
        opp = sample_opponent_speed_bids(n_opponents, rng)
        opp.sort()
        opp = opp[::-1]  # descending
        sm = speed_mult_many(speed_bids, opp)
        pnl[:, k] = prefix * sm - used
    return pnl


# --------------------------------------------------------------------------- #
# Grid search (fast, vectorized over allocations per opponent panel)
# --------------------------------------------------------------------------- #
def grid_search(
    step: float = 2.0,
    n_opponents: int = 400,
    n_sims: int = 300,
    seed: int = 123,
):
    """
    Sweep (R, Sc, Sp) on `step`% grid with R + Sc + Sp <= 100.
    Returns DataFrame-like list sorted by mean PnL descending.
    """
    rs_grid = np.arange(0, 100 + 1e-9, step)
    allocs: list[Allocation] = []
    for r in rs_grid:
        for sc in np.arange(0, 100 - r + 1e-9, step):
            for sp in np.arange(0, 100 - r - sc + 1e-9, step):
                allocs.append(Allocation(float(r), float(sc), float(sp)))

    pnl = simulate_pnl_for(allocs, n_opponents=n_opponents,
                           n_sims=n_sims, seed=seed)
    mean_pnl = pnl.mean(axis=1)
    med_pnl = np.median(pnl, axis=1)
    p10 = np.percentile(pnl, 10, axis=1)
    p90 = np.percentile(pnl, 90, axis=1)

    order = np.argsort(-mean_pnl)
    results = [
        (mean_pnl[i], med_pnl[i], p10[i], p90[i], allocs[i])
        for i in order
    ]
    return results


# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
def compare(
    allocations: list[Allocation],
    n_opponents: int = 400,
    n_sims: int = 4000,
    seed: int = 7,
) -> None:
    pnl = simulate_pnl_for(allocations, n_opponents=n_opponents,
                           n_sims=n_sims, seed=seed)

    print(f"{'R%':>6} {'Sc%':>6} {'Sp%':>6} "
          f"{'Used':>8} {'Res':>9} {'Scl':>7} "
          f"{'E[Sp]':>7} {'MeanPnL':>11} {'MedPnL':>11} "
          f"{'P10':>11} {'P90':>11}")
    print("-" * 108)

    for i, alloc in enumerate(allocations):
        # E[Sp] backed out from E[PnL] = prefix*E[Sp] - used  =>  E[Sp] = (E[PnL]+used)/prefix
        pref = alloc.gross_prefix()
        used = alloc.budget_used()
        mean_pnl = pnl[i].mean()
        e_sp = (mean_pnl + used) / pref if pref > 0 else float("nan")
        print(
            f"{alloc.research_pct:>6.1f} "
            f"{alloc.scale_pct:>6.1f} "
            f"{alloc.speed_pct:>6.1f} "
            f"{used:>8.0f} "
            f"{research(alloc.research_pct):>9.0f} "
            f"{scale(alloc.scale_pct):>7.3f} "
            f"{e_sp:>7.3f} "
            f"{mean_pnl:>11.0f} "
            f"{np.median(pnl[i]):>11.0f} "
            f"{np.percentile(pnl[i], 10):>11.0f} "
            f"{np.percentile(pnl[i], 90):>11.0f}"
        )


# --------------------------------------------------------------------------- #
# If Speed rank were known, what's the best R/Sc split?
# --------------------------------------------------------------------------- #
def best_split_for_known_speed(speed_pct: float, speed_mult: float,
                               step: float = 0.25):
    remaining = 100 - speed_pct
    rs = np.arange(0, remaining + 1e-9, step)
    best_pnl = -np.inf
    best_alloc: Allocation | None = None
    for r in rs:
        scs = np.arange(0, remaining - r + 1e-9, step)
        prefix = research(r) * scale(scs)
        used = (r + scs + speed_pct) / 100.0 * BUDGET
        pnl = prefix * speed_mult - used
        k = int(np.argmax(pnl))
        if pnl[k] > best_pnl:
            best_pnl = float(pnl[k])
            best_alloc = Allocation(float(r), float(scs[k]), speed_pct)
    return best_pnl, best_alloc


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_opponent_distribution(n: int = 50_000, out_path: str | None = None):
    rng = np.random.default_rng(0)
    bids = sample_opponent_speed_bids(n, rng)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(bids, bins=200, color="steelblue", edgecolor="none")
    for x in [33.333, 40, 50, 55]:
        ax.axvline(x, color="red", alpha=0.3, linestyle="--")
    ax.set_xlabel("Speed bid (%)")
    ax.set_ylabel("Count")
    ax.set_title("Modelled opponent Speed-bid distribution "
                 "(bimodal with Schelling spikes)")
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def plot_pillar_curves(out_path: str | None = None):
    xs = np.linspace(0, 100, 500)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, research(xs), label="Research(x)", color="C0")
    ax[0].plot(xs, scale(xs) * 1e4, label="Scale(x) x 1e4", color="C2")
    ax[0].set_xlabel("% of budget allocated")
    ax[0].set_ylabel("Output")
    ax[0].legend()
    ax[0].set_title("Research vs Scale (Scale scaled x 10^4 for visibility)")

    # Marginal Research/XIREC: dR/dx vs x  (per 1% = 500 XIRECs)
    dR_dx = np.gradient(research(xs), xs)
    ax[1].plot(xs, dR_dx)
    ax[1].set_xlabel("Research allocation %")
    ax[1].set_ylabel("Marginal research per 1%")
    ax[1].set_title("Marginal return of Research allocation (diminishing)")
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    plot_opponent_distribution(out_path="opponent_speed_distribution.png")
    plot_pillar_curves(out_path="pillar_curves.png")

    # 1) Hand-picked comparison
    baseline = [
        Allocation(33.33, 33.33, 33.33),
        Allocation(40, 30, 30),
        Allocation(30, 30, 40),
        Allocation(30, 20, 50),
        Allocation(25, 20, 55),
        Allocation(20, 20, 60),
        Allocation(15, 15, 70),
        Allocation(10, 10, 80),
        Allocation(35, 25, 40),
        Allocation(40, 25, 35),
        Allocation(45, 20, 35),
        Allocation(50, 20, 30),
        Allocation(60, 20, 20),
        Allocation(70, 15, 15),
        Allocation(35, 20, 45),
        Allocation(30, 15, 55),
        Allocation(40, 15, 45),
        Allocation(45, 15, 40),
        Allocation(60, 40, 0),
        Allocation(50, 50, 0),
    ]
    print("\n=== Comparison of hand-picked allocations "
          "(n_sims=4000, n_opponents=400) ===")
    compare(baseline, n_sims=4000, n_opponents=400, seed=7)

    # 2) Grid search (coarse)
    print("\n=== Grid-search top 25 (step=2%, n_sims=400) ===")
    results = grid_search(step=2.0, n_opponents=400, n_sims=400, seed=123)
    print(f"{'R%':>6} {'Sc%':>6} {'Sp%':>6} {'tot':>6} "
          f"{'mean':>11} {'median':>11} {'P10':>11} {'P90':>11}")
    for mean_p, med_p, p10_p, p90_p, alloc in results[:25]:
        print(f"{alloc.research_pct:>6.1f} {alloc.scale_pct:>6.1f} "
              f"{alloc.speed_pct:>6.1f} {alloc.total:>6.1f} "
              f"{mean_p:>11.0f} {med_p:>11.0f} "
              f"{p10_p:>11.0f} {p90_p:>11.0f}")

    # 3) Known-speed oracle: upper bound given you somehow knew your speed mult
    print("\n=== Oracle: best R/Sc split given known Speed rank ===")
    print(f"{'Sp%':>5} {'mult':>6} {'best R':>8} {'best Sc':>8} "
          f"{'total':>6} {'best PnL':>12}")
    for sp_pct, sm in [
        (0, 0.10), (5, 0.20), (10, 0.30), (20, 0.40), (30, 0.50),
        (35, 0.55), (40, 0.60), (45, 0.70), (50, 0.80),
        (55, 0.85), (60, 0.88), (70, 0.90),
    ]:
        pnl_val, alloc = best_split_for_known_speed(sp_pct, sm)
        print(f"{sp_pct:>5} {sm:>6.2f} "
              f"{alloc.research_pct:>8.2f} {alloc.scale_pct:>8.2f} "
              f"{alloc.total:>6.1f} {pnl_val:>12.0f}")

    # 4) Finer grid near the winner
    print("\n=== Finer grid (step=1%) around top region ===")
    fine = grid_search(step=1.0, n_opponents=400, n_sims=300, seed=321)
    for mean_p, med_p, p10_p, p90_p, alloc in fine[:15]:
        print(f"R={alloc.research_pct:>5.1f}  "
              f"Sc={alloc.scale_pct:>5.1f}  "
              f"Sp={alloc.speed_pct:>5.1f}  "
              f"tot={alloc.total:>5.1f}  "
              f"mean={mean_p:>10.0f}  "
              f"median={med_p:>10.0f}  "
              f"P10={p10_p:>10.0f}  P90={p90_p:>10.0f}")
