# Invest & Expand — Speed Distribution Analysis

Manual trading challenge for IMC Prosperity Round 2.

---

## The Problem

Budget: **50,000 XIRECs** allocated across three pillars (integer percentages, total ≤ 100%).

```
Net PnL = Research(r%) × Scale(s%) × Speed(v%) − Budget_Used
```

| Pillar | Formula | Range |
|---|---|---|
| **Research** | `200,000 × ln(1+r) / ln(101)` | 0 → 200,000 (logarithmic) |
| **Scale** | `7 × s / 100` | 0 → 7 (linear) |
| **Speed** | Rank-based among all ~10,000 teams | 0.1 → 0.9 multiplier |

**Speed rank rules:** highest investment → 0.9 multiplier, lowest → 0.1, everyone else linearly interpolated. Equal investments share the same rank.

**Key insight:** Research has heavy diminishing returns (log), Scale is linear, and Speed is a zero-sum rank race. This means:
- Pouring too much into Research is wasteful past ~15–20%.
- Scale absorbs leftover budget efficiently.
- Speed is the strategic wildcard — its value depends entirely on what everyone else does.

---

## Speed Distribution Models

Seven candidate distributions were modelled (~10,000 teams each). All speed percentages are integers. Peaks at round numbers (0%, 50%, etc.) also generate small spikes at +1% from players trying to edge above Schelling points.

### Distribution 1 — Exponential

Very few teams pick low percentages. Density rises exponentially, peaking around **70%**, then drops to near-zero from 70–100%.

> Most teams treat Speed as the dominant pillar and race to the top.

### Distribution 2 — Uniform (Low)

Roughly uniform from **0–50%**, a slight step down from 50–70%, near-zero above 70%.

> Teams hedge evenly across the lower half of the speed range.

### Distribution 3 — Uniform (Race)

Similar to Dist 2 but the **50–70% band is denser than 0–50%** — a partial arms race where many teams pile into the mid-high range.

> More competitive than Dist 2 but not a full GPT-style arms race.

### Distribution 4 — Normal (std=10)

A normal distribution centered at **33.33%** with standard deviation ~10%. Represents teams converging tightly on the "fair" even split. Off-by-one spike at 34% from players edging above the Schelling point. 68% of teams fall between **23–43%**.

### Distribution 5 — Normal (std=15)

Same center at **33.33%** but with standard deviation ~15% — a wider, flatter bell. Represents a more dispersed version of the same rational convergence. 68% of teams fall between **18–48%**, with meaningful mass reaching into the 50–60% range.

### Distribution 6 — Bimodal (Gemini)

A psychologically-shaped distribution with three zones:

- **Oblivious peak (0–10%):** ~8% of teams ignored Speed entirely.
- **Dead zone (11–32%):** very few bids — too weak to compete, too costly to waste.
- **War-zone bulge (33–58%):** the dense majority, skewed left.
  - **Schelling point spikes** at 33%, 40%, 50%, 55% — and off-by-one edges at 34%, 41%, 51%, 56%.

> Bidding exactly 50% lands in a massive rank collision but still nets a strong multiplier (~0.85) because so many others cluster there too.

### Distribution 7 — Noisy (Claude)

- **Spike at 0%:** ~9% of teams reason through the math and skip Speed.
- **Core mass at 20–40%:** the hedgers (~57%).
- **Exponential tail to 60+%:** the speed-dominant players (~34%).

### Distribution 8 — Race (GPT)

A full arms-race scenario. ~77% of all teams invest between **55–95%** in Speed.

| Speed Range | % of Teams |
|---|---|
| 85–95% | ~25% |
| 70–85% | ~30% |
| 55–70% | ~22% |
| 40–55% | ~12% |
| 25–40% | ~7% |
| < 25% | ~4% |

> Even investing 59% only yields a 0.34 speed multiplier. R and S are so starved that Net PnL collapses.

### Distribution 9 — KDE Consensus

A **Gaussian KDE** fitted to all 80,000 pooled samples from Distributions 1–8, then resampled to 10,000. Represents a "no strong prior" hedge — the expected distribution if all eight models are equally likely.

> The smooth KDE curve is overlaid on the histogram in the visualization.

---

## Visualizations

| File | Contents |
|---|---|
| `speed_distributions.png` | Histogram of each distribution with annotations and Schelling-point markers |
| `pnl_sensitivity.png` | Net PnL vs Speed % for each distribution, with optimal Speed marked |

---

## Optimal Allocations

Grid search over all integer (r, s, v) with r + s + v ≤ 100. Optimization is fast because Scale is linear in s — for any fixed (r, v), the optimal s is always a boundary value.

| Distribution | Research % | Scale % | Speed % | Speed Mult | Gross PnL | Net PnL |
|---|---|---|---|---|---|---|
| Dist 1: Exponential | 9% | 22% | 69% | 0.816 | 125,356 | 75,356 |
| Dist 2: Uniform (Low) | 15% | 46% | 39% | 0.580 | 224,335 | 174,335 |
| Dist 3: Uniform (Race) | 16% | 48% | 36% | 0.466 | 192,309 | 142,309 |
| **Dist 4: Normal (std=10)** | **14%** | **42%** | **44%** | **0.799** | **275,523** | **225,523** |
| Dist 5: Normal (std=15) | 15% | 42% | 43% | 0.699 | 247,048 | 197,048 |
| Dist 6: Bimodal (Gemini) | 13% | 37% | 50% | 0.850 | 251,658 | 201,658 |
| Dist 7: Noisy (Claude) | 16% | 47% | 37% | 0.609 | 245,922 | 195,922 |
| Dist 8: Race (GPT) | 14% | 39% | 47% | 0.237 | 75,969 | 25,969 |
| Dist 9: KDE Consensus | 14% | 42% | 44% | 0.560 | 193,242 | 143,242 |

---

## Key Takeaways

**Research stays low (9–16%) across all scenarios.** Log diminishing returns make investing beyond ~16% expensive relative to payoff.

**Scale absorbs the non-Speed budget.** Since Scale is linear (no diminishing returns), every spare percent goes there.

**Speed is the pivotal decision.** The optimal Speed allocation ranges from 36% (Dist 3) to 69% (Dist 1) depending on the competitive environment.

**Best case: Dist 4 (Normal std=10) — Net PnL 225,523.** Teams tightly converging on ~33% speed lets you invest 44% and land well above the crowd with a strong 0.80 multiplier.

**Dist 5 (Normal std=15) — Net PnL 197,048.** The wider spread dilutes the advantage — more teams reach into the 40–50% range, compressing your speed multiplier to 0.70.

**Dist 6 (Bimodal/Gemini) — Net PnL 201,658.** Investing exactly 50% exploits the Schelling-point collision for a ~0.85 multiplier.

**Worst case: Dist 8 (Race/GPT) — Net PnL 25,969.** A full arms race destroys value for everyone. Even the optimal counter-strategy only captures a 0.24 speed multiplier.

**KDE Consensus (Dist 9) suggests R=14%, S=42%, V=44%** as a reasonable hedge across all eight scenarios. Net PnL of ~143k.

---

## Files

```
ROUND2_Manual/
├── Directions.txt          # Challenge rules
├── Speed Distributions.txt # Distribution descriptions
├── analyze_invest.py       # Python analysis + optimization
├── speed_distributions.png # Distribution visuals
├── pnl_sensitivity.png     # PnL sensitivity curves
├── Ignore.txt              # Random thoughts and ideas
└── README.md               # This file
```
