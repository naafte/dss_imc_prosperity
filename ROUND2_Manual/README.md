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
| **Speed** | Rank-based among all ~6,000 teams | 0.1 → 0.9 multiplier |

**Speed rank rules:** highest investment → 0.9 multiplier, lowest → 0.1, everyone else linearly interpolated. Equal investments share the same rank.

---

## Speed Distribution Models

Five candidate distributions modelled (~6,000 teams each). All speed percentages are integers.

### Distribution 1 — GPT (Bimodal)

Two peaks: one around **5–20%** (players skipping the speed race) and one around **50–80%** (competitive players). Sparse from 61–81%, very few above 81%.

### Distribution 2 — Normal (std=15)

Normal distribution centered at **33.33%**, standard deviation ~15%. 68% of teams fall between **18–48%**. Off-by-one spike at 34% from players edging above the 33% Schelling point.

### Distribution 3 — Gemini

Clustered distribution with distinct density segments:

| Range | Density |
|---|---|
| 0–25% | Small |
| 26–36% | **Very large** |
| 37–46% | Large |
| 47–58% | Medium |
| 59%+ | Small |

### Distribution 4 — Claude

Low-speed-biased distribution:

| Range | Share of teams |
|---|---|
| 0–5% | ~20% |
| 5–10% | Near-zero |
| 10–25% | ~45% |
| 25–40% | ~22% |
| 40%+ | ~8% outliers |

### Distribution 5 — KDE Consensus

Gaussian KDE fitted to Distributions 1–4 pooled, with **Normal (Dist 2) weighted at half** the other three. Resampled to 6,000 teams.

---

## Visualizations

| File | Contents |
|---|---|
| `speed_distributions.png` | 2×3 histogram grid (5 distributions + 1 blank) with annotations |
| `pnl_sensitivity.png` | Net PnL vs Speed % for each distribution, with optimal Speed marked |

---

## Optimal Allocations

Grid search over all integer (r, s, v) with r + s + v ≤ 100.

| Distribution | Research % | Scale % | Speed % | Speed Mult | Gross PnL | Net PnL |
|---|---|---|---|---|---|---|
| Dist 1: GPT (Bimodal) | 19% | 61% | 20% | 0.3248 | 180,050 | 130,050 |
| Dist 2: Normal (std=15) | 15% | 42% | 43% | 0.6989 | 246,897 | 196,897 |
| Dist 3: Gemini | 14% | 39% | 47% | 0.6929 | 222,002 | 172,002 |
| **Dist 4: Claude** | **18%** | **57%** | **25%** | **0.6657** | **338,940** | **288,940** |
| Dist 5: KDE Consensus | 16% | 46% | 38% | 0.5623 | 222,292 | 172,292 |

---

## Key Takeaways

**Research stays low (14–19%).** Log diminishing returns make anything beyond ~19% inefficient.

**Scale absorbs the non-Speed budget** (linear, no diminishing returns) — ranges 39–61% across scenarios.

**Speed is the pivotal decision** — optimal ranges from 20% (GPT) to 47% (Gemini).

**Best case: Dist 4 (Claude) — Net PnL 288,940.** Most players stay under 25%; investing 25% yourself tops most of the crowd with a 0.67 multiplier while freeing 57% for Scale.

**Worst case: Dist 1 (GPT Bimodal) — Net PnL 130,050.** The second peak at ~56% means even 20% Speed is outranked by half the field (0.32 multiplier). Counter-strategy: abandon Speed, maximize Scale.

**KDE Consensus (Dist 5, Normal half-weighted) — R=16%, S=46%, V=38%, Net PnL 172,292.**

---

## Files

```
ROUND2_Manual/
├── Directions.txt          # Challenge rules
├── Speed Distributions.txt # Distribution descriptions
├── analyze_invest.py       # Python analysis + optimization
├── speed_distributions.png # Distribution visuals (2×3 grid)
├── pnl_sensitivity.png     # PnL sensitivity curves (2×3 grid)
├── breakeven_multipliers.png # Breakeven speed multiplier curve
├── breakeven_plot.py       # Breakeven analysis script
├── invest_and_expand.py    # Monte Carlo simulation approach
├── Ignore.txt              # Random thoughts and ideas
└── README.md               # This file
```
