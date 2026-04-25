# Round 3 Manual Trading — The Celestial Gardeners' Guild

## Problem Summary

Trade against a secret number of counterparties, each with a **reserve price** uniformly distributed between **670 and 920** in increments of 5. You can sell all acquired product the next day at the fair price of **920**.

You submit **two bids**:
- **Bid 1**: If higher than a counterparty's reserve price, you trade at Bid 1.
- **Bid 2**: If higher than a counterparty's reserve price *and* higher than the mean of all players' second bids, you trade at Bid 2. If Bid 2 is above the reserve price but at or below the mean, your PnL is penalised by:

$$\left(\frac{920 - \bar{b}_2}{920 - b_2}\right)^3$$

where $\bar{b}_2$ is the mean second bid across all players and $b_2$ is your second bid.

The penalty cancels exactly when `b2 = avg_b2`, making the mean of the second-bid distribution the natural target.

## Final Submission

| Bid | Value |
|-----|-------|
| Bid 1 | **795** |
| Bid 2 | **862** |

## Analysis

### Competitor Second-Bid Distributions

Three distributions were modelled based on AI model predictions (~3,000 players, whole-number bids only):

| Distribution | Description | Mean |
|---|---|---|
| GPT | Multi-modal; peaks at 858 (14%) and 865 (10%), range rows bell at multiples of 10 with mini-peaks at multiples of 5 | ~860 |
| Gemini | Tight cluster; 60% at 861–862, 20% at 856–860, 20% at 866+ | ~862 |
| Claude | Smooth bell; peaks at 860 (22%), tails from <820 to 910+ | ~861 |
| **Combined** | KDE consensus of all three | **~861** |

### Key Statistics (Combined)

| Stat | Value |
|---|---|
| Mean | 861.0 |
| Median | 862.0 |
| Std | 11.6 |
| P25 / P75 | 858 / 866 |

### Reasoning

- The penalty structure means the optimal Bid 2 is the **expected mean of all players' Bid 2s**.
- All three model distributions converge on a mean of **~861**, with strong consensus around 861–862.
- Bid 2 = **862** targets just above the combined mean, capturing the majority of counterparties while staying within the no-penalty zone.
- The continuous Nash equilibrium is ~857.5; the integer Nash equilibrium is **860** — 862 hedges slightly above this.

## Files

| File | Description |
|---|---|
| `Directions.txt` | Original problem statement |
| `bid2_distributions.txt` | Distribution specifications for GPT, Gemini, and Claude models |
| `viz_bid2_distributions.py` | Script to generate the distribution visualizations |
| `bid2_distributions_viz.png` | Final visualization (3 individual distributions + KDE consensus + stats) |
| `analyze_bid2.py` | Additional analysis script |
| `bid2_analysis.png` | Additional analysis output |
| `notes.txt` | Personal notes: distributions are right-skewed (safer to overbid than underbid), integer Nash equilibrium 860|
