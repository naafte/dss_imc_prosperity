# ROUND 4 — Aether Crystal Options (Manual Trading)

IMC Prosperity Round 4 manual trading challenge. The task is to trade a set of exotic options on a single underlying (Aether Crystal, spot ~50) using a BSM model with `sigma=2.51`, `r=0`.

## Files

| File | Description |
|------|-------------|
| `Directions.txt` | Full problem statement |
| `Directions_compressed.txt` | Condensed product spec and payoff rules |
| `Monte_Carlo.py` | Early Monte Carlo PnL simulation (no optimizer) |
| `Monte_Carlo_CVXPY.py` | Mean–CVaR portfolio optimizer via CVXPY (main analysis) |
| `cvxpy_solution.txt` | Output of `Monte_Carlo_CVXPY.py` — Pareto frontier results |

## Products

| Name | Type | Strike | Expiry | Extra | Bid | Ask | Max Size |
|------|------|--------|--------|-------|-----|-----|----------|
| AC_50_P | Vanilla put | 50 | 21d | — | 12.00 | 12.05 | 50 |
| AC_50_C | Vanilla call | 50 | 21d | — | 12.00 | 12.05 | 50 |
| AC_35_P | Vanilla put | 35 | 21d | — | 4.33 | 4.35 | 50 |
| AC_40_P | Vanilla put | 40 | 21d | — | 6.50 | 6.55 | 50 |
| AC_45_P | Vanilla put | 45 | 21d | — | 9.05 | 9.10 | 50 |
| AC_60_C | Vanilla call | 60 | 21d | — | 8.80 | 8.85 | 50 |
| AC_50_P_2 | Vanilla put | 50 | 14d | — | 9.70 | 9.75 | 50 |
| AC_50_C_2 | Vanilla call | 50 | 14d | — | 9.70 | 9.75 | 50 |
| AC_50_CO | Chooser | 50 | 21d | choose at 14d | 22.20 | 22.30 | 50 |
| AC_40_BP | Binary put | 40 | 21d | pays 10 if S<40 | 5.00 | 5.10 | 50 |
| AC_45_KO | Knock-out put | 45 | 21d | barrier=35 | 0.150 | 0.175 | 500 |
| Spot | Underlying | — | — | — | 49.975 | 50.025 | 200 |

## Key Findings

### Risk-free arbitrage (chooser mispricing)
Chooser parity (Rubinstein 1991, r=0): `Chooser = Call(K,T) + Put(K,t_c)`

Fair value = 12.027 + 9.871 = **21.90**, but market quotes **22.20/22.30**.

**Trade**: Short chooser at 22.20 + buy Call(21d) at 12.05 + buy Put(14d) at 9.75 → **+0.40/unit locked profit**, no risk.

### Recommended low-risk portfolio (Pareto row 2)
| Position | Units |
|----------|-------|
| Short AC_40_P (40-strike put) | -50 |
| Long AC_45_P (45-strike put) | +50 |
| Short AC_40_BP (binary put) | -25 |

Bull put spread + short binary. Expected PnL ≈ 4.6, CVaR95 ≈ 5.0 (Ret/CVaR = 0.92).

### Why high-return rows have large CVaR
High-return rows short large quantities of AC_45_KO (up to 500 units). The KO put pays up to ~10/unit whenever the down-barrier isn't breached, so 500 short units can generate ~5000 in tail losses. The optimizer accepts this for higher expected PnL at low risk-aversion.

## Methodology (`Monte_Carlo_CVXPY.py`)

1. Simulate 100,000 GBM paths (antithetic variates) with `sigma=2.51`, `dt=1/(252×4)`.
2. Compute per-path payoffs for all 12 instruments, including:
   - Chooser: pick call or put based on moneyness at day 14 (valid when r=0)
   - KO put: apply Broadie-Glasserman-Kou (1997) discrete-barrier correction
3. Solve Mean–CVaR LP (Rockafellar-Uryasev 2000) via CVXPY/CLARABEL across 21 risk-aversion levels.
4. Filter to Pareto-optimal portfolios (non-dominated in E[PnL] vs CVaR95 space).
