import numpy as np
import pandas as pd


def run_intara_optimizer(paths=100_000, seed=42):
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1. Market & Engine Parameters (from problem statement)
    # ------------------------------------------------------------------
    S0    = 50.0        # mid-market spot  (bid=49.975, ask=50.025)
    sigma = 2.51        # 251% annualized vol
    dt    = 1.0 / (252 * 4)  # 4 steps/trading day, 252 days/year

    steps_2w = 10 * 4   # 40 steps = 2 trading weeks = 14 Solvenarian days
    steps_3w = 15 * 4   # 60 steps = 3 trading weeks = 21 Solvenarian days

    # ------------------------------------------------------------------
    # 2. GBM path simulation (antithetic variates for variance reduction)
    # ------------------------------------------------------------------
    print(f"Simulating {paths:,} paths...")
    Z_half      = np.random.standard_normal((paths // 2, steps_3w))
    Z           = np.vstack([Z_half, -Z_half])
    log_ret     = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * Z
    price_paths = S0 * np.cumprod(np.exp(log_ret), axis=1)

    S_2w     = price_paths[:, steps_2w - 1]   # spot at 2-week mark
    S_3w     = price_paths[:, steps_3w - 1]   # spot at 3-week mark
    path_min = np.min(price_paths, axis=1)    # running min (barrier check)

    # ------------------------------------------------------------------
    # 3. Per-unit edge vectors: (payoff - ask_price) for each instrument
    #    Positive mean  =>  favourable to buy at ask
    # ------------------------------------------------------------------

    # 21-day vanilla options (K=50 put has the largest displayed edge: +2.71)
    e_50p = np.maximum(50 - S_3w, 0)              - 12.05   # AC_50_P   put  K=50  ask=12.05
    e_35p = np.maximum(35 - S_3w, 0)              -  4.35   # AC_35_P   put  K=35  ask=4.35
    e_60c = np.maximum(S_3w - 60, 0)              -  8.85   # AC_60_C   call K=60  ask=8.85

    # 14-day vanilla (NOT the same as the 14d put used in the chooser arb)
    e_c2  = np.maximum(S_2w - 50, 0)              -  9.75   # AC_50_C_2 call K=50  ask=9.75

    # Binary put: pays 10 if S_3w < 40, else 0
    e_bp  = np.where(S_3w < 40, 10.0, 0.0)        -  5.10   # AC_40_BP  bin-put K=40 ask=5.10

    # Knock-out put: knocked out (worthless) if price ever touches < 35
    survived = path_min >= 35.0
    e_ko  = (np.where(survived, np.maximum(45 - S_3w, 0), 0.0)
             - 0.175)                                         # AC_45_KO  KO-put K=45 barrier=35 ask=0.175

    # Spot hedge: +1 unit = SHORT at bid 49.975  =>  PnL = 49.975 - S_3w
    #             -1 unit = LONG  at ask 50.025  =>  PnL ≈ S_3w - 50.025
    # Represented uniformly as short-spot edge; use negative sizes to go long.
    e_spot = 49.975 - S_3w   # per unit short

    # ------------------------------------------------------------------
    # 4. Riskless arbitrage — locked in regardless of path
    #    Sell 50 AC_50_CO @ 22.20  (chooser: choose-at=14d, expire=21d, K=50)
    #    Buy  50 AC_50_C  @ 12.05  (21d call, K=50)
    #    Buy  50 AC_50_P_2@  9.75  (14d put,  K=50)
    #
    #    Proof: Chooser(t=14) = max(C_7d, P_7d) = C_7d + max(0, K-S_14)  [put-call parity, r=0]
    #    Replication at t=14: long 21d-call (7d remaining) + long 14d-put (just expired) => exact match
    #    Edge per unit = 22.20 - 12.05 - 9.75 = 0.40
    #    For 50 units  => +20.00  (path-independent)
    # ------------------------------------------------------------------
    arb_pnl = 20.0

    # ------------------------------------------------------------------
    # 5. Print per-unit MC summary
    # ------------------------------------------------------------------
    print("\nMC per-unit expected edges  (mean payoff - ask):")
    for label, e in [
        ("AC_50_P  21d put   K=50 ask=12.05", e_50p),
        ("AC_35_P  21d put   K=35 ask=4.35 ", e_35p),
        ("AC_60_C  21d call  K=60 ask=8.85 ", e_60c),
        ("AC_50_C_2 2w call  K=50 ask=9.75 ", e_c2),
        ("AC_40_BP  bin-put  K=40 ask=5.10 ", e_bp),
        ("AC_45_KO  KO-put   K=45 ask=0.175", e_ko),
        ("Spot short at bid 49.975         ", e_spot),
    ]:
        print(f"  {label}:  mean={e.mean():+.4f}   std={e.std():.4f}")

    # ------------------------------------------------------------------
    # 6. Grid search over position sizes
    #    Volume constraints from the problem sheet:
    #      Vanilla / exotics   max  50 units
    #      AC_45_KO            max 500 units
    #      Spot                max 100 units long or short
    #
    #    Grid kept coarse so total combos stay manageable (~15 k).
    #    Chunked evaluation avoids building a (paths × combos) matrix in RAM.
    # ------------------------------------------------------------------
    g_50p  = np.arange(0,  51, 10)    # 0,10,20,30,40,50       → 6 values
    g_35p  = np.array([0, 25, 50])    #                         → 3
    g_60c  = np.array([0, 25, 50])    #                         → 3
    g_c2   = np.array([0, 25, 50])    #                         → 3
    g_bp   = np.array([0, 25, 50])    #                         → 3
    g_ko   = np.arange(0, 501, 100)   # 0,100,...,500           → 6
    g_spot = np.arange(-50, 51, 25)   # -50,-25,0,25,50 (neg=long, pos=short) → 5
    # Total: 6*3*3*3*3*6*5 = 14,580 combinations

    axes   = np.meshgrid(g_50p, g_35p, g_60c, g_c2, g_bp, g_ko, g_spot, indexing='ij')
    combos = np.column_stack([a.ravel() for a in axes])    # (14580, 7)
    n_combos = len(combos)
    print(f"\nEvaluating {n_combos:,} portfolio combinations...")

    # Edge matrix: rows=paths, cols=instruments (same order as combos columns)
    E = np.column_stack([e_50p, e_35p, e_60c, e_c2, e_bp, e_ko, e_spot]).astype(np.float32)
    combos_f32 = combos.astype(np.float32)

    # Chunked stats  (keeps peak RAM ~200 MB per chunk)
    chunk_sz = 500
    exp_pnl_arr = np.empty(n_combos, dtype=np.float32)
    std_arr     = np.empty(n_combos, dtype=np.float32)
    p10_arr     = np.empty(n_combos, dtype=np.float32)
    p5_arr      = np.empty(n_combos, dtype=np.float32)

    for i in range(0, n_combos, chunk_sz):
        j     = min(i + chunk_sz, n_combos)
        block = E @ combos_f32[i:j].T + arb_pnl   # (paths, chunk)
        exp_pnl_arr[i:j] = block.mean(axis=0)
        std_arr[i:j]     = block.std(axis=0)
        p10_arr[i:j]     = np.percentile(block, 10, axis=0)
        p5_arr[i:j]      = np.percentile(block,  5, axis=0)

    # ------------------------------------------------------------------
    # 7. Build results dataframe and Pareto frontier
    # ------------------------------------------------------------------
    results = pd.DataFrame({
        'AC_50_P':   combos[:, 0].astype(int),
        'AC_35_P':   combos[:, 1].astype(int),
        'AC_60_C':   combos[:, 2].astype(int),
        'AC_50_C_2': combos[:, 3].astype(int),
        'AC_40_BP':  combos[:, 4].astype(int),
        'AC_45_KO':  combos[:, 5].astype(int),
        'Spot_Net':  combos[:, 6].astype(int),   # pos=short, neg=long
        'Exp_PnL':   exp_pnl_arr.astype(float),
        'Std_Dev':   std_arr.astype(float),
        'Sharpe':    np.divide(exp_pnl_arr, std_arr, out=np.zeros_like(exp_pnl_arr), where=std_arr != 0).astype(float),
        'P10':       p10_arr.astype(float),       # 10th-percentile worst case
        'P5':        p5_arr.astype(float),        # 5th-percentile worst case
    })

    # Pareto: best Exp_PnL in each 50-unit std-dev risk bucket
    results['Risk_Bin'] = (results['Std_Dev'] // 50) * 50
    pareto = results.loc[results.groupby('Risk_Bin')['Exp_PnL'].idxmax()]
    pareto = pareto[pareto['Exp_PnL'] > arb_pnl]   # must beat arb-only baseline
    pareto = pareto.sort_values('Exp_PnL', ascending=False).reset_index(drop=True)

    return pareto, results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pareto, all_results = run_intara_optimizer(paths=100_000)

    display_cols = [
        'Exp_PnL', 'Std_Dev', 'Sharpe', 'P10', 'P5',
        'AC_50_P', 'AC_35_P', 'AC_60_C', 'AC_50_C_2', 'AC_40_BP', 'AC_45_KO', 'Spot_Net',
    ]

    print("\n" + "=" * 120)
    print("PARETO FRONTIER  --  BEST EXPECTED PNL PER RISK LEVEL  (Std_Dev bucket = 50)")
    print("=" * 120)
    print(pareto[display_cols].to_string(index=True, float_format="%.2f"))
    print("=" * 120)

    print("\nBASE ARBITRAGE  (execute this regardless of which row you pick):")
    print("  Sell 50  AC_50_CO   @ 22.20  bid   [chooser, choose-at=14d, expire=21d, K=50]")
    print("  Buy  50  AC_50_C    @ 12.05  ask   [21-day call, K=50]")
    print("  Buy  50  AC_50_P_2  @  9.75  ask   [14-day put,  K=50]")
    print("  Net credit = +20.00  (riskless by put-call parity, path-independent)")

    print("\nCOLUMN GUIDE:")
    print("  Exp_PnL  - mean PnL across 100k simulated paths  (includes arb +20)")
    print("  Std_Dev  - 1-sigma spread of outcomes")
    print("  Sharpe   - Exp_PnL / Std_Dev  (higher = better risk-adjusted)")
    print("  P10/P5   - 10th/5th percentile worst-case PnL  (downside risk)")
    print("  Spot_Net - positive = SHORT spot at bid 49.975")
    print("             negative = LONG  spot at ask 50.025")
