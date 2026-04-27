import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.stats import norm

# ── helpers ───────────────────────────────────────────────────────────────────

def _bsm(S, K, T, sigma, r=0.0, flag='call'):
    if T <= 0:
        return max(S - K, 0.0) if flag == 'call' else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _pareto_filter(df, ret_col, risk_col):
    """Keep only non-dominated rows (higher ret_col AND lower risk_col)."""
    vals = df[[ret_col, risk_col]].values
    keep = []
    for i, (r_i, s_i) in enumerate(vals):
        dominated = any(
            vals[j, 0] >= r_i and vals[j, 1] <= s_i
            and (vals[j, 0] > r_i or vals[j, 1] < s_i)
            for j in range(len(vals)) if j != i
        )
        keep.append(not dominated)
    return df[keep].reset_index(drop=True)


# ── analytical cross-instrument checks ───────────────────────────────────────

def run_analytical_checks(S0=50.0, sigma=2.51, r=0.0):
    """
    Model-independent parity checks that expose mispricings before MC.
    The chooser parity identity below requires r=0 (put-call parity gives
    C - P = S - K, so chooser = max(C,P) = C + max(P-C,0) = C + Put(K,t_c)).
    """
    T21, T14 = 15 / 252, 10 / 252

    c21 = _bsm(S0, 50, T21, sigma, r, 'call')
    p21 = _bsm(S0, 50, T21, sigma, r, 'put')
    c14 = _bsm(S0, 50, T14, sigma, r, 'call')
    p14 = _bsm(S0, 50, T14, sigma, r, 'put')

    # Chooser(K, T, t_c) = Call(K, T) + Put(K, t_c)  [r=0, Rubinstein 1991]
    chooser_fair = c21 + p14

    mkt = {
        'c21': (12.00, 12.05), 'p21': (12.00, 12.05),
        'c14': (9.70,   9.75), 'p14': (9.70,   9.75),
        'cho': (22.20, 22.30),
    }

    # Arb: short chooser at bid, cover with long Call(21d) + Put(14d) at ask
    arb_short = mkt['cho'][0] - (mkt['c21'][1] + mkt['p14'][1])
    # Arb: long chooser at ask, cover with short Call(21d) + Put(14d) at bid
    arb_long  = (mkt['c21'][0] + mkt['p14'][0]) - mkt['cho'][1]

    print("=" * 70)
    print("ANALYTICAL CROSS-INSTRUMENT CHECKS")
    print("=" * 70)
    print(f"\nBSM fair values (sigma={sigma}, r={r}):")
    print(f"  Call(50,21d) = {c21:.4f}   Put(50,21d) = {p21:.4f}   [mkt 12.00/12.05]")
    print(f"  Call(50,14d) = {c14:.4f}   Put(50,14d) = {p14:.4f}   [mkt  9.70/ 9.75]")
    print(f"\nChooser parity  Call(50,21d)+Put(50,14d) = {chooser_fair:.4f}  [mkt 22.20/22.30]")
    flag_s = "  <-- RISK-FREE ARB" if arb_short > 0 else ""
    flag_l = "  <-- RISK-FREE ARB" if arb_long  > 0 else ""
    print(f"  Short chooser + long  Call+Put: {arb_short:+.4f}/unit{flag_s}")
    print(f"  Long  chooser + short Call+Put: {arb_long:+.4f}/unit{flag_l}")
    print("=" * 70 + "\n")


# ── main optimizer ────────────────────────────────────────────────────────────

def run_convex_optimizer(paths: int = 100_000, seed: int = 42, cvar_alpha: float = 0.95):
    """
    Mean–CVaR portfolio optimization across a Pareto frontier.

    Replaces mean-variance with CVaR (Rockafellar-Uryasev 2000), which is:
      - Linear in scenario weights → LP, not QP (exact solver, no PSD hacks)
      - Coherent: penalises tail losses, not symmetric upside variance
      - Appropriate for skewed option payoffs

    Paths=100k creates an LP with ~100k auxiliary variables. Reduce to 20k
    for faster sweeps; antithetic variates keep estimator quality reasonable.
    """
    np.random.seed(seed)

    S0    = 50.0
    sigma = 2.51
    r     = 0.0
    dt    = 1.0 / (252 * 4)          # quarter-day step

    steps_2w = 10 * 4                 # 40 steps = 14 calendar days
    steps_3w = 15 * 4                 # 60 steps = 21 calendar days

    instruments = [
        {"name": "AC_50_P",   "bid": 12.00,  "ask": 12.05,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_50_C",   "bid": 12.00,  "ask": 12.05,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_35_P",   "bid":  4.33,  "ask":  4.35,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_40_P",   "bid":  6.50,  "ask":  6.55,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_45_P",   "bid":  9.05,  "ask":  9.10,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_60_C",   "bid":  8.80,  "ask":  8.85,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_50_P_2", "bid":  9.70,  "ask":  9.75,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_50_C_2", "bid":  9.70,  "ask":  9.75,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_50_CO",  "bid": 22.20,  "ask": 22.30,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_40_BP",  "bid":  5.00,  "ask":  5.10,  "bid_sz":  50, "ask_sz":  50},
        {"name": "AC_45_KO",  "bid":  0.150, "ask":  0.175, "bid_sz": 500, "ask_sz": 500},
        {"name": "Spot",      "bid": 49.975, "ask": 50.025, "bid_sz": 200, "ask_sz": 200},
    ]

    num_assets = len(instruments)
    bids      = np.array([x["bid"]    for x in instruments])
    asks      = np.array([x["ask"]    for x in instruments])
    bid_sizes = np.array([x["bid_sz"] for x in instruments])
    ask_sizes = np.array([x["ask_sz"] for x in instruments])

    # ── 1. GBM path simulation (antithetic variates) ──────────────────────
    print(f"Simulating {paths:,} paths...")
    Z_half      = np.random.standard_normal((paths // 2, steps_3w))
    Z           = np.vstack([Z_half, -Z_half])
    log_ret     = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * Z
    price_paths = S0 * np.cumprod(np.exp(log_ret), axis=1)

    S_2w     = price_paths[:, steps_2w - 1]
    S_3w     = price_paths[:, steps_3w - 1]
    path_min = np.min(price_paths, axis=1)

    # ── 2. Payoff matrix ──────────────────────────────────────────────────
    P = np.zeros((paths, num_assets))

    P[:, 0] = np.maximum(50 - S_3w, 0)
    P[:, 1] = np.maximum(S_3w - 50, 0)
    P[:, 2] = np.maximum(35 - S_3w, 0)
    P[:, 3] = np.maximum(40 - S_3w, 0)
    P[:, 4] = np.maximum(45 - S_3w, 0)
    P[:, 5] = np.maximum(S_3w - 60, 0)
    P[:, 6] = np.maximum(50 - S_2w, 0)
    P[:, 7] = np.maximum(S_2w - 50, 0)

    # Chooser: at t_c=14d choose call or put. The ITM rule (S>K → call)
    # is equivalent to comparing BSM values only when r=0, because
    # put-call parity gives C-P = S-K, so C>P iff S>K. If r≠0, replace
    # with explicit BSM comparison.
    P[:, 8] = np.where(S_2w > 50,
                       np.maximum(S_3w - 50, 0),
                       np.maximum(50 - S_3w, 0))

    P[:, 9] = np.where(S_3w < 40, 10.0, 0.0)

    # Knockput with Broadie-Glasserman-Kou (1997) discrete-barrier correction.
    # Monitoring at 4 steps/day undercounts barrier crossings versus continuous.
    # BGK shifts the effective down-barrier up by β·σ·√dt (β=0.5826),
    # making the barrier easier to breach and reducing the KO put's value
    # to match its continuous-monitoring fair price.
    bgk_barrier = 35.0 * np.exp(0.5826 * sigma * np.sqrt(dt))
    P[:, 10] = np.where(path_min >= bgk_barrier, np.maximum(45 - S_3w, 0), 0.0)

    P[:, 11] = S_3w   # Spot payoff; edge = E[S_3w] - ask ≈ -0.025 (nearly neutral)

    # ── 3. CVaR-LP optimization (Rockafellar-Uryasev 2000) ───────────────
    print(f"Running CVaR-{int(cvar_alpha*100)} optimization over Pareto frontier...\n")

    # Decision variables
    w_long  = cp.Variable(num_assets, nonneg=True)   # units bought  at ask
    w_short = cp.Variable(num_assets, nonneg=True)   # units sold    at bid
    z       = cp.Variable()                           # scalar VaR level
    u       = cp.Variable(paths, nonneg=True)         # per-path excess loss

    # Per-path P&L: shape (paths,), linear in w_long, w_short
    pnl  = (P - asks) @ w_long + (bids - P) @ w_short
    loss = -pnl

    expected_pnl = cp.sum(pnl) / paths
    cvar_expr    = z + cp.sum(u) / ((1.0 - cvar_alpha) * paths)

    base_constraints = [
        w_long  <= ask_sizes,
        w_short <= bid_sizes,
        u       >= loss - z,    # u_i ≥ max(loss_i - z, 0), enforced with nonneg=True
    ]

    # Sweep risk-aversion (γ=0 → max return, large γ → min CVaR)
    risk_penalties = list(np.logspace(-3, 1, 20)[::-1]) + [0.0]

    results = []
    for gamma in risk_penalties:
        prob = cp.Problem(cp.Maximize(expected_pnl - gamma * cvar_expr), base_constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            pass

        if w_long.value is None:
            continue

        # Clip after rounding to guarantee capacity constraints are not violated
        w_l = np.clip(np.round(w_long.value).astype(int),  0, ask_sizes.astype(int))
        w_s = np.clip(np.round(w_short.value).astype(int), 0, bid_sizes.astype(int))

        pnl_paths = (P - asks) @ w_l + (bids - P) @ w_s
        mean_pnl  = np.mean(pnl_paths)
        std_pnl   = np.std(pnl_paths)

        # Realised CVaR from sorted scenario losses
        losses_sorted = np.sort(-pnl_paths)
        tail_start    = int(np.ceil(cvar_alpha * paths))
        realized_cvar = float(np.mean(losses_sorted[tail_start:])) if tail_start < paths else float(losses_sorted[-1])

        results.append({
            "Gamma":   gamma,
            "Exp_PnL": mean_pnl,
            "Std_Dev": std_pnl,
            f"CVaR{int(cvar_alpha*100)}": realized_cvar,
            "Sharpe":  mean_pnl / std_pnl if std_pnl > 0 else 0.0,
            "Ret/CVaR": mean_pnl / realized_cvar if realized_cvar > 0 else np.inf,
            "Positions": w_l - w_s,
        })

    # ── 4. Build true Pareto frontier ─────────────────────────────────────
    # Deduplicate on both axes before filtering, not just one.
    cvar_col = f"CVaR{int(cvar_alpha*100)}"
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=["Exp_PnL", cvar_col])
    df = _pareto_filter(df, ret_col="Exp_PnL", risk_col=cvar_col)
    df = df.sort_values("Exp_PnL").reset_index(drop=True)

    pos_df      = pd.DataFrame(df["Positions"].to_list(), columns=[x["name"] for x in instruments])
    active_cols = pos_df.columns[(pos_df != 0).any()]
    stat_cols   = ["Exp_PnL", "Std_Dev", cvar_col, "Sharpe", "Ret/CVaR"]
    return pd.concat([df[stat_cols], pos_df[active_cols]], axis=1)


if __name__ == "__main__":
    run_analytical_checks()

    frontier_df = run_convex_optimizer(paths=100_000)

    cvar_col = "CVaR95"
    print("=" * 120)
    print(f"PARETO FRONTIER  (E[PnL] vs {cvar_col} — non-dominated portfolios only)")
    print("=" * 120)
    print(frontier_df.to_string(index=False, float_format="%.3f"))
    print("=" * 120)
