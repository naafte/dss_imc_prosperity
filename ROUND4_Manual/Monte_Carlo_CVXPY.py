import numpy as np
import pandas as pd
import cvxpy as cp

def run_convex_intara_optimizer(paths=100_000, seed=42):
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1. Market Parameters & Instrument Definitions
    # ------------------------------------------------------------------
    S0    = 50.0
    sigma = 2.51 
    dt    = 1.0 / (252 * 4)
    
    steps_2w = 10 * 4
    steps_3w = 15 * 4

    # Definition order matches the Quick Machine-Readable Summary
    instruments = [
        {"name": "AC_50_P",   "bid": 12.00,  "ask": 12.05,  "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_50_C",   "bid": 12.00,  "ask": 12.05,  "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_35_P",   "bid": 4.33,   "ask": 4.35,   "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_40_P",   "bid": 6.50,   "ask": 6.55,   "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_45_P",   "bid": 9.05,   "ask": 9.10,   "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_60_C",   "bid": 8.80,   "ask": 8.85,   "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_50_P_2", "bid": 9.70,   "ask": 9.75,   "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_50_C_2", "bid": 9.70,   "ask": 9.75,   "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_50_CO",  "bid": 22.20,  "ask": 22.30,  "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_40_BP",  "bid": 5.00,   "ask": 5.10,   "bid_sz": 50,  "ask_sz": 50},
        {"name": "AC_45_KO",  "bid": 0.150,  "ask": 0.175,  "bid_sz": 500, "ask_sz": 500},
        {"name": "Spot",      "bid": 49.975, "ask": 50.025, "bid_sz": 200, "ask_sz": 200},
    ]
    
    num_assets = len(instruments)
    bids       = np.array([inst["bid"] for inst in instruments])
    asks       = np.array([inst["ask"] for inst in instruments])
    bid_sizes  = np.array([inst["bid_sz"] for inst in instruments])
    ask_sizes  = np.array([inst["ask_sz"] for inst in instruments])

    # ------------------------------------------------------------------
    # 2. GBM Path Simulation (Antithetic Variates)
    # ------------------------------------------------------------------
    print(f"Simulating {paths:,} paths...")
    Z_half      = np.random.standard_normal((paths // 2, steps_3w))
    Z           = np.vstack([Z_half, -Z_half])
    log_ret     = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * Z
    price_paths = S0 * np.cumprod(np.exp(log_ret), axis=1)

    S_2w     = price_paths[:, steps_2w - 1]
    S_3w     = price_paths[:, steps_3w - 1]
    path_min = np.min(price_paths, axis=1)

    # ------------------------------------------------------------------
    # 3. Build the Master Payoff Matrix (Paths x Instruments)
    # ------------------------------------------------------------------
    P = np.zeros((paths, num_assets))
    
    P[:, 0]  = np.maximum(50 - S_3w, 0)                  # AC_50_P
    P[:, 1]  = np.maximum(S_3w - 50, 0)                  # AC_50_C
    P[:, 2]  = np.maximum(35 - S_3w, 0)                  # AC_35_P
    P[:, 3]  = np.maximum(40 - S_3w, 0)                  # AC_40_P
    P[:, 4]  = np.maximum(45 - S_3w, 0)                  # AC_45_P
    P[:, 5]  = np.maximum(S_3w - 60, 0)                  # AC_60_C
    P[:, 6]  = np.maximum(50 - S_2w, 0)                  # AC_50_P_2 (14d)
    P[:, 7]  = np.maximum(S_2w - 50, 0)                  # AC_50_C_2 (14d)
    
    # Chooser logic: at t=14, auto-converts to ITM. 
    # If S_2w > 50 -> Call (max(S_3w - 50, 0)). If S_2w <= 50 -> Put (max(50 - S_3w, 0))
    P[:, 8]  = np.where(S_2w > 50, np.maximum(S_3w - 50, 0), np.maximum(50 - S_3w, 0))
    
    P[:, 9]  = np.where(S_3w < 40, 10.0, 0.0)            # AC_40_BP
    P[:, 10] = np.where(path_min >= 35.0, np.maximum(45 - S_3w, 0), 0.0) # AC_45_KO
    
    P[:, 11] = S_3w                                      # Spot

    # ------------------------------------------------------------------
    # 4. Statistical Edge & Covariance Setup
    # ------------------------------------------------------------------
    mu_P = np.mean(P, axis=0)
    
    # Covariance matrix of the payoffs
    Sigma = np.cov(P, rowvar=False)
    
    # Ensure Matrix is strictly Positive Semi-Definite for CVXPY
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 1e-8)
    Sigma_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Edges
    long_edge  = mu_P - asks    # Expected PnL per unit bought
    short_edge = bids - mu_P    # Expected PnL per unit sold

    # ------------------------------------------------------------------
    # 5. Convex Optimization Formulation
    # ------------------------------------------------------------------
    print("Running Convex Optimization mapping Pareto frontier...\n")
    
    # Decision Variables
    w_long  = cp.Variable(num_assets, nonneg=True)
    w_short = cp.Variable(num_assets, nonneg=True)
    
    # Constraints: You can't buy/sell more than order book size
    constraints = [
        w_long  <= ask_sizes,
        w_short <= bid_sizes
    ]
    
    # Objective Components
    expected_pnl = cp.sum(cp.multiply(w_long, long_edge) + cp.multiply(w_short, short_edge))
    
    # Net position dictates the variance of the final payoff 
    net_position = w_long - w_short
    variance     = cp.quad_form(net_position, Sigma_psd)
    
    # Sweep risk aversion penalty to map the Pareto frontier
    risk_penalties = np.logspace(-5, 0, 15)[::-1] 
    risk_penalties = np.append(risk_penalties, 0) # Append 0 for absolute max return
    
    results = []
    
    for gamma in risk_penalties:
        # Objective: Maximize (Expected Return - Risk_Penalty * Variance)
        objective = cp.Maximize(expected_pnl - gamma * variance)
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.SCS)
            
            w_l = np.round(w_long.value).astype(int)
            w_s = np.round(w_short.value).astype(int)
            
            # Recalculate true stats based on discrete integer quantities
            net_pos_val = w_l - w_s
            pnl_paths   = (P - asks) @ w_l + (bids - P) @ w_s
            
            mean_pnl = np.mean(pnl_paths)
            std_pnl  = np.std(pnl_paths)
            
            res = {
                "Gamma": gamma,
                "Exp_PnL": mean_pnl,
                "Std_Dev": std_pnl,
                "Sharpe": mean_pnl / std_pnl if std_pnl > 0 else 0,
                "Positions": net_pos_val # Positive = Long, Negative = Short
            }
            results.append(res)
        except Exception as e:
            continue

    # ------------------------------------------------------------------
    # 6. Formatting Output
    # ------------------------------------------------------------------
    # Filter for unique (Exp_PnL, Std_Dev) pairs since multiple gammas might yield identical integer portfolios
    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=["Exp_PnL"]).sort_values("Std_Dev").reset_index(drop=True)
    
    # Format the portfolio positions
    pos_df = pd.DataFrame(df["Positions"].to_list(), columns=[inst["name"] for inst in instruments])
    df = pd.concat([df.drop(columns=["Positions", "Gamma"]), pos_df], axis=1)
    
    # Filter out columns where position is strictly 0 across the entire frontier
    cols_with_pos = pos_df.columns[(pos_df != 0).any()]
    display_cols = ['Exp_PnL', 'Std_Dev', 'Sharpe'] + list(cols_with_pos)

    return df[display_cols]


if __name__ == "__main__":
    frontier_df = run_convex_intara_optimizer(paths=100_000)
    
    print("=" * 100)
    print("OPTIMIZED PARETO FRONTIER (Includes Short Selling)")
    print("=" * 100)
    print(frontier_df.to_string(index=False, float_format="%.2f"))
    print("=" * 100)