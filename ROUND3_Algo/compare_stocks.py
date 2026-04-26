import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def calculate_hurst_exponent(time_series, max_lag=100):
    """
    Calculates the Hurst Exponent of a time series.
    H < 0.5: Mean-reverting (Pairs Trading)
    H = 0.5: Random Walk
    H > 0.5: Trending (Directional Trading)
    """
    lags = range(2, max_lag)
    
    # Calculate the variance of the lagged differences
    # Variance of a random walk scales linearly with time (lag)
    # Variance of a mean-reverting series scales slower than time
    variances = [np.var(time_series[lag:] - time_series[:-lag]) for lag in lags]
    
    # Fit a linear regression to the log-log plot
    # log(Variance) = 2H * log(Lag) + constant
    poly = np.polyfit(np.log(lags), np.log(variances), 1)
    
    # The slope is 2H, so we divide by 2 to get H
    hurst = poly[0] / 2
    return hurst

def analyze_pair(csv_file_path):
    """
    Reads Prosperity historical data and runs cointegration & mean-reversion tests.
    Assumes a CSV format with columns: 'timestamp', 'product', 'mid_price'
    """
    print(f"Loading data from {csv_file_path}...\n")
    
    # 1. Load and format the data
    df = pd.read_csv(csv_file_path, sep=';') # Prosperity often uses ';' delimited files
    
    # Filter for our two assets
    hp_df = df[df['product'] == 'HYDROGEL_PACK'].set_index('timestamp')
    vfe_df = df[df['product'] == 'VELVETFRUIT_EXTRACT'].set_index('timestamp')
    
    # Align the time series
    data = pd.DataFrame({
        'HP': hp_df['mid_price'],
        'VFE': vfe_df['mid_price']
    }).dropna()

    # 2. Find the optimal Hedge Ratio using Ordinary Least Squares (OLS)
    # This answers: How many units of VFE equals 1 unit of HP?
    X = sm.add_constant(data['VFE'])
    model = sm.OLS(data['HP'], X).fit()
    hedge_ratio = model.params['VFE']
    
    print(f"--- 1. HEDGE RATIO ---")
    print(f"Optimal Hedge Ratio: {hedge_ratio:.3f} (HP = {hedge_ratio:.3f} * VFE)")
    
    # Calculate the historical spread
    spread = data['HP'] - (hedge_ratio * data['VFE'])
    
    # 3. Augmented Dickey-Fuller (ADF) Test
    # Tests if the spread is stationary (cointegrated)
    adf_result = adfuller(spread)
    p_value = adf_result[1]
    
    print(f"\n--- 2. ADF TEST (Cointegration) ---")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Verdict: The spread is STATIONARY. The assets are cointegrated.")
    else:
        print("Verdict: The spread is NON-STATIONARY. The assets are NOT cointegrated.")

    # 4. Hurst Exponent
    # Tests the speed/strength of the mean reversion
    hurst = calculate_hurst_exponent(spread.values)
    
    print(f"\n--- 3. HURST EXPONENT (Mean-Reversion) ---")
    print(f"Hurst Exponent: {hurst:.4f}")
    
    if hurst < 0.45:
        print("Verdict: Strong mean-reversion detected. PAIRS TRADING is optimal.")
    elif 0.45 <= hurst <= 0.55:
        print("Verdict: Spread behaves like a Random Walk. Avoid trading the spread.")
    else:
        print("Verdict: Spread is trending. Trade assets SEPARATELY using momentum.")

for day in ['prices_round_3_day_0.csv', 'prices_round_3_day_1.csv', 'prices_round_3_day_2.csv']:
    print(f"\n{'='*50}\n{day}\n{'='*50}")
    analyze_pair(day)