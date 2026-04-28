import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
prices = pd.concat([
    pd.read_csv(f'C:/Users/sab06/dss_imc_prosperity/ROUND4_Algo/prices_round_4_day_{d}.csv', sep=';')
    for d in [1,2,3]
])
trades_list = []
for d in [1,2,3]:
    df = pd.read_csv(f'C:/Users/sab06/dss_imc_prosperity/ROUND4_Algo/trades_round_4_day_{d}.csv', sep=';')
    df['day'] = d
    df['global_ts'] = (d-1)*1000000 + df['timestamp']
    trades_list.append(df)
trades = pd.concat(trades_list, ignore_index=True)

participants = ['Mark 01','Mark 14','Mark 22','Mark 38','Mark 49','Mark 55','Mark 67']
option_syms = sorted([s for s in trades['symbol'].unique() if s.startswith('VEV_')],
                     key=lambda x: int(x.split('_')[1]))

vev_prices = prices[prices['product']=='VELVETFRUIT_EXTRACT'].copy()
hgp_prices = prices[prices['product']=='HYDROGEL_PACK'].copy()
vev_opt_prices = prices[prices['product'].str.startswith('VEV_')].copy()

OUT = 'C:/Users/sab06/dss_imc_prosperity/ROUND4_Algo/'

# ── Figure 1: Price time series ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Round 4: Price Time Series', fontsize=14, fontweight='bold')

for d, ls in zip([1,2,3],['-','--','-.']):
    sub = vev_prices[vev_prices['day']==d]
    axes[0].plot(sub['timestamp'], sub['mid_price'], ls, lw=1, label=f'Day {d}', alpha=0.8)
axes[0].set_title('VELVETFRUIT_EXTRACT Mid Price')
axes[0].set_ylabel('Price')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for d, ls in zip([1,2,3],['-','--','-.']):
    sub = hgp_prices[hgp_prices['day']==d]
    axes[1].plot(sub['timestamp'], sub['mid_price'], ls, lw=1, label=f'Day {d}', alpha=0.8)
axes[1].set_title('HYDROGEL_PACK Mid Price')
axes[1].set_ylabel('Price')
axes[1].set_xlabel('Timestamp')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT + 'fig1_prices.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig1 done")

# ── Figure 2: Option smile / term structure ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('VEV Option Prices', fontsize=14, fontweight='bold')

strikes = [int(s.split('_')[1]) for s in option_syms]
for d, marker in zip([1,2,3],['o','s','^']):
    t0 = prices[(prices['day']==d) & (prices['timestamp']==0)]
    mids = []
    for sym in option_syms:
        row = t0[t0['product']==sym]['mid_price']
        mids.append(row.values[0] if len(row) else np.nan)
    axes[0].plot(strikes, mids, marker=marker, lw=2, label=f'Day {d} (TTE={5-d}d)')
axes[0].set_title('Option Smile (day-start mid prices)')
axes[0].set_xlabel('Strike')
axes[0].set_ylabel('Mid Price')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for sym, color in zip(['VEV_5000','VEV_5300','VEV_5400','VEV_5500'],
                      ['#e74c3c','#3498db','#2ecc71','#f39c12']):
    vals = []
    for d in [1,2,3]:
        sub = prices[(prices['day']==d) & (prices['product']==sym)]
        sub = sub[sub['timestamp'] % 5000 == 0].copy()
        sub['global_ts'] = (d-1)*1000000 + sub['timestamp']
        vals.append(sub)
    combined = pd.concat(vals)
    axes[1].plot(combined['global_ts'], combined['mid_price'], lw=1.5, label=sym, color=color)
axes[1].set_title('ATM/OTM Option Prices Over 3 Days')
axes[1].set_xlabel('Global Timestamp')
axes[1].set_ylabel('Mid Price')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
for d in [1,2]:
    axes[1].axvline(d*1000000, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(OUT + 'fig2_options.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig2 done")

# ── Figure 3: Counterparty heatmap + net positions ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Counterparty Analysis', fontsize=14, fontweight='bold')

matrix = pd.DataFrame(0, index=participants, columns=participants)
for _, row in trades.iterrows():
    b, s = row['buyer'], row['seller']
    if b in participants and s in participants:
        matrix.loc[b, s] += row['quantity']

im = axes[0].imshow(matrix.values, cmap='YlOrRd', aspect='auto')
short = [p.replace('Mark ', 'M') for p in participants]
axes[0].set_xticks(range(len(participants)))
axes[0].set_yticks(range(len(participants)))
axes[0].set_xticklabels(short, rotation=45)
axes[0].set_yticklabels(short)
axes[0].set_xlabel('Seller')
axes[0].set_ylabel('Buyer')
axes[0].set_title('Trade Volume Heatmap (row=buyer, col=seller)')
plt.colorbar(im, ax=axes[0], label='Total Quantity')
for i in range(len(participants)):
    for j in range(len(participants)):
        v = matrix.values[i,j]
        if v > 0:
            axes[0].text(j, i, str(v), ha='center', va='center', fontsize=7,
                        color='white' if v > 2000 else 'black')

cats = ['HYDROGEL_PACK', 'VELVETFRUIT_EXTRACT', 'OPTIONS']
net_data = {}
for p in participants:
    pb = trades[trades['buyer']==p]
    ps = trades[trades['seller']==p]
    hgp_net = (pb[pb['symbol']=='HYDROGEL_PACK']['quantity'].sum()
               - ps[ps['symbol']=='HYDROGEL_PACK']['quantity'].sum())
    vev_net = (pb[pb['symbol']=='VELVETFRUIT_EXTRACT']['quantity'].sum()
               - ps[ps['symbol']=='VELVETFRUIT_EXTRACT']['quantity'].sum())
    opt_net = (pb[pb['symbol'].str.startswith('VEV_')]['quantity'].sum()
               - ps[ps['symbol'].str.startswith('VEV_')]['quantity'].sum())
    net_data[p] = [hgp_net, vev_net, opt_net]

x = np.arange(len(participants))
w = 0.25
bar_colors = ['#3498db','#2ecc71','#e74c3c']
for i, label in enumerate(cats):
    vals = [net_data[p][i] for p in participants]
    axes[1].bar(x + i*w, vals, w, label=label, color=bar_colors[i], alpha=0.8)
axes[1].set_xticks(x + w)
axes[1].set_xticklabels(short)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_title('Net Position by Participant & Product')
axes[1].set_ylabel('Net Quantity (Buy - Sell)')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT + 'fig3_counterparty.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig3 done")

# ── Figure 4: Mark 01 buys below mid ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Mark 01 Option Alpha: Buying Below Mid', fontsize=14, fontweight='bold')

m01_opts = trades[(trades['buyer']=='Mark 01') & (trades['symbol'].str.startswith('VEV_'))]
diff_by_sym = {}
for sym in option_syms:
    rows = m01_opts[m01_opts['symbol']==sym]
    diffs = []
    for _, row in rows.iterrows():
        mp = vev_opt_prices[(vev_opt_prices['product']==sym) & (vev_opt_prices['timestamp']==row['timestamp'])]['mid_price']
        if len(mp) > 0:
            diffs.append(row['price'] - mp.values[0])
    if diffs:
        diff_by_sym[sym] = diffs

syms_with_data = [s for s in option_syms if s in diff_by_sym]
means = [np.mean(diff_by_sym[s]) for s in syms_with_data]
stds = [np.std(diff_by_sym[s]) for s in syms_with_data]
labels = [s.replace('VEV_', '') for s in syms_with_data]

axes[0].bar(labels, means, yerr=stds, capsize=4, color='#e74c3c', alpha=0.7)
axes[0].axhline(0, color='black', lw=1)
axes[0].set_title("Mark 01 Trade Price minus Mid Price\n(negative = buys below mid)")
axes[0].set_xlabel('Strike')
axes[0].set_ylabel('Price Difference')
axes[0].grid(True, alpha=0.3, axis='y')

focus_syms = ['VEV_5300','VEV_5400','VEV_5500']
sym_colors = ['#e74c3c','#3498db','#2ecc71']
all_vals = []
for sym, color in zip(focus_syms, sym_colors):
    rows = m01_opts[m01_opts['symbol']==sym]
    mids_p, trade_prices = [], []
    for _, row in rows.iterrows():
        mp = vev_opt_prices[(vev_opt_prices['product']==sym) & (vev_opt_prices['timestamp']==row['timestamp'])]['mid_price']
        if len(mp) > 0:
            mids_p.append(mp.values[0])
            trade_prices.append(row['price'])
            all_vals.extend([mp.values[0], row['price']])
    axes[1].scatter(mids_p, trade_prices, alpha=0.5, s=20, label=sym, color=color)

if all_vals:
    mn, mx = min(all_vals), max(all_vals)
    axes[1].plot([mn,mx],[mn,mx], 'k--', lw=1, label='Fair (y=x)')
axes[1].set_title('Mark 01 Trade Price vs Mid Price')
axes[1].set_xlabel('Mid Price at Trade Time')
axes[1].set_ylabel('Mark 01 Trade Price')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT + 'fig4_mark01_alpha.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig4 done")

# ── Figure 5: Mark 67 VEV buying ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Mark 67: One-Sided VEV Buyer', fontsize=14, fontweight='bold')

m67 = trades[trades['buyer']=='Mark 67'].copy()
mids_67, diff_67, ts_67 = [], [], []
for _, row in m67.iterrows():
    mp = vev_prices[(vev_prices['day']==row['day']) & (vev_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp) > 0:
        mids_67.append(mp.values[0])
        diff_67.append(row['price'] - mp.values[0])
        ts_67.append(row['global_ts'])

axes[0].scatter(ts_67, diff_67, alpha=0.6, s=30, color='#e67e22')
axes[0].axhline(0, color='black', lw=1)
for d in [1,2]:
    axes[0].axvline(d*1000000, color='gray', linestyle=':', alpha=0.5)
axes[0].set_title('Mark 67: Trade Price - VEV Mid Price')
axes[0].set_xlabel('Global Timestamp')
axes[0].set_ylabel('Difference vs Mid')
axes[0].grid(True, alpha=0.3)

vev_d1 = vev_prices[vev_prices['day']==1]
axes[1].plot(vev_d1['timestamp'], vev_d1['mid_price'], 'b-', lw=0.8, alpha=0.4, label='VEV Mid')
m67_d1 = m67[m67['day']==1]
for seller, color, marker in zip(['Mark 22','Mark 49'],['#e74c3c','#9b59b6'],['o','s']):
    sub = m67_d1[m67_d1['seller']==seller]
    axes[1].scatter(sub['timestamp'], sub['price'], color=color, s=40, zorder=5,
                   label=f'67 buys from {seller}', marker=marker)
axes[1].set_title('Mark 67 Buys on Day 1 vs VEV Mid')
axes[1].set_xlabel('Timestamp')
axes[1].set_ylabel('Price')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT + 'fig5_mark67.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig5 done")

# ── Figure 6: HGP spread + Mark 14/38 market making ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('HYDROGEL_PACK Market Making (Mark 14 & 38)', fontsize=14, fontweight='bold')

hgp = prices[prices['product']=='HYDROGEL_PACK'].copy()
hgp['spread'] = hgp['ask_price_1'] - hgp['bid_price_1']

for d, color in zip([1,2,3],['#e74c3c','#3498db','#2ecc71']):
    sub = hgp[hgp['day']==d]['spread'].dropna()
    axes[0].hist(sub, bins=40, alpha=0.5, label=f'Day {d}', color=color)
axes[0].set_title('HGP Bid-Ask Spread Distribution')
axes[0].set_xlabel('Spread')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

m14_hgp_buy = trades[(trades['buyer']=='Mark 14') & (trades['symbol']=='HYDROGEL_PACK')]
m38_hgp_buy = trades[(trades['buyer']=='Mark 38') & (trades['symbol']=='HYDROGEL_PACK')]
hgp_ds = hgp[hgp['timestamp'] % 10000 == 0].copy()
hgp_ds['global_ts'] = (hgp_ds['day']-1)*1000000 + hgp_ds['timestamp']

axes[1].plot(hgp_ds['global_ts'], hgp_ds['mid_price'], 'k-', lw=0.5, alpha=0.4, label='Mid')
axes[1].scatter(m14_hgp_buy['global_ts'], m14_hgp_buy['price'], s=10, alpha=0.4,
               color='#3498db', label='Mark 14 buys')
axes[1].scatter(m38_hgp_buy['global_ts'], m38_hgp_buy['price'], s=10, alpha=0.4,
               color='#f39c12', label='Mark 38 buys')
for d in [1,2]:
    axes[1].axvline(d*1000000, color='gray', linestyle=':', alpha=0.5)
axes[1].set_title('HGP: Mark 14 vs Mark 38 Buy Prices')
axes[1].set_xlabel('Global Timestamp')
axes[1].set_ylabel('Price')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT + 'fig6_hgp.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig6 done")

# ── Figure 7: Mark 22 option selling PnL estimate ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Mark 22: Option Seller Analysis', fontsize=14, fontweight='bold')

m22_sells = trades[(trades['seller']=='Mark 22') & (trades['symbol'].str.startswith('VEV_'))]
by_sym = m22_sells.groupby('symbol').agg(
    total_qty=('quantity','sum'),
    avg_price=('price','mean'),
    trade_count=('quantity','count')
).reset_index()
by_sym['strike'] = by_sym['symbol'].str.replace('VEV_','').astype(int)
by_sym = by_sym.sort_values('strike')

axes[0].bar(by_sym['strike'].astype(str), by_sym['total_qty'], color='#9b59b6', alpha=0.8)
axes[0].set_title('Mark 22 Total Options Sold by Strike')
axes[0].set_xlabel('Strike')
axes[0].set_ylabel('Total Quantity Sold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(by_sym['strike'].astype(str), by_sym['avg_price'], color='#1abc9c', alpha=0.8)
axes[1].set_title('Mark 22 Avg Price Received by Strike')
axes[1].set_xlabel('Strike')
axes[1].set_ylabel('Avg Sale Price')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT + 'fig7_mark22_sells.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig7 done")

print("\nAll 7 figures saved.")
