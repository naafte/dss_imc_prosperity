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

# ── Figure 8: Mark 55 full profile ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Mark 55: Full Counterparty Profile', fontsize=14, fontweight='bold')

m55_buy  = trades[trades['buyer']  == 'Mark 55']
m55_sell = trades[trades['seller'] == 'Mark 55']

# 8a: Trade count by product (buy vs sell)
all_syms = ['HYDROGEL_PACK', 'VELVETFRUIT_EXTRACT'] + option_syms
buy_counts  = [len(m55_buy[m55_buy['symbol']==s])  for s in all_syms]
sell_counts = [len(m55_sell[m55_sell['symbol']==s]) for s in all_syms]
short_labels = ['HGP', 'VFE'] + [s.replace('VEV_','') for s in option_syms]
x = np.arange(len(all_syms))
w = 0.4
axes[0,0].bar(x - w/2, buy_counts,  w, label='Buys',  color='#2ecc71', alpha=0.8)
axes[0,0].bar(x + w/2, sell_counts, w, label='Sells', color='#e74c3c', alpha=0.8)
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(short_labels, rotation=45, ha='right')
axes[0,0].set_title('M55 Trade Count by Product')
axes[0,0].set_ylabel('# Trades')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3, axis='y')

# 8b: Net quantity by product
net_qty = [(m55_buy[m55_buy['symbol']==s]['quantity'].sum()
            - m55_sell[m55_sell['symbol']==s]['quantity'].sum()) for s in all_syms]
colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in net_qty]
axes[0,1].bar(short_labels, net_qty, color=colors, alpha=0.8)
axes[0,1].axhline(0, color='black', lw=0.8)
axes[0,1].set_xticks(range(len(short_labels)))
axes[0,1].set_xticklabels(short_labels, rotation=45, ha='right')
axes[0,1].set_title('M55 Net Position (Buy − Sell) by Product')
axes[0,1].set_ylabel('Net Quantity')
axes[0,1].grid(True, alpha=0.3, axis='y')

# 8c: M55 HGP trade price vs mid over time
m55_hgp_buy  = m55_buy[m55_buy['symbol']=='HYDROGEL_PACK'].copy()
m55_hgp_sell = m55_sell[m55_sell['symbol']=='HYDROGEL_PACK'].copy()
hgp_ds = hgp_prices[hgp_prices['timestamp'] % 10000 == 0].copy()
hgp_ds['global_ts'] = (hgp_ds['day']-1)*1000000 + hgp_ds['timestamp']
axes[0,2].plot(hgp_ds['global_ts'], hgp_ds['mid_price'], 'k-', lw=0.5, alpha=0.3, label='HGP Mid')
if len(m55_hgp_buy):
    axes[0,2].scatter(m55_hgp_buy['global_ts'], m55_hgp_buy['price'],
                      s=20, alpha=0.6, color='#2ecc71', label='M55 buys', zorder=5)
if len(m55_hgp_sell):
    axes[0,2].scatter(m55_hgp_sell['global_ts'], m55_hgp_sell['price'],
                      s=20, alpha=0.6, color='#e74c3c', label='M55 sells', zorder=5)
for d in [1, 2]:
    axes[0,2].axvline(d*1000000, color='gray', linestyle=':', alpha=0.5)
axes[0,2].set_title('M55 HGP Trades vs Mid Price')
axes[0,2].set_xlabel('Global Timestamp')
axes[0,2].set_ylabel('Price')
axes[0,2].legend(fontsize=8)
axes[0,2].grid(True, alpha=0.3)

# 8d: M55 VFE trade price vs mid over time
m55_vfe_buy  = m55_buy[m55_buy['symbol']=='VELVETFRUIT_EXTRACT'].copy()
m55_vfe_sell = m55_sell[m55_sell['symbol']=='VELVETFRUIT_EXTRACT'].copy()
vev_ds = vev_prices[vev_prices['timestamp'] % 10000 == 0].copy()
vev_ds['global_ts'] = (vev_ds['day']-1)*1000000 + vev_ds['timestamp']
axes[1,0].plot(vev_ds['global_ts'], vev_ds['mid_price'], 'k-', lw=0.5, alpha=0.3, label='VFE Mid')
if len(m55_vfe_buy):
    axes[1,0].scatter(m55_vfe_buy['global_ts'], m55_vfe_buy['price'],
                      s=20, alpha=0.6, color='#2ecc71', label='M55 buys', zorder=5)
if len(m55_vfe_sell):
    axes[1,0].scatter(m55_vfe_sell['global_ts'], m55_vfe_sell['price'],
                      s=20, alpha=0.6, color='#e74c3c', label='M55 sells', zorder=5)
for d in [1, 2]:
    axes[1,0].axvline(d*1000000, color='gray', linestyle=':', alpha=0.5)
axes[1,0].set_title('M55 VFE Trades vs Mid Price')
axes[1,0].set_xlabel('Global Timestamp')
axes[1,0].set_ylabel('Price')
axes[1,0].legend(fontsize=8)
axes[1,0].grid(True, alpha=0.3)

# 8e: M55 trade price vs mid — distribution of (trade_price - mid) for HGP and VFE
diffs_hgp, diffs_vfe = [], []
for _, row in m55_hgp_buy.iterrows():
    mp = hgp_prices[(hgp_prices['day']==row['day']) & (hgp_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp): diffs_hgp.append(row['price'] - mp.values[0])
for _, row in m55_hgp_sell.iterrows():
    mp = hgp_prices[(hgp_prices['day']==row['day']) & (hgp_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp): diffs_hgp.append(row['price'] - mp.values[0])
for _, row in m55_vfe_buy.iterrows():
    mp = vev_prices[(vev_prices['day']==row['day']) & (vev_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp): diffs_vfe.append(row['price'] - mp.values[0])
for _, row in m55_vfe_sell.iterrows():
    mp = vev_prices[(vev_prices['day']==row['day']) & (vev_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp): diffs_vfe.append(row['price'] - mp.values[0])

if diffs_hgp:
    axes[1,1].hist(diffs_hgp, bins=30, color='#3498db', alpha=0.7, label=f'HGP (n={len(diffs_hgp)})')
if diffs_vfe:
    axes[1,1].hist(diffs_vfe, bins=30, color='#e67e22', alpha=0.7, label=f'VFE (n={len(diffs_vfe)})')
axes[1,1].axvline(0, color='black', lw=1)
if diffs_hgp: axes[1,1].axvline(np.mean(diffs_hgp), color='#3498db', lw=2, linestyle='--',
                                  label=f'HGP mean={np.mean(diffs_hgp):.2f}')
if diffs_vfe: axes[1,1].axvline(np.mean(diffs_vfe), color='#e67e22', lw=2, linestyle='--',
                                  label=f'VFE mean={np.mean(diffs_vfe):.2f}')
axes[1,1].set_title('M55 Trade Price − Mid Distribution')
axes[1,1].set_xlabel('Price − Mid')
axes[1,1].set_ylabel('Count')
axes[1,1].legend(fontsize=8)
axes[1,1].grid(True, alpha=0.3)

# 8f: Who does M55 trade with? (buyer/seller counterparties)
m55_as_buyer  = trades[trades['buyer']  == 'Mark 55'][['seller','quantity']].copy()
m55_as_seller = trades[trades['seller'] == 'Mark 55'][['buyer','quantity']].copy()
m55_as_buyer.columns  = ['counterparty', 'quantity']
m55_as_seller.columns = ['counterparty', 'quantity']
cp_buy  = m55_as_buyer.groupby('counterparty')['quantity'].sum().sort_values(ascending=False).head(8)
cp_sell = m55_as_seller.groupby('counterparty')['quantity'].sum().sort_values(ascending=False).head(8)

all_cps = sorted(set(list(cp_buy.index) + list(cp_sell.index)))
x2 = np.arange(len(all_cps))
buy_v  = [cp_buy.get(c, 0)  for c in all_cps]
sell_v = [cp_sell.get(c, 0) for c in all_cps]
axes[1,2].bar(x2 - w/2, buy_v,  w, label='M55 buys from', color='#2ecc71', alpha=0.8)
axes[1,2].bar(x2 + w/2, sell_v, w, label='M55 sells to',  color='#e74c3c', alpha=0.8)
axes[1,2].set_xticks(x2)
axes[1,2].set_xticklabels([c.replace('Mark ','M') for c in all_cps], rotation=45, ha='right')
axes[1,2].set_title('M55 Counterparty Volume')
axes[1,2].set_ylabel('Total Quantity')
axes[1,2].legend(fontsize=8)
axes[1,2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT + 'fig8_mark55.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig8 done")

# ── Figure 9: M55 intraday timing — is he one-sided or clustered? ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Mark 55: Intraday Timing & One-Sidedness', fontsize=14, fontweight='bold')

# 9a: Buy vs sell timestamps within a day (are buys clustered in time?)
for product, color_b, color_s, ax in [
    ('HYDROGEL_PACK',       '#2ecc71', '#e74c3c', axes[0]),
    ('VELVETFRUIT_EXTRACT', '#3498db', '#f39c12', axes[1]),
]:
    b = m55_buy[m55_buy['symbol']==product]
    s = m55_sell[m55_sell['symbol']==product]
    mid_col = hgp_prices if product == 'HYDROGEL_PACK' else vev_prices
    ds = mid_col[mid_col['timestamp'] % 10000 == 0].copy()
    ds['global_ts'] = (ds['day']-1)*1000000 + ds['timestamp']
    ax.plot(ds['global_ts'], ds['mid_price'], 'k-', lw=0.5, alpha=0.2)
    if len(b):
        ax.scatter(b['global_ts'], b['price'], s=25, color=color_b, alpha=0.7,
                   label=f'M55 buys ({len(b)})', zorder=5)
    if len(s):
        ax.scatter(s['global_ts'], s['price'], s=25, color=color_s, alpha=0.7,
                   label=f'M55 sells ({len(s)})', zorder=5, marker='v')
    for d in [1, 2]:
        ax.axvline(d*1000000, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'M55 {product} — all fills')
    ax.set_xlabel('Global Timestamp')
    ax.set_ylabel('Price')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT + 'fig9_mark55_timing.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig9 done")

print("\nAll 9 figures saved.")

# ── Figure 10: Mark 01 spot profile (HGP + VFE) ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Mark 01: Spot Trading Profile (HGP + VFE)', fontsize=14, fontweight='bold')

m01_buy  = trades[(trades['buyer']  == 'Mark 01') & (~trades['symbol'].str.startswith('VEV_'))]
m01_sell = trades[(trades['seller'] == 'Mark 01') & (~trades['symbol'].str.startswith('VEV_'))]

# 10a: Trade count by product
spot_syms   = ['HYDROGEL_PACK', 'VELVETFRUIT_EXTRACT']
spot_labels = ['HGP', 'VFE']
bc = [len(m01_buy[m01_buy['symbol']==s])  for s in spot_syms]
sc = [len(m01_sell[m01_sell['symbol']==s]) for s in spot_syms]
x = np.arange(2)
axes[0,0].bar(x - 0.2, bc, 0.4, label='Buys',  color='#2ecc71', alpha=0.8)
axes[0,0].bar(x + 0.2, sc, 0.4, label='Sells', color='#e74c3c', alpha=0.8)
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(spot_labels)
axes[0,0].set_title('M01 Spot Trade Count'); axes[0,0].set_ylabel('# Trades')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3, axis='y')

# 10b: Net quantity by product
nq = [(m01_buy[m01_buy['symbol']==s]['quantity'].sum()
       - m01_sell[m01_sell['symbol']==s]['quantity'].sum()) for s in spot_syms]
axes[0,1].bar(spot_labels, nq, color=['#2ecc71' if v>=0 else '#e74c3c' for v in nq], alpha=0.8)
axes[0,1].axhline(0, color='black', lw=0.8)
axes[0,1].set_title('M01 Net Spot Position (Buy − Sell)'); axes[0,1].set_ylabel('Net Quantity')
axes[0,1].grid(True, alpha=0.3, axis='y')

# 10c: M01 HGP trade price vs mid over time
m01_hgp_buy  = m01_buy[m01_buy['symbol']=='HYDROGEL_PACK'].copy()
m01_hgp_sell = m01_sell[m01_sell['symbol']=='HYDROGEL_PACK'].copy()
axes[0,2].plot(hgp_ds['global_ts'], hgp_ds['mid_price'], 'k-', lw=0.5, alpha=0.3, label='HGP Mid')
if len(m01_hgp_buy):
    axes[0,2].scatter(m01_hgp_buy['global_ts'], m01_hgp_buy['price'],
                      s=20, alpha=0.6, color='#2ecc71', label=f'M01 buys ({len(m01_hgp_buy)})', zorder=5)
if len(m01_hgp_sell):
    axes[0,2].scatter(m01_hgp_sell['global_ts'], m01_hgp_sell['price'],
                      s=20, alpha=0.6, color='#e74c3c', label=f'M01 sells ({len(m01_hgp_sell)})', zorder=5)
for d in [1,2]: axes[0,2].axvline(d*1000000, color='gray', linestyle=':', alpha=0.5)
axes[0,2].set_title('M01 HGP Trades vs Mid'); axes[0,2].set_xlabel('Global Timestamp')
axes[0,2].set_ylabel('Price'); axes[0,2].legend(fontsize=8); axes[0,2].grid(True, alpha=0.3)

# 10d: M01 VFE trade price vs mid over time
m01_vfe_buy  = m01_buy[m01_buy['symbol']=='VELVETFRUIT_EXTRACT'].copy()
m01_vfe_sell = m01_sell[m01_sell['symbol']=='VELVETFRUIT_EXTRACT'].copy()
axes[1,0].plot(vev_ds['global_ts'], vev_ds['mid_price'], 'k-', lw=0.5, alpha=0.3, label='VFE Mid')
if len(m01_vfe_buy):
    axes[1,0].scatter(m01_vfe_buy['global_ts'], m01_vfe_buy['price'],
                      s=20, alpha=0.6, color='#2ecc71', label=f'M01 buys ({len(m01_vfe_buy)})', zorder=5)
if len(m01_vfe_sell):
    axes[1,0].scatter(m01_vfe_sell['global_ts'], m01_vfe_sell['price'],
                      s=20, alpha=0.6, color='#e74c3c', label=f'M01 sells ({len(m01_vfe_sell)})', zorder=5)
for d in [1,2]: axes[1,0].axvline(d*1000000, color='gray', linestyle=':', alpha=0.5)
axes[1,0].set_title('M01 VFE Trades vs Mid'); axes[1,0].set_xlabel('Global Timestamp')
axes[1,0].set_ylabel('Price'); axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.3)

# 10e: M01 trade price - mid distributions for HGP and VFE
diffs_hgp_01, diffs_vfe_01 = [], []
for _, row in m01_hgp_buy.iterrows():
    mp = hgp_prices[(hgp_prices['day']==row['day']) & (hgp_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp): diffs_hgp_01.append(row['price'] - mp.values[0])
for _, row in m01_hgp_sell.iterrows():
    mp = hgp_prices[(hgp_prices['day']==row['day']) & (hgp_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp): diffs_hgp_01.append(row['price'] - mp.values[0])
for _, row in m01_vfe_buy.iterrows():
    mp = vev_prices[(vev_prices['day']==row['day']) & (vev_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp): diffs_vfe_01.append(row['price'] - mp.values[0])
for _, row in m01_vfe_sell.iterrows():
    mp = vev_prices[(vev_prices['day']==row['day']) & (vev_prices['timestamp']==row['timestamp'])]['mid_price']
    if len(mp): diffs_vfe_01.append(row['price'] - mp.values[0])
if diffs_hgp_01:
    axes[1,1].hist(diffs_hgp_01, bins=30, color='#3498db', alpha=0.7,
                   label=f'HGP (n={len(diffs_hgp_01)}, mean={np.mean(diffs_hgp_01):.2f})')
if diffs_vfe_01:
    axes[1,1].hist(diffs_vfe_01, bins=30, color='#e67e22', alpha=0.7,
                   label=f'VFE (n={len(diffs_vfe_01)}, mean={np.mean(diffs_vfe_01):.2f})')
axes[1,1].axvline(0, color='black', lw=1)
axes[1,1].set_title('M01 Trade Price − Mid Distribution')
axes[1,1].set_xlabel('Price − Mid'); axes[1,1].set_ylabel('Count')
axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.3)

# 10f: M01 counterparty volume (spot only)
m01_as_buyer  = trades[(trades['buyer']  == 'Mark 01') & (~trades['symbol'].str.startswith('VEV_'))][['seller','quantity']].copy()
m01_as_seller = trades[(trades['seller'] == 'Mark 01') & (~trades['symbol'].str.startswith('VEV_'))][['buyer','quantity']].copy()
m01_as_buyer.columns  = ['counterparty', 'quantity']
m01_as_seller.columns = ['counterparty', 'quantity']
cp_b01 = m01_as_buyer.groupby('counterparty')['quantity'].sum().sort_values(ascending=False).head(8)
cp_s01 = m01_as_seller.groupby('counterparty')['quantity'].sum().sort_values(ascending=False).head(8)
all_cps01 = sorted(set(list(cp_b01.index) + list(cp_s01.index)))
x3 = np.arange(len(all_cps01))
axes[1,2].bar(x3 - 0.2, [cp_b01.get(c,0) for c in all_cps01], 0.4,
              label='M01 buys from', color='#2ecc71', alpha=0.8)
axes[1,2].bar(x3 + 0.2, [cp_s01.get(c,0) for c in all_cps01], 0.4,
              label='M01 sells to',  color='#e74c3c', alpha=0.8)
axes[1,2].set_xticks(x3)
axes[1,2].set_xticklabels([c.replace('Mark ','M') for c in all_cps01], rotation=45, ha='right')
axes[1,2].set_title('M01 Spot Counterparty Volume')
axes[1,2].set_ylabel('Total Quantity'); axes[1,2].legend(fontsize=8)
axes[1,2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT + 'fig10_mark01_spot.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig10 done")

# ── Figure 11: Price autocorrelation & intraday patterns ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Price Autocorrelation & Intraday Patterns', fontsize=14, fontweight='bold')

for row_i, (price_df, label, color) in enumerate([
    (hgp_prices, 'HGP', '#3498db'),
    (vev_prices, 'VFE', '#e67e22'),
]):
    # Compute per-day returns (tick-to-tick mid price changes)
    all_returns = []
    day_ts_returns = {}
    for d in [1, 2, 3]:
        sub = price_df[price_df['day']==d].sort_values('timestamp').copy()
        sub['ret'] = sub['mid_price'].diff()
        sub = sub.dropna(subset=['ret'])
        all_returns.append(sub[['timestamp','ret']])
        day_ts_returns[d] = sub

    ret_all = pd.concat(all_returns)

    # 11a/11d: Autocorrelation of returns at lags 1..20
    max_lag = 20
    lags = range(1, max_lag + 1)
    acf_vals = [ret_all['ret'].autocorr(lag=l) for l in lags]
    axes[row_i, 0].bar(lags, acf_vals, color=color, alpha=0.7)
    axes[row_i, 0].axhline(0, color='black', lw=0.8)
    axes[row_i, 0].axhline( 1.96/np.sqrt(len(ret_all)), color='red', lw=1, linestyle='--', label='95% CI')
    axes[row_i, 0].axhline(-1.96/np.sqrt(len(ret_all)), color='red', lw=1, linestyle='--')
    axes[row_i, 0].set_title(f'{label} Return Autocorrelation (lags 1–{max_lag})')
    axes[row_i, 0].set_xlabel('Lag'); axes[row_i, 0].set_ylabel('ACF')
    axes[row_i, 0].legend(fontsize=8); axes[row_i, 0].grid(True, alpha=0.3)

    # 11b/11e: Average mid price by timestamp bucket (intraday pattern)
    bucket_size = 50000
    ret_all2 = []
    for d in [1, 2, 3]:
        sub = price_df[price_df['day']==d].sort_values('timestamp').copy()
        sub['bucket'] = (sub['timestamp'] // bucket_size) * bucket_size
        ret_all2.append(sub)
    combined = pd.concat(ret_all2)
    avg_by_bucket = combined.groupby('bucket')['mid_price'].mean()
    axes[row_i, 1].plot(avg_by_bucket.index, avg_by_bucket.values, color=color, lw=2, marker='o', ms=3)
    axes[row_i, 1].set_title(f'{label} Avg Mid Price by Time-of-Day Bucket')
    axes[row_i, 1].set_xlabel('Timestamp bucket'); axes[row_i, 1].set_ylabel('Avg Mid Price')
    axes[row_i, 1].grid(True, alpha=0.3)

    # 11c/11f: Conditional next-tick return given sign of current return (momentum vs reversion)
    ret_merged = ret_all.copy()
    ret_merged['next_ret'] = ret_merged['ret'].shift(-1)
    ret_merged = ret_merged.dropna()
    up   = ret_merged[ret_merged['ret'] > 0]['next_ret']
    down = ret_merged[ret_merged['ret'] < 0]['next_ret']
    flat = ret_merged[ret_merged['ret'] == 0]['next_ret']
    axes[row_i, 2].bar(['After up', 'After flat', 'After down'],
                       [up.mean(), flat.mean(), down.mean()],
                       color=[color, 'gray', '#e74c3c'], alpha=0.8)
    axes[row_i, 2].axhline(0, color='black', lw=0.8)
    axes[row_i, 2].set_title(f'{label} Avg Next-Tick Return Conditional on Current')
    axes[row_i, 2].set_ylabel('Avg Next Return')
    axes[row_i, 2].grid(True, alpha=0.3, axis='y')
    for bar_i, (val, n) in enumerate([(up.mean(), len(up)), (flat.mean(), len(flat)), (down.mean(), len(down))]):
        axes[row_i, 2].text(bar_i, val + (0.01 if val >= 0 else -0.03),
                            f'n={n}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUT + 'fig11_autocorr_intraday.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig11 done")

print("\nAll 11 figures saved.")
