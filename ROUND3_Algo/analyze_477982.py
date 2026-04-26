import pandas as pd
import json

# Load JSON
with open('477982.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Convert to dataframe
df = pd.json_normalize(data)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Parse the activitiesLog
# We use the same robust cleaning as before to avoid KeyErrors
log_string = df.iloc[0]['activitiesLog']
lines = log_string.strip().split('\n')
headers = [h.strip().lower() for h in lines[0].split(';')]
data = [line.split(';') for line in lines[1:]]
log_df = pd.DataFrame(data, columns=headers)

# Convert numeric columns
for col in ['timestamp', 'mid_price', 'profit_and_loss']:
    log_df[col] = pd.to_numeric(log_df[col], errors='coerce')

# 2. Derive Estimated Position (Delta_PnL / Delta_Price)
log_df['delta_pnl'] = log_df['profit_and_loss'].diff()
log_df['delta_price'] = log_df['mid_price'].diff()
log_df['est_pos'] = np.where(log_df['delta_price'] != 0, log_df['delta_pnl'] / log_df['delta_price'], np.nan)
log_df['est_pos'] = log_df['est_pos'].ffill().fillna(0).round().clip(-35, 35)

# 3. Plotting - Clean Two-Panel Line Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top Plot: Total PnL
ax1.plot(log_df['timestamp'], log_df['profit_and_loss'], color='forestgreen', linewidth=2, label='Total PnL')
ax1.set_ylabel('Profit and Loss')
ax1.set_title('Hydrogel Pack: PnL and Position Analysis', fontsize=14)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

# Bottom Plot: Estimated Position
ax2.plot(log_df['timestamp'], log_df['est_pos'], color='royalblue', linewidth=1.5, label='Estimated Position')
ax2.axhline(y=30, color='red', linestyle='--', linewidth=1, label='Limit (30)')
ax2.axhline(y=-30, color='red', linestyle='--', linewidth=1)
ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
ax2.set_ylabel('Position Quantity')
ax2.set_xlabel('Timestamp')
ax2.set_ylim(-40, 40)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()