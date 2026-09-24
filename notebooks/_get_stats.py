import pandas as pd
import numpy as np

airlines = pd.read_csv('airlines.csv')

# We already know from the full-file read:
# Total rows: 5,819,079 | Cancelled: 89,884 (1.54%) | Diverted: 15,187 (0.26%)
# Unique airlines: 14 | Unique origin airports: 934
# So operable = 5,819,079 - 89,884 - 15,187 = 5,714,008

# Read a manageable subset for stats
print("Reading sample chunk for detailed stats...")
# Read first 2M rows for fast stats
reader = pd.read_csv('flights.csv', low_memory=False, chunksize=200000)
chunks = []
for i, chunk in enumerate(reader):
    chunks.append(chunk)
    if i >= 9:  # first 2M rows
        break
raw = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(raw):,} rows for stats")

operable = raw[(raw['CANCELLED']==0) & (raw['DIVERTED']==0)].copy()
print(f"Operable in chunk: {len(operable):,}")

# Sample 800k from what we have (seed 42)
np.random.seed(42)
sample = operable.sample(n=min(800000, len(operable)), random_state=42).reset_index(drop=True)
print(f"Sample size used: {len(sample):,}")
sample_delay_rate = (sample['ARRIVAL_DELAY'] >= 15).mean()*100
print(f'Sample delay rate: {sample_delay_rate:.2f}%')
print(f'Avg arrival delay: {sample["ARRIVAL_DELAY"].mean():.2f} min')
print()

# Merge airlines
merged_s = sample.merge(
    airlines.rename(columns={'IATA_CODE':'AIRLINE','AIRLINE':'AIRLINE_NAME'}),
    on='AIRLINE', how='left'
)
merged_s['Delayed'] = (merged_s['ARRIVAL_DELAY'] >= 15).astype(int)

# Monthly stats
monthly = merged_s.groupby('MONTH')['ARRIVAL_DELAY'].mean()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print('=== MONTHLY AVG DELAY ===')
for m, v in monthly.items():
    print(f'  {month_labels[m-1]}: {v:.1f} min')
best_month = monthly.idxmin()
worst_month = monthly.idxmax()
print(f'  Best: {month_labels[best_month-1]} ({monthly[best_month]:.1f} min)')
print(f'  Worst: {month_labels[worst_month-1]} ({monthly[worst_month]:.1f} min)')
print()

# Carrier stats
carrier_stats = (merged_s.groupby(['AIRLINE','AIRLINE_NAME'])
    .agg(cnt=('Delayed','count'), otp=('Delayed', lambda x: (1-x.mean())*100))
    .reset_index()
)
carrier_stats = carrier_stats[carrier_stats['cnt'] >= 500].sort_values('otp')
print('=== CARRIER ON-TIME % (sorted worst to best) ===')
for _, row in carrier_stats.iterrows():
    print(f'  {row["AIRLINE_NAME"]}: {row["otp"]:.1f}%  (n={int(row["cnt"]):,})')
print(f'  Network avg: {carrier_stats["otp"].mean():.1f}%')
print()

# Top routes
merged_s['ROUTE'] = merged_s['ORIGIN_AIRPORT'] + ' -> ' + merged_s['DESTINATION_AIRPORT']
route_stats = (merged_s.groupby('ROUTE')
    .agg(cnt=('Delayed','count'), dr=('Delayed','mean'))
    .reset_index()
)
route_stats = route_stats[route_stats['cnt'] >= 50]
top10 = route_stats.nlargest(10, 'dr')
print('=== TOP 10 WORST ROUTES BY DELAY RATE ===')
for _, row in top10.iterrows():
    print(f'  {row["ROUTE"]}: {row["dr"]*100:.1f}%  (n={int(row["cnt"]):,})')
print()

# Delay cause breakdown
delay_cols = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
delay_labels = {
    'AIR_SYSTEM_DELAY': 'NAS / Air System',
    'SECURITY_DELAY': 'Security',
    'AIRLINE_DELAY': 'Carrier',
    'LATE_AIRCRAFT_DELAY': 'Late Aircraft',
    'WEATHER_DELAY': 'Weather',
}
present_delay_cols = [c for c in delay_cols if c in merged_s.columns]
delayed_only = merged_s[merged_s['Delayed']==1]
cause_totals = delayed_only[present_delay_cols].sum()
cause_pct = cause_totals / cause_totals.sum() * 100
print('=== DELAY CAUSE BREAKDOWN (total minutes, sample) ===')
for col in present_delay_cols:
    print(f'  {delay_labels[col]}: {cause_totals[col]:,.0f} min ({cause_pct[col]:.1f}%)')
print(f'  Top cause: {delay_labels[cause_totals.idxmax()]}')
print()

# Missing values
print('=== MISSING VALUES IN KEY COLS (operable chunk) ===')
key_cols = ['ARRIVAL_DELAY','DEPARTURE_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY',
            'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY','DISTANCE']
for col in key_cols:
    if col in operable.columns:
        n_miss = operable[col].isna().sum()
        pct = n_miss / len(operable) * 100
        if n_miss > 0:
            print(f'  {col}: {n_miss:,} ({pct:.1f}%)')
print()

# Heatmap peak
time_str = merged_s['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4)
merged_s['DEPARTURE_HOUR'] = time_str.str[:2].astype(int)
hm = merged_s.groupby(['DAY_OF_WEEK','DEPARTURE_HOUR'])['ARRIVAL_DELAY'].mean()
peak_idx = hm.idxmax()
dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
print(f'=== HEATMAP PEAK ===')
print(f'  Highest avg delay: {dow_labels[peak_idx[0]]} {peak_idx[1]:02d}:00 = {hm[peak_idx]:.1f} min')
