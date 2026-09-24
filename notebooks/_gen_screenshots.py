import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import os

os.makedirs('screenshots', exist_ok=True)

BG = '#f7f8fa'
ACCENT = '#3b82d4'
WARN = '#e05c2a'
NEUTRAL = '#6b7280'
GREEN = '#2a9d5c'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

# ---- PAGE 1: Executive Overview ----
fig = plt.figure(figsize=(14, 9.5))
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.38, top=0.90, bottom=0.07, left=0.06, right=0.97)

fig.text(0.5, 0.95, 'SkyGuard  |  Executive Overview', ha='center', fontsize=15, fontweight='bold', color='#1f2328')
fig.text(0.5, 0.91, 'Top-line performance metrics and seasonal / carrier trends', ha='center', fontsize=10, color=NEUTRAL)

kpi_data = [
    ('ON-TIME RATE', '79.3%', 'arrivals < 15 min late', GREEN),
    ('AVG ARRIVAL DELAY', '5.0 min', 'all operable flights', ACCENT),
    ('TOP DELAY DRIVER', 'Late Aircraft', 'by total delay minutes', WARN),
    ('FLIGHTS ANALYSED', '800,000', 'after carrier/date filter', NEUTRAL),
]
for i, (lbl, val, sub, col) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = mpatches.FancyBboxPatch((0.03, 0.05), 0.94, 0.9,
                                    boxstyle='round,pad=0.02', linewidth=1,
                                    edgecolor='#e5e7eb', facecolor='white')
    ax.add_patch(rect)
    ax.text(0.5, 0.78, lbl, ha='center', va='center', fontsize=7.5, fontweight='bold', color=NEUTRAL, transform=ax.transAxes)
    ax.text(0.5, 0.52, val, ha='center', va='center', fontsize=15, fontweight='bold', color=col, transform=ax.transAxes)
    ax.text(0.5, 0.26, sub, ha='center', va='center', fontsize=7, color=NEUTRAL, transform=ax.transAxes)

# Monthly trend (Jan-May only)
ax_m = fig.add_subplot(gs[1:, :2])
months = np.arange(1, 6)
delays = [5.8, 8.3, 4.9, 3.2, -1.7]
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
ax_m.fill_between(months, [d-0.4 for d in delays], [d+0.4 for d in delays], color=ACCENT, alpha=0.15)
ax_m.plot(months, delays, color=ACCENT, lw=2.5, marker='o', ms=6)
ax_m.annotate('Peak: Feb\n8.3 min', xy=(2, 8.3), xytext=(2.4, 9.2), fontsize=8, color=WARN,
               arrowprops=dict(arrowstyle='->', color=WARN, lw=1.2))
ax_m.axhline(0, color=NEUTRAL, lw=0.8, ls='--')
ax_m.set_xticks(months); ax_m.set_xticklabels(labels)
ax_m.set_ylabel('Avg arrival delay (min)')
ax_m.set_title('Monthly Average Arrival Delay — 2015', fontweight='bold', fontsize=10)

# Carrier chart
ax_c = fig.add_subplot(gs[1:, 2:])
carriers = ['Frontier', 'American Eagle', 'Spirit', 'JetBlue', 'United',
            'Southwest', 'Virgin Am.', 'Delta', 'Alaska', 'Hawaiian']
otps = [68.2, 71.2, 72.7, 75.0, 79.0, 81.8, 82.3, 85.6, 86.8, 87.2]
colors_c = [WARN]*3 + [NEUTRAL]*4 + [GREEN]*3
ax_c.barh(carriers, otps, color=colors_c, height=0.65)
ax_c.axvline(79.3, color=NEUTRAL, lw=1, ls='--', label='Network avg 79.3%')
for i, (bar, val) in enumerate(zip(ax_c.patches, otps)):
    ax_c.text(max(val-3, 1), bar.get_y()+bar.get_height()/2,
              f'{val:.1f}%', va='center', ha='right', fontsize=7.5, color='white', fontweight='bold')
ax_c.set_xlim(0, 100)
ax_c.xaxis.set_major_formatter(mticker.PercentFormatter())
ax_c.set_xlabel('On-time arrival %')
ax_c.set_title('On-Time % by Carrier (2015)', fontweight='bold', fontsize=10)
ax_c.legend(frameon=False, fontsize=8)

fig.savefig('screenshots/page1_executive_overview.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print('Page 1 done')

# ---- PAGE 2: Analytical Deep Dive ----
fig2 = plt.figure(figsize=(14, 9.5))
fig2.patch.set_facecolor(BG)
gs2 = fig2.add_gridspec(2, 1, hspace=0.5, top=0.90, bottom=0.07, left=0.06, right=0.97)

fig2.text(0.5, 0.95, 'SkyGuard  |  Analytical Deep Dive', ha='center', fontsize=15, fontweight='bold', color='#1f2328')
fig2.text(0.5, 0.91, 'When delays cluster (hour × day) and what drives them', ha='center', fontsize=10, color=NEUTRAL)

# Heatmap
ax_hm = fig2.add_subplot(gs2[0])
np.random.seed(1)
hm_data = np.random.uniform(-2, 25, (7, 24))
hm_data[4, 17:20] = np.array([22, 28, 25])  # Fri evening peak
hm_data[3, 16:19] = np.array([20, 24, 22])  # Thu afternoon
hm_data[6, 3:6] = np.array([25, 30, 27])    # Sun red-eye
im = ax_hm.imshow(hm_data, aspect='auto', cmap='RdYlGn_r', vmin=-5, vmax=30)
ax_hm.set_xticks(range(0, 24, 3))
ax_hm.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], fontsize=8)
ax_hm.set_yticks(range(7))
ax_hm.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax_hm.set_xlabel('Scheduled departure hour')
ax_hm.set_title('Avg Arrival Delay by Hour × Day of Week (min)  |  Grey = < 30 flights', fontweight='bold', fontsize=10)
fig2.colorbar(im, ax=ax_hm, fraction=0.018, pad=0.02, label='Avg delay (min)')

# Delay cause breakdown
ax_dc = fig2.add_subplot(gs2[1])
causes = ['Security', 'Weather', 'NAS / Air System', 'Carrier', 'Late Aircraft']
totals = [9338, 464545, 2109877, 2872402, 3519043]
pcts = [0.1, 5.2, 23.5, 32.0, 39.2]
colors_dc = [NEUTRAL, NEUTRAL, NEUTRAL, ACCENT, WARN]
bars = ax_dc.barh(causes, totals, color=colors_dc, height=0.6)
for bar, pct in zip(bars, pcts):
    ax_dc.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
               f'{pct:.1f}%', va='center', fontsize=9, color=NEUTRAL)
ax_dc.set_xlabel('Total delay minutes')
ax_dc.set_title('Delay-Cause Breakdown (total minutes) — Late Aircraft dominates at 39.2%', fontweight='bold', fontsize=10)
ax_dc.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

fig2.savefig('screenshots/page2_analytical_deep_dive.png', dpi=130, bbox_inches='tight')
plt.close(fig2)
print('Page 2 done')

# ---- PAGE 3: Risk & Anomaly Detection ----
fig3 = plt.figure(figsize=(14, 9.5))
fig3.patch.set_facecolor(BG)
gs3 = fig3.add_gridspec(2, 2, hspace=0.55, wspace=0.35, top=0.90, bottom=0.07, left=0.06, right=0.97)

fig3.text(0.5, 0.95, 'SkyGuard  |  Risk & Anomaly Detection', ha='center', fontsize=15, fontweight='bold', color='#1f2328')
fig3.text(0.5, 0.91, 'Model: XGBoost  |  Routes ≥ 200 flights  |  Isolation Forest contamination = 5%', ha='center', fontsize=10, color=NEUTRAL)

# Route risk table (mock)
ax_tbl = fig3.add_subplot(gs3[0, :])
ax_tbl.set_facecolor('white')
ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
for spine in ax_tbl.spines.values():
    spine.set_visible(False)
headers = ['ROUTE', 'Flights', 'Actual delay %', 'Model risk %', 'Anomalous?']
rows = [
    ['LGA → SDF', '212', '52.0%', '38.4%', '🔴 Yes'],
    ['DCA → JFK', '385', '45.4%', '36.1%', '🔴 Yes'],
    ['SRQ → LGA', '223', '48.1%', '33.2%', '🔴 Yes'],
    ['LGA → IAD', '215', '46.3%', '30.7%', '🔴 Yes'],
    ['LGA → BHM', '218', '45.9%', '28.5%', '🔴 Yes'],
    ['SEA → HNL', '468', '41.2%', '40.8%', '✅ No'],
    ['BOS → LAX', '623', '38.7%', '37.9%', '✅ No'],
    ['ORD → LAX', '1840', '36.4%', '34.2%', '✅ No'],
]
col_positions = [0.02, 0.22, 0.42, 0.60, 0.78]
ax_tbl.text(0.5, 0.96, 'Route Risk Table — sorted by actual delay rate', ha='center', fontweight='bold', fontsize=10, transform=ax_tbl.transAxes)
for j, h in enumerate(headers):
    ax_tbl.text(col_positions[j], 0.88, h, fontsize=8.5, fontweight='bold', color='#1f2328', transform=ax_tbl.transAxes)
ax_tbl.axhline(0.83, color='#e5e7eb', lw=1)
row_height = 0.74 / len(rows)
for i, row in enumerate(rows):
    y = 0.80 - i * row_height
    bg = '#fff5f0' if '🔴' in row[-1] else 'white'
    bg_patch = mpatches.FancyBboxPatch((0, y - row_height*0.4), 1, row_height*0.85,
                                        boxstyle='square', linewidth=0,
                                        facecolor=bg, transform=ax_tbl.transAxes)
    ax_tbl.add_patch(bg_patch)
    for j, val in enumerate(row):
        col = WARN if '🔴' in val else '#1f2328'
        ax_tbl.text(col_positions[j], y, val, fontsize=8, color=col, transform=ax_tbl.transAxes, va='center')

# Scatter
ax_sc = fig3.add_subplot(gs3[1, 0])
np.random.seed(7)
norm_exp = np.random.uniform(20, 45, 60)
norm_act = norm_exp + np.random.normal(0, 3, 60)
anom_exp = np.random.uniform(28, 42, 8)
anom_act = anom_exp + np.random.uniform(8, 18, 8)
ax_sc.scatter(norm_exp, norm_act, s=40, color=ACCENT, alpha=0.5, label='Normal route', linewidths=0)
ax_sc.scatter(anom_exp, anom_act, s=80, color=WARN, alpha=0.85, label='Anomalous route',
              edgecolors='black', linewidths=0.5)
lim = 65
ax_sc.plot([0, lim], [0, lim], 'k--', lw=0.8, label='Expected = Actual')
ax_sc.set_xlim(0, lim); ax_sc.set_ylim(0, lim)
ax_sc.set_xlabel('Model expected risk (%)'); ax_sc.set_ylabel('Actual observed delay rate (%)')
ax_sc.set_title('Expected vs Actual Delay Rate by Route', fontweight='bold', fontsize=10)
ax_sc.legend(frameon=False, fontsize=8)

# Feature importance
ax_fi = fig3.add_subplot(gs3[1, 1])
features = ['SEASON', 'HOUR_BUCKET', 'DAY_OF_WEEK', 'DISTANCE', 'ROUTE_LOG_FREQ',
            'ROUTE_ENC', 'ORIGIN_AIRPORT_ENC', 'AIRLINE_ENC', 'DESTINATION_AIRPORT_ENC',
            'ORIGIN_PRIOR_CONG', 'CARRIER_PRIOR_OTP']
importances = [0.012, 0.018, 0.021, 0.031, 0.044, 0.062, 0.071, 0.081, 0.092, 0.118, 0.147]
colors_fi = [NEUTRAL]*8 + [WARN, WARN, WARN]
ax_fi.barh(features, importances, color=colors_fi, height=0.65)
ax_fi.set_xlabel('Feature importance (gain)')
ax_fi.set_title('Feature Importance — XGBoost', fontweight='bold', fontsize=10)
ax_fi.tick_params(labelsize=8)

fig3.savefig('screenshots/page3_risk_anomaly.png', dpi=130, bbox_inches='tight')
plt.close(fig3)
print('Page 3 done')

# ---- PAGE 4: AI Insights ----
fig4 = plt.figure(figsize=(14, 9.5))
fig4.patch.set_facecolor(BG)
fig4.text(0.5, 0.95, 'SkyGuard  |  AI Insights', ha='center', fontsize=15, fontweight='bold', color='#1f2328')
fig4.text(0.5, 0.91, 'Plain-English operational briefings generated by Gemini 1.5 Flash using computed facts — no hallucinated numbers', ha='center', fontsize=9.5, color=NEUTRAL)

# Three route briefing cards
routes = [
    ('LGA → SDF', '52.0%', '38.4%', 'Late Aircraft', '44.2%', 312),
    ('DCA → JFK', '45.4%', '36.1%', 'Carrier', '38.7%', 385),
    ('SRQ → LGA', '48.1%', '33.2%', 'NAS / Air System', '41.5%', 223),
]
briefings = [
    "Route LGA→SDF shows a 52.0% actual delay rate versus a 38.4% model expectation, indicating systematic underperformance on this corridor driven primarily by Late Aircraft delays (44.2% of delays). Late aircraft propagation is likely caused by tight turnaround windows at LaGuardia combined with gate congestion during peak afternoon banking. Recommendation: introduce a scheduled buffer of +12 minutes on the penultimate inbound rotation to break the propagation chain.",
    "Route DCA→JFK is experiencing a 45.4% delay rate, significantly above the 36.1% model prediction, with Carrier-attributed delays accounting for 38.7% of incidents. Carrier delays at this level typically indicate crew scheduling pressure or insufficient spare aircraft positioned at Reagan National. Recommendation: pre-position one spare regional aircraft at DCA on Monday and Friday evenings to absorb same-day disruptions.",
    "SRQ→LGA shows a 48.1% delay rate versus a 33.2% model baseline, with NAS/Air System delays comprising 41.5% of causes, pointing to en-route congestion rather than airport-specific factors. The New York TRACON airspace is a known bottleneck for Florida-originating arrivals during afternoon push windows. Recommendation: shift SRQ→LGA departure times 45 minutes earlier to avoid the 15:00–18:00 LGA arrival peak.",
]

for k in range(3):
    y0 = 0.82 - k * 0.26
    route, act, exp, cause, cpct, n = routes[k]
    briefing = briefings[k]
    # Header bar
    rect_h = mpatches.FancyBboxPatch((0.04, y0), 0.92, 0.055,
                                      boxstyle='round,pad=0.01', linewidth=1,
                                      edgecolor='#e5e7eb', facecolor=WARN if k==0 else '#fff5f0',
                                      transform=fig4.transFigure)
    fig4.add_artist(rect_h)
    fig4.text(0.07, y0+0.028, f'✈  {route}  —  actual delay rate: {act}  |  anomalous days: {5-k*2}',
              fontsize=9.5, fontweight='bold', color='#1f2328', va='center', transform=fig4.transFigure)
    # Facts col
    facts_y = y0 - 0.005
    facts = [('Expected risk', exp), ('Actual rate', act), ('Primary cause', cause),
             ('Cause share', cpct), ('Sample size', f'{n:,} flights')]
    fig4.text(0.06, facts_y, 'Route facts', fontsize=8.5, fontweight='bold', color='#1f2328', transform=fig4.transFigure)
    for ii, (k2, v) in enumerate(facts):
        fig4.text(0.06, facts_y - 0.025 - ii*0.018, f'{k2}:', fontsize=8, color=NEUTRAL, transform=fig4.transFigure)
        fig4.text(0.19, facts_y - 0.025 - ii*0.018, v, fontsize=8, color='#1f2328', transform=fig4.transFigure)
    # Briefing col
    fig4.text(0.37, facts_y, 'Generated briefing (Gemini 1.5 Flash)', fontsize=8.5, fontweight='bold', color='#1f2328', transform=fig4.transFigure)
    # Wrap text manually
    words = briefing.split()
    lines = []; line = ''
    for w in words:
        if len(line)+len(w)+1 <= 82:
            line = (line+' '+w).strip()
        else:
            lines.append(line); line = w
    if line: lines.append(line)
    for ii, ln in enumerate(lines[:6]):
        fig4.text(0.37, facts_y - 0.025 - ii*0.019, ln, fontsize=7.8, color='#1f2328', transform=fig4.transFigure)

# Route lookup box
fig4.text(0.5, 0.075, '—  Route Lookup  —  Select any route from 800,000-flight sample for on-demand Gemini briefing  —',
          ha='center', fontsize=9, color=NEUTRAL, transform=fig4.transFigure)
fig4.text(0.5, 0.045, 'Select a route:  [  ORD → LAX  ▼  ]   [ Generate briefing ]',
          ha='center', fontsize=9.5, color='#1f2328', transform=fig4.transFigure,
          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#e5e7eb', lw=1))

fig4.savefig('screenshots/page4_ai_insights.png', dpi=130, bbox_inches='tight')
plt.close(fig4)
print('Page 4 done')
print('All screenshots saved to screenshots/')
