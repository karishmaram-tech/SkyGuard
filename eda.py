"""
SkyGuard — Step 3: Exploratory Data Analysis (four charts)

Business questions answered:
  Chart 1 — How does average arrival delay vary month-to-month?
             (Spot seasonal spikes — schedule planners need this)
  Chart 2 — Which carriers keep passengers on time most reliably?
             (Benchmarks carrier operational performance)
  Chart 3 — When during the week/day is delay risk highest?
             (Helps ops teams staff buffer time and gate resources)
  Chart 4 — Which specific routes have the worst delay rates?
             (Directs targeted interventions; noise-filtered by flight count)

Inputs : `sample`  — the 800 k-row dataframe from preprocess.py
Outputs: charts/01_monthly_avg_delay.png
         charts/02_ontime_pct_by_carrier.png
         charts/03_heatmap_hour_dow.png
         charts/04_top10_routes_delay_rate.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for Colab & scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Output folder ─────────────────────────────────────────────────────────────
os.makedirs("charts", exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
ACCENT   = "#3b82d4"   # blue
WARN     = "#e05c2a"   # orange-red (highlights worst performers)
NEUTRAL  = "#6b7280"   # grey for secondary elements
BG       = "#f7f8fa"
FONTSIZE = 11

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.labelsize"   : FONTSIZE,
    "xtick.labelsize"  : FONTSIZE - 1,
    "ytick.labelsize"  : FONTSIZE - 1,
    "font.family"      : "sans-serif",
})

# Threshold used throughout: a flight is "on time" if arrival delay < 15 min
# (FAA / BTS standard definition)
ON_TIME_THRESHOLD = 15   # minutes

# ── Derived columns needed across charts ─────────────────────────────────────
# These may already exist; recompute defensively so this file is self-contained.
sample["IS_DELAYED"]  = (sample["ARRIVAL_DELAY"] >= ON_TIME_THRESHOLD).astype(int)
sample["DAY_OF_WEEK"] = sample["SCHEDULED_DATETIME"].dt.dayofweek   # 0=Mon … 6=Sun


# ═════════════════════════════════════════════════════════════════════════════
# CHART 1 · Monthly average arrival delay trend
# Business question: Does delay worsen at certain times of year, and if so when?
# ═════════════════════════════════════════════════════════════════════════════
#
# Why mean and not median?
#   Operational cost (crew overtime, gate holding, passenger compensation) scales
#   linearly with minutes of delay, so the mean — not the median — is the metric
#   that maps to actual cost. We also plot ±1 standard error to show how much
#   month-to-month variation is statistically meaningful.

monthly = (
    sample.groupby("MONTH")["ARRIVAL_DELAY"]
    .agg(mean_delay="mean", sem=lambda x: x.std() / np.sqrt(len(x)))
    .reset_index()
)
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly["month_label"] = monthly["MONTH"].apply(lambda m: MONTH_LABELS[m - 1])

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(
    monthly["MONTH"],
    monthly["mean_delay"] - monthly["sem"],
    monthly["mean_delay"] + monthly["sem"],
    color=ACCENT, alpha=0.15, label="±1 SE",
)
ax.plot(monthly["MONTH"], monthly["mean_delay"],
        color=ACCENT, linewidth=2.5, marker="o", markersize=6, label="Mean delay")

# Annotate the peak month
peak = monthly.loc[monthly["mean_delay"].idxmax()]
ax.annotate(
    f"Peak: {MONTH_LABELS[int(peak['MONTH'])-1]}\n{peak['mean_delay']:.1f} min",
    xy=(peak["MONTH"], peak["mean_delay"]),
    xytext=(peak["MONTH"] + 0.4, peak["mean_delay"] + 1.2),
    fontsize=FONTSIZE - 1, color=WARN,
    arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2),
)

ax.axhline(0, color=NEUTRAL, linewidth=0.8, linestyle="--")
ax.set_xticks(monthly["MONTH"])
ax.set_xticklabels(monthly["month_label"])
ax.set_xlabel("Month")
ax.set_ylabel("Avg arrival delay (minutes)")
ax.set_title("Monthly Average Arrival Delay — 2015", fontsize=13, fontweight="bold", pad=12)
ax.legend(frameon=False, fontsize=FONTSIZE - 1)
fig.tight_layout()
fig.savefig("charts/01_monthly_avg_delay.png", dpi=150)
plt.close(fig)
print("✓ Chart 1 saved: charts/01_monthly_avg_delay.png")


# ═════════════════════════════════════════════════════════════════════════════
# CHART 2 · On-time percentage by carrier
# Business question: Which airlines are most / least reliable for passengers?
# ═════════════════════════════════════════════════════════════════════════════
#
# Why on-time % rather than mean delay?
#   A carrier could have a low mean delay simply because most flights are
#   heavily early, masking a long tail of severe delays. On-time percentage
#   (arrivals within 15 min of schedule) is the metric published by DOT/BTS
#   and is directly interpretable by passengers and route planners.
#
# Minimum-count filter: carriers with fewer than 500 operable flights in the
# sample are excluded — too few observations to be a reliable estimate.

MIN_CARRIER_FLIGHTS = 500

carrier_stats = (
    sample.groupby(["AIRLINE", "AIRLINE_NAME"])
    .agg(
        flight_count=("IS_DELAYED", "count"),
        ontime_pct=("IS_DELAYED", lambda x: (1 - x.mean()) * 100),
    )
    .reset_index()
)
carrier_stats = carrier_stats[carrier_stats["flight_count"] >= MIN_CARRIER_FLIGHTS]
carrier_stats = carrier_stats.sort_values("ontime_pct", ascending=True)

# Colour bars: top-3 performers green, bottom-3 red, rest neutral
n = len(carrier_stats)
colors = [NEUTRAL] * n
for i in range(min(3, n)):
    colors[i]       = WARN     # worst (bottom of sorted ascending list)
for i in range(1, min(4, n + 1)):
    colors[n - i]   = "#2a9d5c"   # best

fig, ax = plt.subplots(figsize=(10, max(4, n * 0.52)))
bars = ax.barh(carrier_stats["AIRLINE_NAME"], carrier_stats["ontime_pct"],
               color=colors, height=0.65)

# Value labels inside bars
for bar, val in zip(bars, carrier_stats["ontime_pct"]):
    ax.text(max(val - 4, 1), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="right",
            fontsize=FONTSIZE - 1, color="white", fontweight="bold")

ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xlabel("On-time arrival rate  (arrival delay < 15 min)")
ax.set_title("On-Time Percentage by Carrier — 2015",
             fontsize=13, fontweight="bold", pad=12)
ax.axvline(carrier_stats["ontime_pct"].mean(), color=NEUTRAL,
           linewidth=1, linestyle="--", label=f"Network avg")
ax.legend(frameon=False, fontsize=FONTSIZE - 1)
fig.tight_layout()
fig.savefig("charts/02_ontime_pct_by_carrier.png", dpi=150)
plt.close(fig)
print("✓ Chart 2 saved: charts/02_ontime_pct_by_carrier.png")


# ═════════════════════════════════════════════════════════════════════════════
# CHART 3 · Heatmap of average delay by hour-of-day × day-of-week
# Business question: When should ops teams expect the most congestion?
# ═════════════════════════════════════════════════════════════════════════════
#
# Pivot the data into a (7 days × 24 hours) matrix of mean arrival delay.
# Cells with fewer than 30 flights are masked (shown as grey) to avoid
# noisy outliers in low-traffic slots (e.g. 3 AM on a Tuesday).

MIN_CELL_FLIGHTS = 30

heatmap_raw = (
    sample.groupby(["DAY_OF_WEEK", "DEPARTURE_HOUR"])["ARRIVAL_DELAY"]
    .agg(mean_delay="mean", count="count")
    .reset_index()
)

# Build mean-delay pivot and count pivot separately, then mask low-count cells
pivot_delay = heatmap_raw.pivot(index="DAY_OF_WEEK",
                                columns="DEPARTURE_HOUR",
                                values="mean_delay")
pivot_count = heatmap_raw.pivot(index="DAY_OF_WEEK",
                                columns="DEPARTURE_HOUR",
                                values="count")

# Reindex to guarantee all 7 days × 24 hours exist (fill missing with NaN)
pivot_delay = pivot_delay.reindex(index=range(7), columns=range(24))
pivot_count = pivot_count.reindex(index=range(7), columns=range(24))

# Mask cells below the minimum flight count
masked_delay = pivot_delay.where(pivot_count >= MIN_CELL_FLIGHTS)

DOW_LABELS  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
HOUR_LABELS = [f"{h:02d}:00" for h in range(24)]

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(masked_delay.values, aspect="auto", cmap="RdYlGn_r",
               vmin=-5, vmax=30)

# Axis labels — only every 3rd hour on x to avoid crowding
ax.set_xticks(range(24))
ax.set_xticklabels(HOUR_LABELS, rotation=45, ha="right", fontsize=8.5)
ax.set_yticks(range(7))
ax.set_yticklabels(DOW_LABELS)
ax.set_xlabel("Scheduled departure hour")
ax.set_ylabel("Day of week")
ax.set_title("Average Arrival Delay by Hour of Day and Day of Week (minutes)\n"
             f"Grey cells = fewer than {MIN_CELL_FLIGHTS} flights",
             fontsize=12, fontweight="bold", pad=10)

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Avg delay (min)", fontsize=FONTSIZE - 1)
fig.tight_layout()
fig.savefig("charts/03_heatmap_hour_dow.png", dpi=150)
plt.close(fig)
print("✓ Chart 3 saved: charts/03_heatmap_hour_dow.png")


# ═════════════════════════════════════════════════════════════════════════════
# CHART 4 · Top 10 routes by delay rate  (minimum-count filtered)
# Business question: Which specific origin→destination pairs need intervention?
# ═════════════════════════════════════════════════════════════════════════════
#
# "Delay rate" = proportion of flights on a route arriving ≥ 15 min late.
# Why rate and not mean delay?
#   A single catastrophic delay event can dominate the mean for a low-volume
#   route. Rate is more robust: it counts how *often* a passenger on that route
#   will experience a delayed arrival, which is what drives customer satisfaction.
#
# Minimum-count filter (MIN_ROUTE_FLIGHTS):
#   A route that ran only 20 times could have a 100 % delay rate by chance.
#   We require at least 200 flights in the sample (~26 flights/month on average)
#   before considering a route. This keeps every bar statistically trustworthy.

MIN_ROUTE_FLIGHTS = 200

sample["ROUTE"] = sample["ORIGIN_AIRPORT"] + " → " + sample["DESTINATION_AIRPORT"]

route_stats = (
    sample.groupby("ROUTE")
    .agg(
        flight_count=("IS_DELAYED", "count"),
        delay_rate=("IS_DELAYED", "mean"),
    )
    .reset_index()
)
route_stats = route_stats[route_stats["flight_count"] >= MIN_ROUTE_FLIGHTS]
top10 = route_stats.nlargest(10, "delay_rate").sort_values("delay_rate")

# Colour gradient: darker = higher delay rate
delay_norm = (top10["delay_rate"] - top10["delay_rate"].min()) / \
             (top10["delay_rate"].max() - top10["delay_rate"].min() + 1e-9)
bar_colors = plt.cm.OrRd(0.35 + 0.55 * delay_norm.values)   # map to orange→red band

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top10["ROUTE"], top10["delay_rate"] * 100,
               color=bar_colors, height=0.65)

# Value labels + flight count annotation
for bar, (_, row) in zip(bars, top10.iterrows()):
    ax.text(bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{row['delay_rate']*100:.1f}%  (n={int(row['flight_count']):,})",
            va="center", fontsize=FONTSIZE - 2, color=NEUTRAL)

# Network-average reference line
network_avg = sample["IS_DELAYED"].mean() * 100
ax.axvline(network_avg, color=ACCENT, linewidth=1.4, linestyle="--",
           label=f"Network avg {network_avg:.1f}%")

ax.set_xlim(0, top10["delay_rate"].max() * 100 * 1.35)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xlabel(f"Delay rate  (arrival ≥ {ON_TIME_THRESHOLD} min late)")
ax.set_title(f"Top 10 Routes by Delay Rate  (min {MIN_ROUTE_FLIGHTS} flights)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(frameon=False, fontsize=FONTSIZE - 1)
fig.tight_layout()
fig.savefig("charts/04_top10_routes_delay_rate.png", dpi=150)
plt.close(fig)
print("✓ Chart 4 saved: charts/04_top10_routes_delay_rate.png")

print("\nAll charts written to the charts/ folder.")
