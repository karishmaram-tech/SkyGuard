"""
SkyGuard — Step 2: Datetime reconstruction, cancellation handling,
                   missing-value audit, and reproducible sampling.

Picks up from the `merged` dataframe produced by load_data.py.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · Reconstruct a proper SCHEDULED_DEPARTURE datetime
# ─────────────────────────────────────────────────────────────────────────────
#
# The dataset stores date as three separate int columns (YEAR, MONTH, DAY) and
# time as a 24-hour integer like 30 → "0030" or 1425 → "14:25".
# We build a real datetime column so we can later sort by time, extract
# hour-of-day, do time-aware train/test splits, and compute rolling windows.

def build_scheduled_datetime(df):
    """
    Combine YEAR, MONTH, DAY, and SCHEDULED_DEPARTURE into a single
    pandas Timestamp column called SCHEDULED_DATETIME.

    SCHEDULED_DEPARTURE is stored as an integer 0–2359 (HHMM format).
    Midnight is represented as 0, not 2400, so no wrap-around is needed.
    """
    # Zero-pad the time integer to 4 digits: 30 → "0030", 1425 → "1425"
    time_str = df["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4)

    # Build a single string "YYYY-MM-DD HH:MM" and parse it in one vectorised
    # call — much faster than row-wise apply() on 5 million rows.
    date_str = (
        df["YEAR"].astype(str)
        + "-"
        + df["MONTH"].astype(str).str.zfill(2)
        + "-"
        + df["DAY"].astype(str).str.zfill(2)
        + " "
        + time_str.str[:2]          # hours
        + ":"
        + time_str.str[2:]          # minutes
    )

    df["SCHEDULED_DATETIME"] = pd.to_datetime(date_str, format="%Y-%m-%d %H:%M",
                                              errors="coerce")
    return df

merged = build_scheduled_datetime(merged)

print("── Datetime reconstruction ──────────────────────────────────────────")
print(f"  SCHEDULED_DATETIME dtype : {merged['SCHEDULED_DATETIME'].dtype}")
print(f"  Null datetimes (bad rows) : {merged['SCHEDULED_DATETIME'].isna().sum()}")
print(f"  Earliest flight : {merged['SCHEDULED_DATETIME'].min()}")
print(f"  Latest  flight  : {merged['SCHEDULED_DATETIME'].max()}")
print()

# Extract hour-of-day as a plain integer (0–23) — useful feature for ML later
merged["DEPARTURE_HOUR"] = merged["SCHEDULED_DATETIME"].dt.hour


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · Handle cancelled and diverted flights
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY we treat these separately, not just drop them:
#
#   CANCELLED == 1  → the flight never departed.  DEPARTURE_DELAY and
#   ARRIVAL_DELAY are NaN by definition — there is no delay to measure.
#   Mixing them into a delay-prediction model would corrupt the target
#   variable.  However, cancellations ARE operationally meaningful, so
#   we keep them in the dataframe under a clear flag instead of silently
#   discarding them. A separate model could predict cancellation risk.
#
#   DIVERTED == 1   → the flight departed but landed at an unintended
#   airport.  ARRIVAL_DELAY is NaN or misleading (the divert destination
#   is different from DESTINATION_AIRPORT).  We flag these the same way.
#
# Strategy:
#   • Add boolean flags CANCELLED and DIVERTED (already in the raw data
#     as 0/1 ints; we just make them explicit).
#   • Create a separate `operable` view that contains only flights that
#     both departed AND arrived at their intended destination — this is
#     the subset used for delay modelling.

# Ensure the columns are boolean for clarity
merged["CANCELLED"] = merged["CANCELLED"].astype(bool)
merged["DIVERTED"]  = merged["DIVERTED"].astype(bool)

# Operable flights: departed, not cancelled, not diverted
operable_mask = (~merged["CANCELLED"]) & (~merged["DIVERTED"])
operable = merged[operable_mask].copy()

print("── Cancellation / diversion breakdown ───────────────────────────────")
print(f"  Total rows in merged df  : {len(merged):>9,}")
print(f"  Cancelled flights        : {merged['CANCELLED'].sum():>9,}  "
      f"({merged['CANCELLED'].mean()*100:.2f} %)")
print(f"  Diverted  flights        : {merged['DIVERTED'].sum():>9,}  "
      f"({merged['DIVERTED'].mean()*100:.2f} %)")
print(f"  Operable  flights        : {len(operable):>9,}  "
      f"({len(operable)/len(merged)*100:.2f} %)")
print()
print("  These non-operable rows are retained in `merged` under their flags.")
print("  Delay modelling uses `operable` only, where ARRIVAL_DELAY is valid.")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 · Missing-value audit on key columns
# ─────────────────────────────────────────────────────────────────────────────
#
# We audit BOTH the full `merged` df (to catch systemic data-entry gaps) and
# the `operable` subset (to know what the ML pipeline actually needs to handle).

KEY_COLUMNS = [
    # Identifiers
    "AIRLINE", "AIRLINE_NAME",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    # Timing
    "SCHEDULED_DATETIME", "DEPARTURE_HOUR",
    "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY",
    # Target
    "ARRIVAL_DELAY",
    # Delay-cause breakdown
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
    # Route/flight info
    "DISTANCE", "AIR_TIME",
    # Status flags
    "CANCELLED", "DIVERTED",
]

def missing_value_report(df, label):
    """Print a tidy table of missing counts and percentages for KEY_COLUMNS."""
    present_cols = [c for c in KEY_COLUMNS if c in df.columns]
    missing_counts = df[present_cols].isna().sum()
    missing_pct    = (missing_counts / len(df) * 100).round(2)

    report = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_pct"  : missing_pct,
    })
    report = report[report["missing_count"] > 0].sort_values("missing_pct",
                                                              ascending=False)
    print(f"── Missing values in `{label}` (n={len(df):,}) ─────────────────")
    if report.empty:
        print("  ✓ No missing values in key columns.")
    else:
        print(report.to_string())
    print()

missing_value_report(merged,   "merged")
missing_value_report(operable, "operable")

# NOTE on delay-cause columns (AIR_SYSTEM_DELAY, WEATHER_DELAY, etc.):
# These are only populated when ARRIVAL_DELAY > 0. A NaN here means the flight
# was on time or early — it is NOT a data-quality problem. We will fill them
# with 0 during feature engineering (Step 3) when building ML features.


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · Reproducible random sample of 800,000 operable flights
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY SAMPLING DOES NOT BIAS DELAY-RATE ANALYSIS
# ───────────────────────────────────────────────
# We use simple random sampling (every row drawn with equal probability).
# Under this scheme the expected delay rate in the sample equals the true
# population delay rate, because:
#
#   E[delayed in sample / sample size]
#       = E[sum of Bernoulli(p_i)] / n
#       = (1/n) * sum(p_i)              ← same as the population mean
#
# In other words, a uniformly drawn sample is an unbiased estimator of any
# proportion or mean computed over the full dataset. With n = 800,000 out of
# ~5.3 M operable rows (~15 %), the standard error on a 20 % delay rate is:
#
#   SE = sqrt(0.20 * 0.80 / 800_000) ≈ 0.00045  (i.e. ± 0.045 percentage pts)
#
# That is negligible. Subgroup rates (per airline, per route) will be equally
# reliable as long as the subgroup is large enough — which all major carriers
# and busy routes will be.
#
# IMPORTANT: `random_state=42` makes the sample identical on every run, so
# results are fully reproducible in Colab notebooks, CI, or shared code.

SAMPLE_SIZE  = 800_000
RANDOM_STATE = 42

if len(operable) >= SAMPLE_SIZE:
    sample = operable.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
else:
    # If the operable set is already smaller than requested, use all of it.
    sample = operable.copy()
    print(f"  ⚠ Operable rows ({len(operable):,}) < {SAMPLE_SIZE:,}; "
          "using all operable rows.")

# Reset the index so iloc-based operations work cleanly downstream
sample = sample.reset_index(drop=True)

print("── Sampling ─────────────────────────────────────────────────────────")
print(f"  Operable flights available : {len(operable):>9,}")
print(f"  Sample size (random_state={RANDOM_STATE})  : {len(sample):>9,}")
print(f"  Sample shape               : {sample.shape}")
print()

# Quick sanity check: delay rate in full operable set vs sample
full_delay_rate   = (operable["ARRIVAL_DELAY"] >= 15).mean() * 100
sample_delay_rate = (sample["ARRIVAL_DELAY"]   >= 15).mean() * 100
print(f"  Delay rate (≥15 min) — full operable : {full_delay_rate:.2f} %")
print(f"  Delay rate (≥15 min) — sample        : {sample_delay_rate:.2f} %")
print("  (These should be within ~0.1 pp of each other)")
print()

print("── Ready ─────────────────────────────────────────────────────────────")
print("  `sample` is the dataframe to use for all downstream analysis.")
print(f"  Columns available: {list(sample.columns)}")
