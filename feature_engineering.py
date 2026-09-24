"""
SkyGuard — Step 4: Feature Engineering

TARGET
  Delayed = 1 if ARRIVAL_DELAY >= 15 min (FAA / BTS official definition), else 0.

FEATURES BUILT — only pre-departure information is used:
  1. HOUR_BUCKET        — scheduled hour grouped into 4 operational time-of-day bands
  2. DAY_OF_WEEK        — 0 (Monday) … 6 (Sunday)
  3. MONTH              — 1-12 (kept numeric; ordinal season encoding added separately)
  4. SEASON             — Winter / Spring / Summer / Fall (integer 0-3)
  5. ROUTE              — "ORIGIN → DESTINATION" string identifier
  6. CARRIER_PRIOR_OTP  — carrier's on-time rate computed ONLY from flights before
                          the current flight's date (expanding-window approach)
  7. ORIGIN_PRIOR_CONG  — origin airport's prior congestion rate (same method)

DATA LEAKAGE POLICY — enforced throughout this file:
  • ARRIVAL_DELAY and DEPARTURE_DELAY are the target and a post-departure
    observation respectively — neither is used as a feature.
  • The five delay-cause breakdown columns (WEATHER_DELAY, AIRLINE_DELAY, etc.)
    are only populated after the flight lands — excluded entirely.
  • Historical rates (features 6 & 7) are computed with an EXCLUSIVE look-back:
    for a flight on date D, only flights on dates < D contribute to the estimate.
    This is a strict expanding window with no same-day contamination.
  • DISTANCE and AIR_TIME: DISTANCE is the scheduled great-circle distance —
    known before departure. AIR_TIME is block time actually flown — not known
    before departure and therefore excluded.

Input : `sample`  — 800 k-row dataframe from preprocess.py
Output: `features_df`  — ready-to-model dataframe (features + target only)
        `FEATURE_COLS` — list of column names to pass to the classifier
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · Target variable
# ─────────────────────────────────────────────────────────────────────────────
# FAA / BTS official definition: a flight is delayed when it arrives 15 or more
# minutes after its published scheduled arrival time.

FAA_DELAY_THRESHOLD = 15  # minutes

sample["Delayed"] = (sample["ARRIVAL_DELAY"] >= FAA_DELAY_THRESHOLD).astype(int)

print("── Target distribution ──────────────────────────────────────────────")
vc = sample["Delayed"].value_counts(normalize=True) * 100
print(f"  On-time  (0): {vc.get(0, 0):.1f} %")
print(f"  Delayed  (1): {vc.get(1, 0):.1f} %")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · Static / schedule-derived features
# ─────────────────────────────────────────────────────────────────────────────
# All of these are knowable the moment a ticket is issued.

# ── 2a. Hour bucket ───────────────────────────────────────────────────────────
# Grouping hours into 4 bands captures the operational reality:
#   Early morning  (05-08): low congestion, clean slate
#   Morning peak   (09-12): banking of short-haul connections
#   Afternoon      (13-17): busiest period, maximum propagation risk
#   Evening        (18-23): delays compound from earlier rotations
# Hours 00-04 ("red-eye") are their own band.
#
# We encode as an integer 0-4 so tree models can split on it directly.

def hour_to_bucket(h):
    if   0  <= h <= 4:  return 0   # red-eye
    elif 5  <= h <= 8:  return 1   # early morning
    elif 9  <= h <= 12: return 2   # morning peak
    elif 13 <= h <= 17: return 3   # afternoon
    else:               return 4   # evening

sample["HOUR_BUCKET"] = sample["DEPARTURE_HOUR"].apply(hour_to_bucket)

# ── 2b. Day of week ───────────────────────────────────────────────────────────
# Already computed in eda.py; recreate defensively so this file is standalone.
sample["DAY_OF_WEEK"] = sample["SCHEDULED_DATETIME"].dt.dayofweek  # 0=Mon, 6=Sun

# ── 2c. Month ─────────────────────────────────────────────────────────────────
# Already in the dataframe as integers 1-12.  No transformation needed;
# tree models handle ordinal integers natively.

# ── 2d. Season ────────────────────────────────────────────────────────────────
# Season captures the non-linear jump in delay risk that isn't smooth across
# months (e.g. June and December are both high-delay but far apart numerically).
# Winter: Dec-Feb  Spring: Mar-May  Summer: Jun-Aug  Fall: Sep-Nov
MONTH_TO_SEASON = {
    12: 0, 1: 0, 2: 0,   # Winter
     3: 1, 4: 1, 5: 1,   # Spring
     6: 2, 7: 2, 8: 2,   # Summer
     9: 3,10: 3,11: 3,   # Fall
}
sample["SEASON"] = sample["MONTH"].map(MONTH_TO_SEASON)

# ── 2e. Route identifier ──────────────────────────────────────────────────────
# "ORD → LAX" is a unique operational unit: same aircraft type, crew base,
# gate congestion pattern, and weather exposure pairing.
sample["ROUTE"] = sample["ORIGIN_AIRPORT"] + " → " + sample["DESTINATION_AIRPORT"]

# ── 2f. Distance (scheduled) ─────────────────────────────────────────────────
# Great-circle distance between origin and destination — published in the
# schedule, available before departure. Longer flights have more airtime to
# recover from gate holds; shorter hops are more sensitive to any ground delay.
# No transformation needed here; the tree model will handle the scale.
# Fill the rare NaN (unmatched airport code) with the median.
sample["DISTANCE"] = sample["DISTANCE"].fillna(sample["DISTANCE"].median())

print("── Static features created ───────────────────────────────────────────")
print(f"  HOUR_BUCKET  : {sorted(sample['HOUR_BUCKET'].unique())}")
print(f"  DAY_OF_WEEK  : 0 (Mon) … 6 (Sun)")
print(f"  SEASON       : {sorted(sample['SEASON'].unique())}  (0=Win 1=Spr 2=Sum 3=Fall)")
print(f"  Unique routes: {sample['ROUTE'].nunique():,}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 · Historical carrier on-time rate  (leak-free expanding window)
# ─────────────────────────────────────────────────────────────────────────────
#
# LEAKAGE EXPLANATION
# ───────────────────
# A naïve approach would compute each carrier's overall delay rate and attach
# it to every row — but that uses future outcomes to describe past flights.
# For example, if AA has a bad December, that December data would inflate AA's
# "historical" rate even when predicting a January flight.
#
# CORRECT APPROACH — strict date-based expanding window:
#   1. Sort all flights by DATE (calendar date, not datetime — daily granularity
#      is sufficient and avoids same-flight contamination).
#   2. For each flight on date D, the carrier rate is computed from ALL flights
#      by that carrier on dates strictly BEFORE D.
#   3. This is implemented with a groupby + expanding mean on daily aggregates,
#      then merged back by (AIRLINE, DATE).  This is O(n) and avoids any
#      per-row Python loop.
#
# COLD-START: the very first date a carrier appears has no prior history.
# We fill those NaNs with the overall network delay rate — a conservative
# neutral prior rather than a misleading 0 or 1.

NETWORK_DELAY_RATE = sample["Delayed"].mean()   # computed from sample only

# Step 1: create a DATE column (date only, strips time-of-day)
sample["DATE"] = sample["SCHEDULED_DATETIME"].dt.date

# Step 2: compute daily per-carrier delay rate from the sample
carrier_daily = (
    sample.groupby(["AIRLINE", "DATE"])["Delayed"]
    .mean()
    .reset_index()
    .rename(columns={"Delayed": "daily_carrier_rate"})
    .sort_values(["AIRLINE", "DATE"])
)

# Step 3: for each carrier, compute cumulative (expanding) mean of past daily
# rates — using shift(1) ensures today's rate is NOT included.
carrier_daily["CARRIER_PRIOR_OTP"] = (
    carrier_daily
    .groupby("AIRLINE")["daily_carrier_rate"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

# Step 4: fill cold-start NaNs with the network average
carrier_daily["CARRIER_PRIOR_OTP"] = carrier_daily["CARRIER_PRIOR_OTP"].fillna(
    NETWORK_DELAY_RATE
)

# Step 5: merge back onto the sample (left join — every flight gets a value)
sample = sample.merge(
    carrier_daily[["AIRLINE", "DATE", "CARRIER_PRIOR_OTP"]],
    on=["AIRLINE", "DATE"],
    how="left",
)
sample["CARRIER_PRIOR_OTP"] = sample["CARRIER_PRIOR_OTP"].fillna(NETWORK_DELAY_RATE)

print("── Carrier prior on-time rate (CARRIER_PRIOR_OTP) ───────────────────")
print(sample[["AIRLINE", "DATE", "CARRIER_PRIOR_OTP"]].head(6).to_string(index=False))
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · Historical origin-airport congestion rate  (same expanding method)
# ─────────────────────────────────────────────────────────────────────────────
#
# "Congestion" here means: what fraction of departures from this airport were
# delayed on prior days?  A busy hub on a rainy week will propagate delays to
# all downstream rotations; knowing its prior congestion rate captures this
# systemic pressure without using any same-day or post-departure information.
#
# Same shift(1)-expanding pattern as Section 3 — same leakage guarantee.

airport_daily = (
    sample.groupby(["ORIGIN_AIRPORT", "DATE"])["Delayed"]
    .mean()
    .reset_index()
    .rename(columns={"Delayed": "daily_airport_rate"})
    .sort_values(["ORIGIN_AIRPORT", "DATE"])
)

airport_daily["ORIGIN_PRIOR_CONG"] = (
    airport_daily
    .groupby("ORIGIN_AIRPORT")["daily_airport_rate"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

airport_daily["ORIGIN_PRIOR_CONG"] = airport_daily["ORIGIN_PRIOR_CONG"].fillna(
    NETWORK_DELAY_RATE
)

sample = sample.merge(
    airport_daily[["ORIGIN_AIRPORT", "DATE", "ORIGIN_PRIOR_CONG"]],
    on=["ORIGIN_AIRPORT", "DATE"],
    how="left",
)
sample["ORIGIN_PRIOR_CONG"] = sample["ORIGIN_PRIOR_CONG"].fillna(NETWORK_DELAY_RATE)

print("── Origin airport prior congestion rate (ORIGIN_PRIOR_CONG) ────────")
print(sample[["ORIGIN_AIRPORT", "DATE", "ORIGIN_PRIOR_CONG"]].head(6).to_string(index=False))
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 · Route frequency (log-scaled)
# ─────────────────────────────────────────────────────────────────────────────
# High-frequency routes (e.g. BOS→LGA shuttle) have tighter turnaround windows
# and more propagation risk; very thin routes are often operated by smaller
# regional aircraft with different delay dynamics.
# We use the log of the route's total flight count in the sample — this
# compresses the long tail and gives tree models a smooth signal.
# NOTE: this is computed from the full sample (not just prior dates) because
# route frequency is a stable structural characteristic of the route network,
# not an outcome variable.  It does not reveal delay outcomes.

route_counts = sample["ROUTE"].value_counts().rename("route_total_flights")
sample = sample.join(route_counts, on="ROUTE")
sample["ROUTE_LOG_FREQ"] = np.log1p(sample["route_total_flights"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 · Encode categorical columns for ML
# ─────────────────────────────────────────────────────────────────────────────
# Tree-based models (XGBoost, Random Forest) can handle integer-encoded
# categoricals directly — no one-hot encoding needed for high-cardinality
# columns like ROUTE (thousands of unique values).  We use pandas Categorical
# codes, which produce stable integer mappings.

for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"]:
    sample[col + "_ENC"] = pd.Categorical(sample[col]).codes


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 · Assemble the final feature matrix
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Time features
    "HOUR_BUCKET",          # 0-4 operational band
    "DAY_OF_WEEK",          # 0 Mon … 6 Sun
    "MONTH",                # 1-12
    "SEASON",               # 0 Win … 3 Fall
    # Route & carrier (encoded)
    "AIRLINE_ENC",          # integer-encoded IATA carrier code
    "ORIGIN_AIRPORT_ENC",   # integer-encoded origin
    "DESTINATION_AIRPORT_ENC",
    "ROUTE_ENC",            # integer-encoded origin→destination pair
    # Structural
    "DISTANCE",             # scheduled great-circle distance (miles)
    "ROUTE_LOG_FREQ",       # log(route flight count) — structural network signal
    # Historical rates (leak-free)
    "CARRIER_PRIOR_OTP",    # carrier's prior on-time rate (expanding window)
    "ORIGIN_PRIOR_CONG",    # origin airport's prior congestion rate
]

TARGET_COL = "Delayed"

# Verify no feature has nulls before handing off to the model
null_check = sample[FEATURE_COLS + [TARGET_COL]].isna().sum()
null_check = null_check[null_check > 0]
if null_check.empty:
    print("✓ Zero nulls in feature matrix — ready for modelling.")
else:
    print("⚠ Null values remain in features (investigate before modelling):")
    print(null_check.to_string())

features_df = sample[FEATURE_COLS + [TARGET_COL]].copy()

print()
print("── Feature matrix summary ───────────────────────────────────────────")
print(f"  Shape  : {features_df.shape}")
print(f"  Target : '{TARGET_COL}'  (0 = on-time, 1 = delayed ≥ {FAA_DELAY_THRESHOLD} min)")
print(f"  Features ({len(FEATURE_COLS)}):")
for col in FEATURE_COLS:
    print(f"    {col}")
print()
print("`features_df` and `FEATURE_COLS` are ready for the classifier (Step 5).")
