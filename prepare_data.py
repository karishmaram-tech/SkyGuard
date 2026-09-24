"""
SkyGuard — one-time data preparation script.

Run this ONCE before launching the Streamlit app:
    python prepare_data.py

It reads the raw CSVs (flights.csv, airlines.csv, airports.csv),
merges them, filters to operable flights, takes the reproducible
800 k-row sample, and saves the result to sample.parquet.

After this script finishes, the Streamlit app loads from
sample.parquet in under 1 second instead of processing 5.8 M rows
on every cold start.
"""

import gc
import pandas as pd

SAMPLE_SIZE         = 800_000
RANDOM_STATE        = 42
FAA_DELAY_THRESHOLD = 15
DELAY_CAUSE_COLS    = [
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]
OUTPUT_FILE = "sample.parquet"

# Only keep the columns the app actually uses — avoids holding the full
# ~50-column merged frame in memory across 5.8 M rows.
KEEP_COLS = [
    "YEAR", "MONTH", "DAY",
    "AIRLINE", "AIRLINE_NAME",
    "ORIGIN_AIRPORT", "ORIGIN_AIRPORT_NAME", "ORIGIN_CITY", "ORIGIN_STATE",
    "ORIGIN_LATITUDE", "ORIGIN_LONGITUDE",
    "DESTINATION_AIRPORT", "DEST_AIRPORT_NAME", "DEST_CITY", "DEST_STATE",
    "DEST_LATITUDE", "DEST_LONGITUDE",
    "SCHEDULED_DEPARTURE", "DEPARTURE_DELAY", "ARRIVAL_DELAY",
    "CANCELLED", "DIVERTED",
    "DISTANCE", "AIR_TIME",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]

if __name__ == "__main__":
    print("Reading flights.csv …  (this is the slow part — runs once only)")
    flights  = pd.read_csv("flights.csv",  low_memory=False)
    airlines = pd.read_csv("airlines.csv")
    airports = pd.read_csv("airports.csv")
    print(f"  flights rows : {len(flights):,}")

    # ── Merge airlines ────────────────────────────────────────────────────────
    df = flights.merge(
        airlines.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}),
        on="AIRLINE", how="left",
    )
    del flights, airlines
    gc.collect()

    # ── Merge airports (origin + destination) ─────────────────────────────────
    orig = airports.rename(columns={
        "IATA_CODE": "ORIGIN_AIRPORT",  "AIRPORT": "ORIGIN_AIRPORT_NAME",
        "CITY": "ORIGIN_CITY",          "STATE": "ORIGIN_STATE",
        "LATITUDE": "ORIGIN_LATITUDE",  "LONGITUDE": "ORIGIN_LONGITUDE",
    })
    dest = airports.rename(columns={
        "IATA_CODE": "DESTINATION_AIRPORT", "AIRPORT": "DEST_AIRPORT_NAME",
        "CITY": "DEST_CITY",                "STATE": "DEST_STATE",
        "LATITUDE": "DEST_LATITUDE",        "LONGITUDE": "DEST_LONGITUDE",
    })
    df = df.merge(orig, on="ORIGIN_AIRPORT",      how="left")
    df = df.merge(dest, on="DESTINATION_AIRPORT", how="left")
    del airports, orig, dest
    gc.collect()

    # ── Drop unused columns before filtering to save memory ───────────────────
    present_keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[present_keep]
    gc.collect()

    # ── Datetime reconstruction ───────────────────────────────────────────────
    time_str = df["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4)
    date_str = (
        df["YEAR"].astype(str) + "-" +
        df["MONTH"].astype(str).str.zfill(2) + "-" +
        df["DAY"].astype(str).str.zfill(2) + " " +
        time_str.str[:2] + ":" + time_str.str[2:]
    )
    df["SCHEDULED_DATETIME"] = pd.to_datetime(date_str, format="%Y-%m-%d %H:%M", errors="coerce")
    df["DEPARTURE_HOUR"]     = df["SCHEDULED_DATETIME"].dt.hour
    df["DATE"]               = df["SCHEDULED_DATETIME"].dt.date
    del time_str, date_str
    gc.collect()

    # ── Filter to operable flights and sample ─────────────────────────────────
    df["CANCELLED"] = df["CANCELLED"].astype(bool)
    df["DIVERTED"]  = df["DIVERTED"].astype(bool)

    operable_idx = df.index[(~df["CANCELLED"]) & (~df["DIVERTED"])]
    print(f"  operable rows: {len(operable_idx):,}")

    n          = min(SAMPLE_SIZE, len(operable_idx))
    sample_idx = operable_idx.to_series().sample(n=n, random_state=RANDOM_STATE)
    sample     = df.loc[sample_idx].reset_index(drop=True)
    del df
    gc.collect()

    # ── Target + basic features ───────────────────────────────────────────────
    sample["Delayed"]     = (sample["ARRIVAL_DELAY"] >= FAA_DELAY_THRESHOLD).astype(int)
    sample["DAY_OF_WEEK"] = sample["SCHEDULED_DATETIME"].dt.dayofweek
    sample["ROUTE"]       = sample["ORIGIN_AIRPORT"] + " → " + sample["DESTINATION_AIRPORT"]
    sample["DISTANCE"]    = sample["DISTANCE"].fillna(sample["DISTANCE"].median())
    for col in DELAY_CAUSE_COLS:
        if col in sample.columns:
            sample[col] = sample[col].fillna(0)

    # ── Save ──────────────────────────────────────────────────────────────────
    sample.to_parquet(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved {len(sample):,} rows → {OUTPUT_FILE}")
    print("  You can now run:  python train_models.py")
