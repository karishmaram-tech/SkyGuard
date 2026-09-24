"""
SkyGuard — Step 1: Load and merge the three dataset files.

Datasets (Kaggle "2015 Flight Delays and Cancellations"):
  flights.csv   — one row per flight
  airlines.csv  — IATA airline code → full airline name
  airports.csv  — IATA airport code → airport name, city, state, lat/lon
"""

import pandas as pd

# ── 1. Load raw files ─────────────────────────────────────────────────────────

flights  = pd.read_csv("flights.csv",  low_memory=False)
airlines = pd.read_csv("airlines.csv")
airports = pd.read_csv("airports.csv")

# Quick sanity check — print real column names so you can spot any mismatch
print("flights  columns:", list(flights.columns))
print("airlines columns:", list(airlines.columns))
print("airports columns:", list(airports.columns))
print()

# ── 2. Merge airlines onto flights ────────────────────────────────────────────
# airlines.csv has: IATA_CODE, AIRLINE
# flights.csv  has: AIRLINE  (contains the same IATA code)
#
# After the merge we rename AIRLINE (the name column) so the two columns
# don't collide, giving us a clean AIRLINE_NAME column.

merged = flights.merge(
    airlines.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}),
    on="AIRLINE",       # join key: the IATA code shared by both tables
    how="left",         # keep every flight even if the code is missing in airlines
)

# ── 3. Merge origin-airport details onto flights ──────────────────────────────
# airports.csv has: IATA_CODE, AIRPORT, CITY, STATE, LATITUDE, LONGITUDE
# We join twice — once for the origin airport, once for the destination —
# so we know the full name/city/state for both ends of every flight.

# Rename airports columns before each join to avoid duplicate column names
origin_airports = airports.rename(columns={
    "IATA_CODE" : "ORIGIN_AIRPORT",
    "AIRPORT"   : "ORIGIN_AIRPORT_NAME",
    "CITY"      : "ORIGIN_CITY",
    "STATE"     : "ORIGIN_STATE",
    "LATITUDE"  : "ORIGIN_LATITUDE",
    "LONGITUDE" : "ORIGIN_LONGITUDE",
})

dest_airports = airports.rename(columns={
    "IATA_CODE" : "DESTINATION_AIRPORT",
    "AIRPORT"   : "DEST_AIRPORT_NAME",
    "CITY"      : "DEST_CITY",
    "STATE"     : "DEST_STATE",
    "LATITUDE"  : "DEST_LATITUDE",
    "LONGITUDE" : "DEST_LONGITUDE",
})

merged = merged.merge(origin_airports, on="ORIGIN_AIRPORT",      how="left")
merged = merged.merge(dest_airports,   on="DESTINATION_AIRPORT", how="left")

# ── 4. Inspect the result ─────────────────────────────────────────────────────

print("Merged dataframe shape (rows, columns):", merged.shape)
print()
print("First 5 rows:")
print(merged.head())
