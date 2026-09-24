"""
SkyGuard — one-time model training script.

Run this ONCE after prepare_data.py:
    python prepare_data.py   # (already done if sample.parquet exists)
    python train_models.py

Reads sample.parquet, runs feature engineering, trains all three models,
picks the best one, computes route risk and anomaly tables, then saves:

    models/sample_with_probs.parquet  — 800k sample with delay_prob column
    models/best_name.txt              — name of the winning model
    models/feat_imp.parquet           — feature importance dataframe
    models/route_risk.parquet         — per-route risk table
    models/anomaly_routes.parquet     — anomalous routes table

After this the Streamlit app loads all five artifacts instantly (~1 s total)
instead of retraining on every cold start.
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import roc_auc_score
from xgboost               import XGBClassifier

RANDOM_STATE = 42
OUT_DIR      = "models"

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load sample ───────────────────────────────────────────────────────────
    print("Loading sample.parquet …")
    sample = pd.read_parquet("sample.parquet")
    sample["SCHEDULED_DATETIME"] = pd.to_datetime(sample["SCHEDULED_DATETIME"], errors="coerce")
    sample["DATE"]               = pd.to_datetime(sample["DATE"], errors="coerce").dt.date
    print(f"  rows: {len(sample):,}")

    # ── Expanding-window carrier on-time rate ─────────────────────────────────
    print("Engineering features …")
    NETWORK_RATE = sample["Delayed"].mean()

    carrier_daily = (
        sample.groupby(["AIRLINE", "DATE"])["Delayed"]
        .mean().reset_index().rename(columns={"Delayed": "dcr"})
        .sort_values(["AIRLINE", "DATE"])
    )
    carrier_daily["CARRIER_PRIOR_OTP"] = (
        carrier_daily.groupby("AIRLINE")["dcr"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(NETWORK_RATE)
    )
    sample = sample.merge(carrier_daily[["AIRLINE", "DATE", "CARRIER_PRIOR_OTP"]],
                          on=["AIRLINE", "DATE"], how="left")
    sample["CARRIER_PRIOR_OTP"] = sample["CARRIER_PRIOR_OTP"].fillna(NETWORK_RATE)

    # ── Expanding-window origin congestion rate ───────────────────────────────
    airport_daily = (
        sample.groupby(["ORIGIN_AIRPORT", "DATE"])["Delayed"]
        .mean().reset_index().rename(columns={"Delayed": "dar"})
        .sort_values(["ORIGIN_AIRPORT", "DATE"])
    )
    airport_daily["ORIGIN_PRIOR_CONG"] = (
        airport_daily.groupby("ORIGIN_AIRPORT")["dar"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(NETWORK_RATE)
    )
    sample = sample.merge(airport_daily[["ORIGIN_AIRPORT", "DATE", "ORIGIN_PRIOR_CONG"]],
                          on=["ORIGIN_AIRPORT", "DATE"], how="left")
    sample["ORIGIN_PRIOR_CONG"] = sample["ORIGIN_PRIOR_CONG"].fillna(NETWORK_RATE)

    # ── Hour bucket, season, route freq, categoricals ─────────────────────────
    def hour_bucket(h):
        if   h <= 4:  return 0
        elif h <= 8:  return 1
        elif h <= 12: return 2
        elif h <= 17: return 3
        else:         return 4

    sample["HOUR_BUCKET"]    = sample["DEPARTURE_HOUR"].apply(hour_bucket)
    sample["SEASON"]         = sample["MONTH"].map(
        {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
    )
    route_counts             = sample["ROUTE"].value_counts()
    sample["ROUTE_LOG_FREQ"] = np.log1p(sample["ROUTE"].map(route_counts))

    for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"]:
        sample[col + "_ENC"] = pd.Categorical(sample[col]).codes

    MODEL_FEATURES = [
        "HOUR_BUCKET", "DAY_OF_WEEK", "SEASON",
        "AIRLINE_ENC", "ORIGIN_AIRPORT_ENC", "DESTINATION_AIRPORT_ENC", "ROUTE_ENC",
        "DISTANCE", "ROUTE_LOG_FREQ",
        "CARRIER_PRIOR_OTP", "ORIGIN_PRIOR_CONG",
    ]

    # ── Time-based split ──────────────────────────────────────────────────────
    X_train = sample.loc[sample["MONTH"] <= 10, MODEL_FEATURES]
    X_test  = sample.loc[sample["MONTH"] >= 11, MODEL_FEATURES]
    y_train = sample.loc[sample["MONTH"] <= 10, "Delayed"]
    y_test  = sample.loc[sample["MONTH"] >= 11, "Delayed"]
    print(f"  train rows: {len(X_train):,}  |  test rows: {len(X_test):,}")

    # ── Train three models ────────────────────────────────────────────────────
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())

    models = [
        ("Logistic Regression", Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ])),
        ("Random Forest", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
        )),
        ("XGBoost", XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg / pos,
            eval_metric="auc", n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
        )),
    ]

    results = {}
    for name, mdl in models:
        print(f"  Training {name} …")
        mdl.fit(X_train, y_train)
        proba = mdl.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        results[name] = (mdl, auc)
        print(f"    AUC = {auc:.4f}")

    best_name  = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"\n  Best model: {best_name}  (AUC={results[best_name][1]:.4f})")

    # ── Feature importances ───────────────────────────────────────────────────
    if isinstance(best_model, Pipeline):
        importances = np.abs(best_model.named_steps["clf"].coef_[0])
    else:
        importances = best_model.feature_importances_

    feat_imp = (
        pd.DataFrame({"feature": MODEL_FEATURES, "importance": importances})
        .sort_values("importance")
    )

    # ── Per-route predicted probability ──────────────────────────────────────
    sample["delay_prob"] = best_model.predict_proba(sample[MODEL_FEATURES])[:, 1]

    # ── Route risk table ──────────────────────────────────────────────────────
    route_risk = (
        sample.groupby("ROUTE")
        .agg(
            flight_count  = ("Delayed",    "count"),
            actual_rate   = ("Delayed",    "mean"),
            expected_risk = ("delay_prob", "mean"),
        )
        .reset_index()
    )
    route_risk = route_risk[route_risk["flight_count"] >= 200]
    route_risk["actual_pct"]   = route_risk["actual_rate"]   * 100
    route_risk["expected_pct"] = route_risk["expected_risk"] * 100

    # ── Isolation Forest anomaly detection ───────────────────────────────────
    print("Running Isolation Forest …")
    route_daily = (
        sample.groupby(["ROUTE", "DATE"])
        .agg(daily_rate=("Delayed", "mean"), cnt=("Delayed", "count"))
        .reset_index().sort_values(["ROUTE", "DATE"])
    )
    route_daily["route_mean"] = route_daily.groupby("ROUTE")["daily_rate"].transform("mean")
    route_daily["roll30"]     = route_daily.groupby("ROUTE")["daily_rate"].transform(
        lambda s: s.rolling(30, min_periods=5).mean()
    )

    valid_routes = route_daily.groupby("ROUTE")["DATE"].count()[lambda x: x >= 30].index
    rmd = route_daily[route_daily["ROUTE"].isin(valid_routes)].dropna(subset=["roll30"]).copy()

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_STATE)
    rmd["anomaly"]       = iso.fit_predict(rmd[["daily_rate", "roll30", "route_mean", "cnt"]])
    rmd["anomaly_score"] = iso.decision_function(rmd[["daily_rate", "roll30", "route_mean", "cnt"]])

    anomaly_routes = (
        rmd[rmd["anomaly"] == -1]
        .groupby("ROUTE")
        .agg(
            anomalous_days = ("anomaly",       "count"),
            worst_score    = ("anomaly_score", "min"),
            peak_rate      = ("daily_rate",    "max"),
            baseline_rate  = ("route_mean",    "first"),
        )
        .reset_index()
        .sort_values("worst_score")
    )

    anomaly_flag = anomaly_routes[["ROUTE"]].copy()
    anomaly_flag["anomalous"] = True
    route_risk = route_risk.merge(anomaly_flag, on="ROUTE", how="left")
    route_risk["anomalous"] = route_risk["anomalous"].fillna(False)

    # ── Save all artifacts ────────────────────────────────────────────────────
    print("\nSaving artifacts …")
    sample.to_parquet(f"{OUT_DIR}/sample_with_probs.parquet", index=False)
    feat_imp.to_parquet(f"{OUT_DIR}/feat_imp.parquet",        index=False)
    route_risk.to_parquet(f"{OUT_DIR}/route_risk.parquet",    index=False)
    anomaly_routes.to_parquet(f"{OUT_DIR}/anomaly_routes.parquet", index=False)
    with open(f"{OUT_DIR}/best_name.txt", "w") as f:
        f.write(best_name)

    print(f"  ✓ models/sample_with_probs.parquet  ({len(sample):,} rows)")
    print(f"  ✓ models/feat_imp.parquet           ({len(feat_imp)} features)")
    print(f"  ✓ models/route_risk.parquet         ({len(route_risk)} routes)")
    print(f"  ✓ models/anomaly_routes.parquet     ({len(anomaly_routes)} anomalous routes)")
    print(f"  ✓ models/best_name.txt              → {best_name}")
    print("\nYou can now run:  streamlit run streamlit_app.py")
