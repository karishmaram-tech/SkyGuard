"""
SkyGuard — Step 5: Model Training, Evaluation, and Anomaly Detection

MODELS
  1. Logistic Regression  — linear baseline; fast, interpretable coefficients
  2. Random Forest        — non-linear ensemble; handles interactions natively
  3. XGBoost              — gradient boosting; typically highest AUC on tabular data

SPLIT STRATEGY
  Train : months 1–10  (January – October 2015)
  Test  : months 11–12 (November – December 2015)
  Why time-based, not random?
    A random split would mix future dates into training, so the expanding-window
    historical features (CARRIER_PRIOR_OTP, ORIGIN_PRIOR_CONG) would be computed
    with knowledge of November/December data. The time-based split enforces the
    same causal boundary as real deployment: the model always predicts a future
    flight using only the past.

ANOMALY DETECTION
  Isolation Forest on daily route-level delay-rate aggregates.
  Flags routes whose recent 30-day delay-rate window deviates from their
  full-year norm — a structural signal, not a per-flight prediction.

Inputs : `features_df`  — from feature_engineering.py
         `sample`       — for anomaly detection (needs ROUTE, DATE, Delayed)
Outputs: charts/05_feature_importance.png
         anomalous_routes.csv
         model_report.txt
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, roc_auc_score,
                                     classification_report, RocCurveDisplay)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs("charts", exist_ok=True)

RANDOM_STATE = 42
BG      = "#f7f8fa"
ACCENT  = "#3b82d4"
WARN    = "#e05c2a"
NEUTRAL = "#6b7280"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "sans-serif",
    "axes.labelsize"   : 11,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
})

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · Time-based train / test split
# ─────────────────────────────────────────────────────────────────────────────
# We split on the MONTH column that is already in features_df.
# Train = months 1-10; Test = months 11-12.
# This mirrors real-world deployment: the model is trained on available history
# and evaluated on the most-recent unseen period.

FEATURE_COLS = [c for c in features_df.columns if c != "Delayed"]
TARGET_COL   = "Delayed"

# features_df was built from `sample` which has MONTH; we need it for the split.
# Attach MONTH back if it was dropped during assembly.
if "MONTH" not in features_df.columns:
    features_df["MONTH"] = sample["MONTH"].values

train_mask = features_df["MONTH"] <= 10
test_mask  = features_df["MONTH"] >= 11

# Drop MONTH from the feature list — it was engineered into SEASON already,
# and leaving it in would let the model trivially memorise the split boundary.
MODEL_FEATURES = [c for c in FEATURE_COLS if c != "MONTH"]

X_train = features_df.loc[train_mask, MODEL_FEATURES]
X_test  = features_df.loc[test_mask,  MODEL_FEATURES]
y_train = features_df.loc[train_mask, TARGET_COL]
y_test  = features_df.loc[test_mask,  TARGET_COL]

print("── Train / test split ───────────────────────────────────────────────")
print(f"  Train (Jan–Oct) : {len(X_train):>9,} rows  "
      f"| delay rate: {y_train.mean()*100:.1f} %")
print(f"  Test  (Nov–Dec) : {len(X_test):>9,} rows  "
      f"| delay rate: {y_test.mean()*100:.1f} %")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · Helper: evaluate any trained classifier
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    """
    Fit the model, predict on the test set, and return a metrics dict.
    Uses predict_proba for AUC; falls back to decision_function if unavailable.
    """
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)

    metrics = {
        "model"    : name,
        "accuracy" : accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall"   : recall_score(y_te, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_te, y_prob),
    }
    return metrics, model, y_prob


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 · Model 1 — Logistic Regression (baseline)
# ─────────────────────────────────────────────────────────────────────────────
# Why Logistic Regression first?
#   It provides a linear baseline: if a simple linear decision boundary captures
#   most of the signal, complex models are adding noise, not value. It also
#   produces well-calibrated probabilities by default, making its AUC a fair
#   benchmark.
#
# Scale features: LR is sensitive to feature magnitude; StandardScaler brings
# all features to zero-mean / unit-variance. Tree models don't need scaling,
# but we include it here in a Pipeline so the scaler fits only on train data.

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
                   max_iter=1000,
                   class_weight="balanced",   # compensate for class imbalance
                   random_state=RANDOM_STATE,
               )),
])
print("Training Logistic Regression …")
lr_metrics, lr_model, lr_probs = evaluate(
    "Logistic Regression", lr_pipeline, X_train, y_train, X_test, y_test
)
print(f"  Done.  AUC = {lr_metrics['roc_auc']:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · Model 2 — Random Forest
# ─────────────────────────────────────────────────────────────────────────────
# n_estimators=300: enough trees to stabilise feature importances without
# excessive memory use in Colab (300 × ~12 features is manageable).
# max_depth=12: prevents individual trees from overfitting on the training months.
# class_weight="balanced_subsample": adjusts sample weights per tree,
#   appropriate for imbalanced targets in a subsampling ensemble.

print("Training Random Forest …")
rf_model = RandomForestClassifier(
    n_estimators   = 300,
    max_depth      = 12,
    min_samples_leaf = 20,     # each leaf needs ≥20 flights → guards against noise
    class_weight   = "balanced_subsample",
    n_jobs         = -1,       # use all available CPU cores
    random_state   = RANDOM_STATE,
)
rf_metrics, rf_model, rf_probs = evaluate(
    "Random Forest", rf_model, X_train, y_train, X_test, y_test
)
print(f"  Done.  AUC = {rf_metrics['roc_auc']:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 · Model 3 — XGBoost
# ─────────────────────────────────────────────────────────────────────────────
# XGBoost typically outperforms RF on structured/tabular data because:
#   • It builds trees sequentially, each correcting the residuals of the last.
#   • It has built-in L1/L2 regularisation (reg_alpha, reg_lambda).
#   • subsample + colsample_bytree add stochasticity that prevents overfitting.
#
# scale_pos_weight: ratio of negative to positive class — tells XGBoost how
# much to upweight the minority class (delayed flights).

neg  = int((y_train == 0).sum())
pos  = int((y_train == 1).sum())
spw  = neg / pos   # e.g. ~2.3 if 70 % on-time, 30 % delayed

print(f"Training XGBoost  (scale_pos_weight={spw:.2f}) …")
xgb_model = XGBClassifier(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = spw,
    use_label_encoder = False,
    eval_metric       = "auc",
    n_jobs            = -1,
    random_state      = RANDOM_STATE,
    verbosity         = 0,
)
xgb_metrics, xgb_model, xgb_probs = evaluate(
    "XGBoost", xgb_model, X_train, y_train, X_test, y_test
)
print(f"  Done.  AUC = {xgb_metrics['roc_auc']:.4f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 · Comparison table
# ─────────────────────────────────────────────────────────────────────────────

results = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
results_display = results.copy()
for col in ["accuracy", "precision", "recall", "roc_auc"]:
    results_display[col] = results_display[col].map("{:.4f}".format)

print("── Model comparison (test set: Nov–Dec 2015) ────────────────────────")
print(results_display.to_string(index=False))
print()

# Identify the best model by ROC-AUC
best_name = results.loc[results["roc_auc"].idxmax(), "model"]
best_model = {"Logistic Regression": lr_model,
              "Random Forest"      : rf_model,
              "XGBoost"            : xgb_model}[best_name]
best_probs = {"Logistic Regression": lr_probs,
              "Random Forest"      : rf_probs,
              "XGBoost"            : xgb_probs}[best_name]

print(f"  Best model by ROC-AUC: {best_name}")
print()

# Save full classification reports to text file
report_lines = [
    "SkyGuard — Model Evaluation Report",
    "=" * 60,
    f"\nTest set: November–December 2015",
    f"Train set: January–October 2015",
    f"\n{results_display.to_string(index=False)}",
    f"\nBest model: {best_name}\n",
]
for name, model, probs in [
    ("Logistic Regression", lr_model, lr_probs),
    ("Random Forest",       rf_model, rf_probs),
    ("XGBoost",             xgb_model, xgb_probs),
]:
    preds = model.predict(X_test)
    report_lines.append(f"\n{'─'*40}\n{name}\n{'─'*40}")
    report_lines.append(classification_report(y_test, preds,
                                              target_names=["On-time","Delayed"]))

with open("model_report.txt", "w") as fh:
    fh.write("\n".join(report_lines))
print("✓ Detailed classification reports saved to model_report.txt")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 · ROC curve comparison chart
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
colors  = [NEUTRAL, ACCENT, WARN]
for (name, probs), color in zip(
    [("Logistic Regression", lr_probs),
     ("Random Forest",       rf_probs),
     ("XGBoost",             xgb_probs)],
    colors,
):
    auc = roc_auc_score(y_test, probs)
    RocCurveDisplay.from_predictions(
        y_test, probs, name=f"{name}  (AUC={auc:.3f})",
        ax=ax, color=color,
    )

ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
ax.set_title("ROC Curve — All Models  (Test: Nov–Dec 2015)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig("charts/05_roc_curves.png", dpi=150)
plt.close(fig)
print("✓ ROC curves saved: charts/05_roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 · Feature importance chart for the best model
# ─────────────────────────────────────────────────────────────────────────────
# For tree-based models (RF, XGBoost) we use the native feature_importances_
# attribute (mean impurity decrease / gain).  For Logistic Regression we use
# the absolute value of the standardised coefficients.

def get_importances(model, feature_names):
    """Extract feature importances regardless of model type."""
    if isinstance(model, Pipeline):
        clf = model.named_steps["clf"]
        return np.abs(clf.coef_[0])          # LR: abs coefficients post-scaling
    elif isinstance(model, XGBClassifier):
        return model.feature_importances_    # XGB: gain-based importance
    elif isinstance(model, RandomForestClassifier):
        return model.feature_importances_    # RF: mean impurity decrease
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

importances = get_importances(best_model, MODEL_FEATURES)
imp_df = (
    pd.DataFrame({"feature": MODEL_FEATURES, "importance": importances})
    .sort_values("importance", ascending=True)
)

# Colour the top-3 features distinctly
bar_colors = [NEUTRAL] * len(imp_df)
bar_colors[-1] = WARN
bar_colors[-2] = WARN
bar_colors[-3] = WARN

fig, ax = plt.subplots(figsize=(9, max(4, len(imp_df) * 0.55)))
ax.barh(imp_df["feature"], imp_df["importance"], color=bar_colors, height=0.65)
ax.set_xlabel("Feature importance" +
              (" (|coefficient|)" if isinstance(best_model, Pipeline) else " (gain)"))
ax.set_title(f"Feature Importance — {best_name}",
             fontsize=12, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig("charts/05_feature_importance.png", dpi=150)
plt.close(fig)
print(f"✓ Feature importance chart saved: charts/05_feature_importance.png")
print()

print("── Top feature importances ──────────────────────────────────────────")
print(imp_df[::-1].to_string(index=False))
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 · Isolation Forest — route-level anomaly detection
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT WE ARE DETECTING
# ─────────────────────
# A per-flight classifier tells you the risk for a single departure.
# Route-level anomaly detection asks a different question:
#   "Is this route's recent delay pattern structurally different from its
#    own historical norm?"
# That catches systemic problems: a hub that has been steadily degrading,
# a carrier pulling resources from a thin route, or a recurring weather pattern
# on a specific corridor.
#
# METHOD
# ──────
# 1. Aggregate the sample to daily (route, date) delay rates.
# 2. For each route, compute a 30-day rolling mean delay rate — this is the
#    "recent pattern" feature the Isolation Forest sees alongside the raw
#    daily rate and the route's overall mean.
# 3. Fit Isolation Forest on routes that have at least MIN_DAYS observations
#    (sparse routes produce noisy rolling windows).
# 4. Flag routes where the model scores them as anomalous (prediction = -1).
# 5. Rank by anomaly score (lower = more anomalous) and save to CSV.
#
# contamination=0.05 means we expect ~5 % of route-days to be genuinely
# anomalous — conservative enough to surface real operational outliers
# without generating unworkable alert volumes.

MIN_ROUTE_DAYS  = 30    # minimum daily observations for a route to be analysed
CONTAMINATION   = 0.05  # expected fraction of anomalous route-day observations

print("── Isolation Forest anomaly detection ───────────────────────────────")

# Step 1: daily route-level delay rate
route_daily = (
    sample.groupby(["ROUTE", "DATE"])
    .agg(
        daily_delay_rate=("Delayed", "mean"),
        flight_count    =("Delayed", "count"),
    )
    .reset_index()
    .sort_values(["ROUTE", "DATE"])
)

# Step 2: route overall mean and rolling 30-day mean (per route)
route_daily["route_overall_mean"] = (
    route_daily.groupby("ROUTE")["daily_delay_rate"]
    .transform("mean")
)
route_daily["rolling_30d_rate"] = (
    route_daily.groupby("ROUTE")["daily_delay_rate"]
    .transform(lambda s: s.rolling(30, min_periods=5).mean())
)

# Step 3: keep only routes with enough observations
route_obs_count = route_daily.groupby("ROUTE")["DATE"].count()
valid_routes    = route_obs_count[route_obs_count >= MIN_ROUTE_DAYS].index
route_model_df  = route_daily[route_daily["ROUTE"].isin(valid_routes)].dropna(
    subset=["rolling_30d_rate"]
).copy()

IF_FEATURES = ["daily_delay_rate", "rolling_30d_rate",
               "route_overall_mean", "flight_count"]

iso = IsolationForest(
    n_estimators  = 200,
    contamination = CONTAMINATION,
    random_state  = RANDOM_STATE,
)
route_model_df["anomaly_score"]     = iso.fit_predict(route_model_df[IF_FEATURES])
# Raw decision scores: more negative = more anomalous
route_model_df["anomaly_raw_score"] = iso.decision_function(route_model_df[IF_FEATURES])

anomalies = (
    route_model_df[route_model_df["anomaly_score"] == -1]
    .copy()
    .sort_values("anomaly_raw_score")   # most anomalous first
)

# Summarise at route level: how many anomalous days, how extreme
route_summary = (
    anomalies.groupby("ROUTE")
    .agg(
        anomalous_days    =("anomaly_score",     "count"),
        worst_score       =("anomaly_raw_score", "min"),
        peak_daily_rate   =("daily_delay_rate",  "max"),
        route_overall_mean=("route_overall_mean","first"),
    )
    .reset_index()
    .sort_values("worst_score")         # most anomalous route first
)

# Save full anomalous route list
route_summary.to_csv("anomalous_routes.csv", index=False)
print(f"  Routes analysed (≥{MIN_ROUTE_DAYS} days of data): {len(valid_routes):,}")
print(f"  Anomalous route-days flagged (contamination={CONTAMINATION}): "
      f"{len(anomalies):,}")
print(f"  Unique anomalous routes: {len(route_summary):,}")
print()
print("  Top 10 most anomalous routes:")
top10_anom = route_summary.head(10)[
    ["ROUTE", "anomalous_days", "peak_daily_rate", "route_overall_mean"]
].copy()
top10_anom["peak_daily_rate"]    = top10_anom["peak_daily_rate"].map("{:.1%}".format)
top10_anom["route_overall_mean"] = top10_anom["route_overall_mean"].map("{:.1%}".format)
print(top10_anom.to_string(index=False))
print()
print("✓ Full anomalous route list saved to anomalous_routes.csv")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 · Anomaly chart — top 15 routes by anomalous-day count
# ─────────────────────────────────────────────────────────────────────────────

top15 = route_summary.head(15).sort_values("anomalous_days")

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(top15["ROUTE"], top15["anomalous_days"], color=WARN, height=0.65)

for bar, (_, row) in zip(bars, top15.iterrows()):
    ax.text(bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"peak {row['peak_daily_rate']}  (norm {row['route_overall_mean']})",
            va="center", fontsize=9, color=NEUTRAL)

ax.set_xlabel("Number of anomalous days flagged")
ax.set_title(
    f"Top 15 Routes with Most Anomalous Delay Days\n"
    f"(Isolation Forest, contamination={CONTAMINATION}, min {MIN_ROUTE_DAYS} days observed)",
    fontsize=12, fontweight="bold", pad=10,
)
fig.tight_layout()
fig.savefig("charts/06_anomalous_routes.png", dpi=150)
plt.close(fig)
print("✓ Anomaly chart saved: charts/06_anomalous_routes.png")
print()

print("═" * 60)
print("  SkyGuard Step 5 complete.")
print(f"  Best classifier : {best_name}")
print(f"  ROC-AUC         : {results.loc[results['roc_auc'].idxmax(),'roc_auc']:.4f}")
print(f"  Anomalous routes: {len(route_summary):,}  →  anomalous_routes.csv")
print("═" * 60)
