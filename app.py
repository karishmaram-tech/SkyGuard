"""
SkyGuard — Streamlit Application  (single file)
================================================
Four pages:
  1. Executive Overview      — KPI cards + monthly trend + on-time by carrier
  2. Analytical Deep Dive    — heatmap (hour × DOW) + delay-cause breakdown
  3. Risk & Anomaly Detection — route risk table + expected-vs-actual scatter
                                + feature importance chart
  4. AI Insights             — Gemini briefings for top-3 anomalous routes
                                + route-lookup search box

Sidebar filters (carrier + date range) apply across all pages.

Run:
    streamlit run app.py

Dependencies:
    pip install streamlit pandas numpy matplotlib scikit-learn xgboost
                google-generativeai

Data files expected in the same folder:
    flights.csv  airlines.csv  airports.csv
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyGuard · Flight Analytics",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette (matches the existing analysis files) ─────────────────────────────
ACCENT  = "#3b82d4"
WARN    = "#e05c2a"
NEUTRAL = "#6b7280"
GREEN   = "#2a9d5c"
BG      = "#f7f8fa"

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

FAA_DELAY_THRESHOLD = 15          # minutes — FAA / BTS official definition
RANDOM_STATE        = 42
SAMPLE_SIZE         = 800_000

DELAY_CAUSE_COLS = [
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
]
DELAY_CAUSE_LABELS = {
    "AIR_SYSTEM_DELAY"  : "NAS / Air System",
    "SECURITY_DELAY"    : "Security",
    "AIRLINE_DELAY"     : "Carrier",
    "LATE_AIRCRAFT_DELAY": "Late Aircraft",
    "WEATHER_DELAY"     : "Weather",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading data …")
def load_data() -> pd.DataFrame:
    """
    Load the preprocessed sample from sample.parquet (fast path, < 1 s).
    If sample.parquet does not exist, falls back to building it from the raw
    CSVs — but you should run `python prepare_data.py` once beforehand so
    the app starts instantly every time.
    """
    PARQUET = "sample.parquet"

    if os.path.exists(PARQUET):
        sample = pd.read_parquet(PARQUET)
        # DATE column is stored as object in parquet when it comes from .dt.date;
        # restore it so downstream groupbys work correctly.
        if "SCHEDULED_DATETIME" in sample.columns:
            sample["SCHEDULED_DATETIME"] = pd.to_datetime(
                sample["SCHEDULED_DATETIME"], errors="coerce")
        if "DATE" in sample.columns:
            sample["DATE"] = pd.to_datetime(sample["DATE"], errors="coerce").dt.date
        return sample

    # ── Fallback: build from raw CSVs (slow — run prepare_data.py instead) ────
    st.warning(
        "**sample.parquet not found.** "
        "Run `python prepare_data.py` once to pre-build the dataset and "
        "eliminate this slow path. Loading from raw CSVs now …"
    )
    flights  = pd.read_csv("flights.csv",  low_memory=False)
    airlines = pd.read_csv("airlines.csv")
    airports = pd.read_csv("airports.csv")

    df = flights.merge(
        airlines.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}),
        on="AIRLINE", how="left",
    )
    orig = airports.rename(columns={
        "IATA_CODE": "ORIGIN_AIRPORT", "AIRPORT": "ORIGIN_AIRPORT_NAME",
        "CITY": "ORIGIN_CITY", "STATE": "ORIGIN_STATE",
        "LATITUDE": "ORIGIN_LATITUDE", "LONGITUDE": "ORIGIN_LONGITUDE",
    })
    dest = airports.rename(columns={
        "IATA_CODE": "DESTINATION_AIRPORT", "AIRPORT": "DEST_AIRPORT_NAME",
        "CITY": "DEST_CITY", "STATE": "DEST_STATE",
        "LATITUDE": "DEST_LATITUDE", "LONGITUDE": "DEST_LONGITUDE",
    })
    df = df.merge(orig, on="ORIGIN_AIRPORT",      how="left")
    df = df.merge(dest, on="DESTINATION_AIRPORT", how="left")

    time_str = df["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4)
    date_str = (
        df["YEAR"].astype(str) + "-" +
        df["MONTH"].astype(str).str.zfill(2) + "-" +
        df["DAY"].astype(str).str.zfill(2) + " " +
        time_str.str[:2] + ":" + time_str.str[2:]
    )
    df["SCHEDULED_DATETIME"] = pd.to_datetime(date_str, format="%Y-%m-%d %H:%M",
                                               errors="coerce")
    df["DEPARTURE_HOUR"] = df["SCHEDULED_DATETIME"].dt.hour
    df["DATE"]           = df["SCHEDULED_DATETIME"].dt.date

    df["CANCELLED"] = df["CANCELLED"].astype(bool)
    df["DIVERTED"]  = df["DIVERTED"].astype(bool)
    operable = df[(~df["CANCELLED"]) & (~df["DIVERTED"])].copy()

    n = min(SAMPLE_SIZE, len(operable))
    sample = operable.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    sample["Delayed"]     = (sample["ARRIVAL_DELAY"] >= FAA_DELAY_THRESHOLD).astype(int)
    sample["DAY_OF_WEEK"] = sample["SCHEDULED_DATETIME"].dt.dayofweek
    sample["ROUTE"]       = sample["ORIGIN_AIRPORT"] + " → " + sample["DESTINATION_AIRPORT"]
    sample["DISTANCE"]    = sample["DISTANCE"].fillna(sample["DISTANCE"].median())
    for col in DELAY_CAUSE_COLS:
        if col in sample.columns:
            sample[col] = sample[col].fillna(0)

    return sample


@st.cache_data(show_spinner="Loading model artifacts …")
def build_features_and_models(_sample: pd.DataFrame):
    """
    Fast path: load pre-built artifacts from the models/ folder produced by
    train_models.py.  Falls back to full training if the folder is missing
    (e.g. first run without having run train_models.py).
    Returns (sample_with_probs, best_name, feat_imp, route_risk, anomaly_routes).
    """
    OUT_DIR = "models"

    if os.path.exists(OUT_DIR) and os.path.exists(f"{OUT_DIR}/best_name.txt"):
        sample         = pd.read_parquet(f"{OUT_DIR}/sample_with_probs.parquet")
        feat_imp       = pd.read_parquet(f"{OUT_DIR}/feat_imp.parquet")
        route_risk     = pd.read_parquet(f"{OUT_DIR}/route_risk.parquet")
        anomaly_routes = pd.read_parquet(f"{OUT_DIR}/anomaly_routes.parquet")
        with open(f"{OUT_DIR}/best_name.txt") as f:
            best_name = f.read().strip()
        # Restore datetime types lost in parquet round-trip
        sample["SCHEDULED_DATETIME"] = pd.to_datetime(
            sample["SCHEDULED_DATETIME"], errors="coerce")
        sample["DATE"] = pd.to_datetime(sample["DATE"], errors="coerce").dt.date
        return sample, best_name, feat_imp, route_risk, anomaly_routes

    # ── Fallback: train from scratch (run train_models.py to avoid this) ──────
    st.warning(
        "**models/ folder not found.** "
        "Run `python train_models.py` once to pre-train and eliminate this slow path."
    )
    from sklearn.linear_model  import LogisticRegression
    from sklearn.ensemble      import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline      import Pipeline
    from sklearn.metrics       import roc_auc_score
    from xgboost               import XGBClassifier

    sample = _sample.copy()

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

    X_train = sample.loc[sample["MONTH"] <= 10, MODEL_FEATURES]
    X_test  = sample.loc[sample["MONTH"] >= 11, MODEL_FEATURES]
    y_train = sample.loc[sample["MONTH"] <= 10, "Delayed"]
    y_test  = sample.loc[sample["MONTH"] >= 11, "Delayed"]

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())

    lr = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    rf  = RandomForestClassifier(n_estimators=300, max_depth=12,
                                 min_samples_leaf=20, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RANDOM_STATE)
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg/pos,
                        eval_metric="auc", n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)

    results = []
    fitted  = {}
    for name, mdl in [("Logistic Regression", lr), ("Random Forest", rf), ("XGBoost", xgb)]:
        mdl.fit(X_train, y_train)
        proba = mdl.predict_proba(X_test)[:, 1]
        results.append({"model": name, "roc_auc": roc_auc_score(y_test, proba)})
        fitted[name] = mdl

    best_name  = max(results, key=lambda r: r["roc_auc"])["model"]
    best_model = fitted[best_name]

    if isinstance(best_model, Pipeline):
        importances = np.abs(best_model.named_steps["clf"].coef_[0])
    else:
        importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({"feature": MODEL_FEATURES,
                             "importance": importances}).sort_values("importance")

    sample["delay_prob"] = best_model.predict_proba(sample[MODEL_FEATURES])[:, 1]

    route_risk = (
        sample.groupby("ROUTE")
        .agg(flight_count=("Delayed","count"), actual_rate=("Delayed","mean"),
             expected_risk=("delay_prob","mean"))
        .reset_index()
    )
    route_risk = route_risk[route_risk["flight_count"] >= 200]
    route_risk["actual_pct"]   = route_risk["actual_rate"]   * 100
    route_risk["expected_pct"] = route_risk["expected_risk"] * 100

    route_daily = (
        sample.groupby(["ROUTE", "DATE"])
        .agg(daily_rate=("Delayed","mean"), cnt=("Delayed","count"))
        .reset_index().sort_values(["ROUTE","DATE"])
    )
    route_daily["route_mean"] = route_daily.groupby("ROUTE")["daily_rate"].transform("mean")
    route_daily["roll30"]     = route_daily.groupby("ROUTE")["daily_rate"].transform(
        lambda s: s.rolling(30, min_periods=5).mean()
    )
    valid_routes = route_daily.groupby("ROUTE")["DATE"].count()[lambda x: x >= 30].index
    rmd = route_daily[route_daily["ROUTE"].isin(valid_routes)].dropna(subset=["roll30"]).copy()

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=RANDOM_STATE)
    rmd["anomaly"]       = iso.fit_predict(rmd[["daily_rate","roll30","route_mean","cnt"]])
    rmd["anomaly_score"] = iso.decision_function(rmd[["daily_rate","roll30","route_mean","cnt"]])

    anomaly_routes = (
        rmd[rmd["anomaly"] == -1]
        .groupby("ROUTE")
        .agg(anomalous_days=("anomaly","count"), worst_score=("anomaly_score","min"),
             peak_rate=("daily_rate","max"), baseline_rate=("route_mean","first"))
        .reset_index().sort_values("worst_score")
    )

    anomaly_flag = anomaly_routes[["ROUTE"]].copy()
    anomaly_flag["anomalous"] = True
    route_risk = route_risk.merge(anomaly_flag, on="ROUTE", how="left")
    route_risk["anomalous"] = route_risk["anomalous"].fillna(False)

    return sample, best_name, feat_imp, route_risk, anomaly_routes


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS  (return matplotlib Figure — Streamlit renders with st.pyplot)
# ══════════════════════════════════════════════════════════════════════════════

def chart_monthly_trend(df: pd.DataFrame) -> plt.Figure:
    # Compute mean and SEM separately — mixing string shortcuts and lambdas
    # in a single .agg() call on a Series raises SpecificationError (pandas >= 1.3)
    grp = df.groupby("MONTH")["ARRIVAL_DELAY"]
    monthly = pd.DataFrame({
        "MONTH": grp.mean().index,
        "mean" : grp.mean().values,
        "sem"  : (grp.std() / np.sqrt(grp.count())).values,
    })
    labels = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(monthly["MONTH"],
                    monthly["mean"] - monthly["sem"],
                    monthly["mean"] + monthly["sem"],
                    color=ACCENT, alpha=0.15)
    ax.plot(monthly["MONTH"], monthly["mean"],
            color=ACCENT, lw=2.5, marker="o", ms=5)
    ax.axhline(0, color=NEUTRAL, lw=0.8, ls="--")
    ax.set_xticks(monthly["MONTH"])
    ax.set_xticklabels([labels[m-1] for m in monthly["MONTH"]])
    ax.set_ylabel("Avg arrival delay (min)")
    ax.set_title("Monthly Average Arrival Delay", fontweight="bold")
    fig.tight_layout()
    return fig


def chart_ontime_carrier(df: pd.DataFrame) -> plt.Figure:
    stats = (
        df.groupby(["AIRLINE","AIRLINE_NAME"])
        .agg(cnt=("Delayed","count"), otp=("Delayed", lambda x: (1-x.mean())*100))
        .reset_index()
    )
    stats = stats[stats["cnt"] >= 200].sort_values("otp")
    n = len(stats)
    colors = [NEUTRAL]*n
    for i in range(min(3,n)):      colors[i]   = WARN
    for i in range(1,min(4,n+1)): colors[n-i] = GREEN

    fig, ax = plt.subplots(figsize=(9, max(3.5, n*0.52)))
    bars = ax.barh(stats["AIRLINE_NAME"], stats["otp"], color=colors, height=0.65)
    for bar, val in zip(bars, stats["otp"]):
        ax.text(max(val-3,1), bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="right",
                fontsize=9, color="white", fontweight="bold")
    net = stats["otp"].mean()
    ax.axvline(net, color=NEUTRAL, lw=1, ls="--", label=f"Network avg {net:.1f}%")
    ax.set_xlim(0,100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("On-time arrival rate (< 15 min late)")
    ax.set_title("On-Time % by Carrier", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def chart_heatmap(df: pd.DataFrame) -> plt.Figure:
    raw = (
        df.groupby(["DAY_OF_WEEK","DEPARTURE_HOUR"])["ARRIVAL_DELAY"]
        .agg(mean_delay="mean", count="count")
        .reset_index()
    )
    piv_d = raw.pivot(index="DAY_OF_WEEK", columns="DEPARTURE_HOUR",
                      values="mean_delay").reindex(index=range(7), columns=range(24))
    piv_c = raw.pivot(index="DAY_OF_WEEK", columns="DEPARTURE_HOUR",
                      values="count").reindex(index=range(7), columns=range(24))
    masked = piv_d.where(piv_c >= 30)

    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(masked.values, aspect="auto", cmap="RdYlGn_r", vmin=-5, vmax=30)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    ax.set_title("Avg Arrival Delay by Hour × Day of Week  (grey = < 30 flights)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="Avg delay (min)")
    fig.tight_layout()
    return fig


def chart_delay_cause(df: pd.DataFrame) -> plt.Figure:
    present = [c for c in DELAY_CAUSE_COLS if c in df.columns]
    totals  = df[present].sum()
    totals  = totals[totals > 0].sort_values()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    if totals.empty:
        # Guard: no delayed flights in current filter — avoid .max() on empty Series
        ax.text(0.5, 0.5, "No delay-cause data for current filter",
                ha="center", va="center", transform=ax.transAxes, color=NEUTRAL)
        ax.set_title("Delay-Cause Breakdown (total minutes)", fontweight="bold")
        fig.tight_layout()
        return fig

    labels  = [DELAY_CAUSE_LABELS.get(c, c) for c in totals.index]
    max_val = totals.max()
    colors  = [WARN if v == max_val else ACCENT for v in totals.values]
    bars = ax.barh(labels, totals.values, color=colors, height=0.6)
    for bar, val in zip(bars, totals.values):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f"{val:,.0f} min", va="center", fontsize=9, color=NEUTRAL)
    ax.set_xlabel("Total delay minutes")
    ax.set_title("Delay-Cause Breakdown (total minutes)", fontweight="bold")
    fig.tight_layout()
    return fig


def chart_scatter(route_risk: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))

    if route_risk.empty:
        # Guard: no qualifying routes in current filter — avoid NaN lim calculation
        ax.text(0.5, 0.5, "No route data for current filter",
                ha="center", va="center", transform=ax.transAxes, color=NEUTRAL)
        ax.set_title("Expected vs Actual Delay Rate by Route", fontweight="bold")
        fig.tight_layout()
        return fig

    norm = route_risk[~route_risk["anomalous"]]
    anom = route_risk[route_risk["anomalous"]]

    if not norm.empty:
        ax.scatter(norm["expected_pct"], norm["actual_pct"],
                   s=norm["flight_count"].clip(upper=3000)/30,
                   color=ACCENT, alpha=0.5, label="Normal route", linewidths=0)
    if not anom.empty:
        ax.scatter(anom["expected_pct"], anom["actual_pct"],
                   s=anom["flight_count"].clip(upper=3000)/30,
                   color=WARN, alpha=0.85, label="Anomalous route",
                   edgecolors="black", linewidths=0.5)

    e_max = route_risk["expected_pct"].max()
    a_max = route_risk["actual_pct"].max()
    lim = max(e_max if pd.notna(e_max) else 0,
              a_max if pd.notna(a_max) else 0) * 1.05 or 100
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Expected = Actual")
    ax.set_xlabel("Model expected risk (%)")
    ax.set_ylabel("Actual observed delay rate (%)")
    ax.set_title("Expected vs Actual Delay Rate by Route\n"
                 "(dot size ∝ flight volume)", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def chart_feature_importance(feat_imp: pd.DataFrame, best_name: str) -> plt.Figure:
    colors = [NEUTRAL]*len(feat_imp)
    colors[-1] = WARN;  colors[-2] = WARN;  colors[-3] = WARN

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(feat_imp)*0.55)))
    ax.barh(feat_imp["feature"], feat_imp["importance"],
            color=colors, height=0.65)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Feature Importance — {best_name}", fontweight="bold")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# AI INSIGHTS  (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RouteFacts:
    route             : str
    expected_risk_pct : float
    actual_rate_pct   : float
    top_cause         : str
    top_cause_pct     : float
    sample_size       : int


_PROMPT_TEMPLATE = (
    "You are an airline operations analyst. "
    "Given these facts about Route {route}: "
    "expected on-time risk {expected_risk_pct:.1f}%, "
    "actual last-30-day rate {actual_rate_pct:.1f}%, "
    "primary delay cause {top_cause} ({top_cause_pct:.1f}% of delays), "
    "sample size {sample_size} flights — "
    "write a 3-sentence briefing: "
    "(1) state the fact, "
    "(2) explain the likely driver, "
    "(3) recommend one concrete operational action. "
    "Do not invent numbers not provided above."
)


def generate_route_briefing(facts: RouteFacts,
                            temperature: float = 0.3,
                            max_tokens: int = 256) -> str:
    """Call Gemini API with the fixed prompt template. Reads key from env."""
    from google import genai  # lazy import — google-genai (replaces google-generativeai)
    from google.genai import types as genai_types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your environment or "
            "Streamlit secrets."
        )
    client = genai.Client(api_key=api_key)
    prompt = _PROMPT_TEMPLATE.format(**facts.__dict__)
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text.strip()


def top_cause_for_route(df: pd.DataFrame, route: str):
    """Return (cause_label, pct_of_delays) for the dominant delay cause on a route."""
    sub = df[df["ROUTE"] == route]
    delayed = sub[sub["Delayed"] == 1]
    if delayed.empty:
        return "Unknown", 0.0
    present = [c for c in DELAY_CAUSE_COLS if c in delayed.columns]
    totals  = delayed[present].sum()
    if totals.sum() == 0:
        return "Unknown", 0.0
    top_col = totals.idxmax()
    pct = totals[top_col] / totals.sum() * 100
    return DELAY_CAUSE_LABELS.get(top_col, top_col), float(pct)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  — carrier + date-range filters
# ══════════════════════════════════════════════════════════════════════════════

def apply_sidebar_filters(sample: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar widgets and return the filtered dataframe."""
    with st.sidebar:
        st.markdown("## ✈ SkyGuard Filters")
        st.markdown("---")

        # Carrier filter
        all_carriers = sorted(sample["AIRLINE_NAME"].dropna().unique())
        chosen_carriers = st.multiselect(
            "Carrier",
            options=all_carriers,
            default=all_carriers,
            help="Select one or more carriers to include across all pages.",
        )

        # Date-range filter (month granularity — keeps it fast)
        st.markdown("**Month range**")
        month_min, month_max = st.select_slider(
            "Months (2015)",
            options=list(range(1, 13)),
            value=(1, 12),
            format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                   "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
        )

        st.markdown("---")
        st.caption(f"Sample size (full): {len(sample):,}")

    # Apply filters
    mask = (
        sample["AIRLINE_NAME"].isin(chosen_carriers) &
        sample["MONTH"].between(month_min, month_max)
    )
    filtered = sample[mask].copy()
    with st.sidebar:
        st.caption(f"Filtered rows: {len(filtered):,}")

    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS HELPER
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(col, label: str, value: str, delta: str = "", color: str = ACCENT):
    col.markdown(
        f"""
        <div style="background:{BG};border:1px solid #e5e7eb;border-radius:8px;
                    padding:16px 20px;text-align:center;">
            <div style="font-size:12px;color:{NEUTRAL};font-weight:600;
                        letter-spacing:0.05em;text-transform:uppercase;">
                {label}
            </div>
            <div style="font-size:28px;font-weight:700;color:{color};
                        margin:6px 0 2px;">
                {value}
            </div>
            <div style="font-size:11px;color:{NEUTRAL};">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data & models ────────────────────────────────────────────────────
    try:
        raw_sample = load_data()
    except FileNotFoundError as e:
        st.error(
            f"Could not find a required CSV file: **{e}**\n\n"
            "Place `flights.csv`, `airlines.csv`, and `airports.csv` in the "
            "same folder as `app.py` and restart."
        )
        st.stop()

    # Build features & models once (cached — spinner shown by @st.cache_data)
    (sample_feat, best_name, feat_imp,
     route_risk, anomaly_routes) = build_features_and_models(raw_sample)

    # Apply sidebar filters to the feature-enriched sample
    df = apply_sidebar_filters(sample_feat)

    if len(df) == 0:
        st.warning("No flights match the current filters. Adjust the sidebar.")
        st.stop()

    # ── Page navigation ───────────────────────────────────────────────────────
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Executive Overview",
         "🔍 Analytical Deep Dive",
         "⚠ Risk & Anomaly Detection",
         "🤖 AI Insights"],
        label_visibility="collapsed",
    )

    # ── Filter route_risk to carriers present in filtered df ─────────────────
    visible_routes = df["ROUTE"].unique()
    rr_filtered    = route_risk[route_risk["ROUTE"].isin(visible_routes)]

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 · Executive Overview
    # ══════════════════════════════════════════════════════════════════════════
    if page == "📊 Executive Overview":
        st.title("📊 Executive Overview")
        st.caption("Top-line performance metrics and seasonal / carrier trends.")
        st.markdown("---")

        # KPI cards
        ontime_pct    = (1 - df["Delayed"].mean()) * 100
        avg_delay     = df["ARRIVAL_DELAY"].mean()
        total_flights = len(df)

        # Top delay cause
        present = [c for c in DELAY_CAUSE_COLS if c in df.columns]
        cause_totals  = df[df["Delayed"]==1][present].sum()
        top_cause_col = cause_totals.idxmax() if cause_totals.sum() > 0 else None
        top_cause_lbl = DELAY_CAUSE_LABELS.get(top_cause_col, "N/A") \
                        if top_cause_col else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "On-Time Rate",     f"{ontime_pct:.1f}%",
                 "arrivals < 15 min late", GREEN if ontime_pct >= 80 else WARN)
        kpi_card(c2, "Avg Arrival Delay", f"{avg_delay:.1f} min",
                 "all operable flights",
                 WARN if avg_delay > 10 else ACCENT)
        kpi_card(c3, "Top Delay Driver",  top_cause_lbl,
                 "by total delay minutes", WARN)
        kpi_card(c4, "Flights Analysed",  f"{total_flights:,}",
                 "after carrier/date filter", NEUTRAL)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Monthly Avg Delay Trend")
            st.pyplot(chart_monthly_trend(df), use_container_width=True)
        with col_right:
            st.subheader("On-Time % by Carrier")
            st.pyplot(chart_ontime_carrier(df), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 · Analytical Deep Dive
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Analytical Deep Dive":
        st.title("🔍 Analytical Deep Dive")
        st.caption("When delays cluster (hour × day) and what drives them.")
        st.markdown("---")

        st.subheader("Delay Heatmap — Hour of Day × Day of Week")
        st.markdown(
            "Colour shows mean arrival delay (minutes). "
            "Grey cells have fewer than 30 flights in the filtered selection."
        )
        st.pyplot(chart_heatmap(df), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Delay-Cause Breakdown")
        st.markdown(
            "Total delay minutes attributed to each official BTS cause category "
            "across delayed flights in the current filter."
        )
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            st.pyplot(chart_delay_cause(df), use_container_width=True)
        with col_table:
            present = [c for c in DELAY_CAUSE_COLS if c in df.columns]
            delayed_df = df[df["Delayed"]==1]
            cause_sum  = delayed_df[present].sum().rename(
                index=DELAY_CAUSE_LABELS)
            cause_pct  = (cause_sum / cause_sum.sum() * 100).round(1)
            tbl = pd.DataFrame({
                "Total minutes"  : cause_sum.map("{:,.0f}".format),
                "Share of delays": cause_pct.map("{}%".format),
            })
            st.dataframe(tbl, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 · Risk & Anomaly Detection
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "⚠ Risk & Anomaly Detection":
        st.title("⚠ Risk & Anomaly Detection")
        st.caption(
            f"Model: **{best_name}** · "
            "Routes with ≥ 200 flights · "
            "Anomalies flagged by Isolation Forest (contamination = 5 %)"
        )
        st.markdown("---")

        # ── Route risk table ──────────────────────────────────────────────────
        st.subheader("Route Risk Table")
        st.markdown(
            "Sorted by actual delay rate. "
            "🔴 = flagged anomalous by Isolation Forest."
        )
        display_rr = rr_filtered.copy().sort_values("actual_pct", ascending=False)
        display_rr["anomalous"] = display_rr["anomalous"].apply(
            lambda x: "🔴 Yes" if x else "✅ No")
        display_rr["actual_pct"]   = display_rr["actual_pct"].round(1)
        display_rr["expected_pct"] = display_rr["expected_pct"].round(1)
        st.dataframe(
            display_rr[["ROUTE","flight_count","actual_pct",
                         "expected_pct","anomalous"]]
            .rename(columns={
                "flight_count" : "Flights",
                "actual_pct"   : "Actual delay %",
                "expected_pct" : "Model risk %",
                "anomalous"    : "Anomalous?",
            }),
            use_container_width=True,
            height=320,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Expected vs Actual Delay Rate")
            st.markdown(
                "Points above the dashed line are performing *worse* than "
                "the model predicted. Red dots = anomalous routes."
            )
            st.pyplot(chart_scatter(rr_filtered), use_container_width=True)
        with col_r:
            st.subheader(f"Feature Importance — {best_name}")
            st.pyplot(chart_feature_importance(feat_imp, best_name),
                      use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 · AI Insights
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🤖 AI Insights":
        st.title("🤖 AI Insights")
        st.caption(
            "Plain-English operational briefings generated by Gemini 1.5 Flash "
            "using the exact facts computed from this dataset — no hallucinated numbers."
        )
        st.markdown("---")

        # API key check
        api_key_present = bool(os.environ.get("GEMINI_API_KEY"))
        if not api_key_present:
            st.warning(
                "**GEMINI_API_KEY** is not set. "
                "Set it in your environment before using this page:\n\n"
                "```bash\nexport GEMINI_API_KEY='your-key'\n```"
            )

        # ── Top-3 anomalous routes briefings ──────────────────────────────────
        st.subheader("Briefings for the 3 Most Anomalous Routes")

        top3 = anomaly_routes.head(3)

        for _, arow in top3.iterrows():
            route = arow["ROUTE"]
            # Pull facts from the full sample (before sidebar filtering)
            route_sample = sample_feat[sample_feat["ROUTE"] == route]
            actual_rate  = route_sample["Delayed"].mean() * 100
            expected_pct = route_sample["delay_prob"].mean() * 100 \
                           if "delay_prob" in route_sample.columns else actual_rate
            cause_lbl, cause_pct = top_cause_for_route(sample_feat, route)
            n_flights = len(route_sample)

            facts = RouteFacts(
                route              = route,
                expected_risk_pct  = round(expected_pct, 1),
                actual_rate_pct    = round(actual_rate,  1),
                top_cause          = cause_lbl,
                top_cause_pct      = round(cause_pct,    1),
                sample_size        = n_flights,
            )

            with st.expander(f"✈  {route}  —  actual delay rate: "
                             f"{actual_rate:.1f}%  |  anomalous days: "
                             f"{int(arow['anomalous_days'])}", expanded=True):
                col_facts, col_brief = st.columns([1, 2])

                with col_facts:
                    st.markdown("**Route facts fed to Gemini**")
                    st.markdown(f"""
| Fact | Value |
|---|---|
| Expected risk | {facts.expected_risk_pct:.1f}% |
| Actual 30-day rate | {facts.actual_rate_pct:.1f}% |
| Primary cause | {facts.top_cause} |
| Cause share | {facts.top_cause_pct:.1f}% |
| Sample size | {facts.sample_size:,} flights |
""")

                with col_brief:
                    st.markdown("**Generated briefing**")
                    if not api_key_present:
                        st.info("Set GEMINI_API_KEY to generate this briefing.")
                    else:
                        cache_key = f"briefing_{route}"
                        if cache_key not in st.session_state:
                            with st.spinner(f"Calling Gemini for {route} …"):
                                try:
                                    st.session_state[cache_key] = \
                                        generate_route_briefing(facts)
                                except Exception as exc:
                                    st.session_state[cache_key] = \
                                        f"⚠ API error: {exc}"
                        st.markdown(st.session_state[cache_key])

        # ── Route-lookup search box ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Route Lookup")
        st.markdown(
            "Search any route in the dataset and generate an on-demand briefing."
        )

        all_routes = sorted(sample_feat["ROUTE"].unique())
        chosen = st.selectbox("Select a route", options=["— choose —"] + all_routes)

        if chosen != "— choose —":
            rs = sample_feat[sample_feat["ROUTE"] == chosen]
            actual_rate  = rs["Delayed"].mean() * 100
            expected_pct = rs["delay_prob"].mean() * 100 \
                           if "delay_prob" in rs.columns else actual_rate
            cause_lbl, cause_pct = top_cause_for_route(sample_feat, chosen)
            n_flights = len(rs)

            facts = RouteFacts(
                route              = chosen,
                expected_risk_pct  = round(expected_pct, 1),
                actual_rate_pct    = round(actual_rate,  1),
                top_cause          = cause_lbl,
                top_cause_pct      = round(cause_pct,    1),
                sample_size        = n_flights,
            )

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown("**Route facts**")
                st.markdown(f"""
| Fact | Value |
|---|---|
| Expected risk | {facts.expected_risk_pct:.1f}% |
| Actual 30-day rate | {facts.actual_rate_pct:.1f}% |
| Primary cause | {facts.top_cause} |
| Cause share | {facts.top_cause_pct:.1f}% |
| Sample size | {facts.sample_size:,} flights |
""")

            with col_b:
                st.markdown("**On-demand briefing**")
                if not api_key_present:
                    st.info("Set GEMINI_API_KEY to generate this briefing.")
                else:
                    if st.button("Generate briefing", key="lookup_btn"):
                        with st.spinner("Calling Gemini …"):
                            try:
                                result = generate_route_briefing(facts)
                                st.markdown(result)
                            except Exception as exc:
                                st.error(f"API error: {exc}")


if __name__ == "__main__":
    main()
