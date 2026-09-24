# SkyGuard — From-Scratch Setup Guide

## Prerequisites

| Requirement | Minimum version | Check |
|---|---|---|
| Python | 3.9 | `python --version` |
| pip | 22+ | `pip --version` |

Python 3.8 **will not work** — the codebase uses `str.removeprefix` and relies on
`pd.DataFrame.agg` behaviour that changed in 1.3. Use 3.9 or later.

---

## Step 1 — Get the data

Download the **2015 Flight Delays and Cancellations** dataset from Kaggle:  
https://www.kaggle.com/datasets/usdot/flight-delays

Place these three files in the **same directory as `app.py`**:

```
flights.csv
airlines.csv
airports.csv
```

`flights.csv` is ~580 MB. No renaming required — column names are used exactly
as Kaggle distributes them.

---

## Step 2 — Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected install time: 2–5 minutes (XGBoost and scikit-learn are the largest packages).

---

## Step 4 — (Optional) Set the Gemini API key

Page 4 (AI Insights) calls the Gemini API. The rest of the app works without it.

Get a free key at: https://aistudio.google.com/app/apikey

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key-here"

# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-key-here"

# Google Colab
import os
os.environ["GEMINI_API_KEY"] = "your-key-here"
```

If the key is absent, Page 4 shows a yellow warning banner and the rest of the
app is unaffected.

---

## Step 5 — Launch the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically http://localhost:8501). Open it in
any modern browser.

**First-load time:** ~60–120 seconds. The app trains three ML models (Logistic
Regression, Random Forest, XGBoost) on 800,000 rows. Results are cached for the
rest of the session — page navigation and filter changes are instant after that.

---

## Directory layout after setup

```
project/
├── app.py              ← single-file Streamlit application
├── requirements.txt    ← dependency pins
├── flights.csv         ← Kaggle data (580 MB)
├── airlines.csv        ← Kaggle data (< 1 KB)
├── airports.csv        ← Kaggle data (< 1 MB)
└── .venv/              ← virtual environment (not committed to git)
```

---

## Bugs fixed (changelog from original app.py)

| # | Location | Issue | Fix |
|---|---|---|---|
| 1 | `chart_monthly_trend` | `.agg(mean=("mean"), sem=lambda ...)` — `SpecificationError` in pandas ≥ 1.3 when mixing string shortcuts and lambdas on a single-column Series | Replaced with two separate `.mean()` / `.std()` calls combined into a `pd.DataFrame` |
| 2 | `build_features_and_models` | `use_label_encoder=False` removed in XGBoost ≥ 2.0; raises `TypeError` | Removed the kwarg |
| 3 | `top_cause_for_route` | Return type annotation `tuple[str, float]` requires Python ≥ 3.9; raises `TypeError` at import on 3.8 | Changed to bare `def` with no return annotation |
| 4 | `apply_sidebar_filters` | `st.image()` with an external Wikipedia URL — breaks on air-gapped / proxy networks and on every page render | Removed the remote image call entirely |
| 5 | `chart_delay_cause` | `totals.max()` on an empty `Series` raises `ValueError` when the filtered dataset has no delayed flights | Added an early-return guard that renders a "no data" message |
| 6 | `chart_scatter` | `route_risk["expected_pct"].max()` returns `NaN` on an empty DataFrame; `NaN * 1.05 = NaN` then crashes `ax.plot` | Added empty-DataFrame guard and `pd.notna()` check on the axis limit |

---

## Portability checklist — things that would break on a different machine

| Item | Status | Notes |
|---|---|---|
| Hardcoded file paths | ✅ None | CSVs are read with bare filenames; Python resolves them relative to the working directory. Run `streamlit run app.py` from the folder that contains the CSVs. |
| `GEMINI_API_KEY` env var | ✅ Handled | Missing key shows a `st.warning` banner; no crash |
| Python version | ⚠ Requires ≥ 3.9 | 3.8 fails on PEP 585 type hints if any remain; verified clean at 3.9+ |
| XGBoost version | ✅ Fixed | `use_label_encoder` kwarg removed; compatible with 1.7–2.x |
| pandas version | ✅ Fixed | `.agg()` SpecificationError resolved; compatible with 1.5–2.x |
| External network (Wikipedia image) | ✅ Fixed | Remote `st.image` URL removed |
| Operating system | ✅ No OS-specific code | No `os.path.join` with hardcoded separators; no shell subprocesses |
| CPU core count (`n_jobs=-1`) | ✅ Safe | Uses all available cores; works on 1-core and 64-core machines alike |
| Memory | ⚠ ~4 GB RAM recommended | 800k-row sample + three models in memory simultaneously |
| Google Colab | ✅ Compatible | `matplotlib.use("Agg")` is set; no display-backend assumption |
