"""
SCRIPT 04: MONTHLY MACRO VARIABLES  (FRED)
===========================================
Downloads monthly US macroeconomic variables from FRED for the second-stage
analysis: BVAR with the financial tail-risk shock (identified in the daily VAR)
as exogenous variable.

─────────────────────────────────────────────────────────────────────────────
VARIABLE SET — 10 monthly macro variables  (matching Miranda-Agrippino & Rey
               + FEDFUNDS for monetary-policy control)
─────────────────────────────────────────────────────────────────────────────

  Channel           Variable    FRED Code    Notes
  ──────────────── ──────────  ───────────  ──────────────────────────────────
  Real activity     INDPRO      INDPRO       Industrial Production Index (SA)
  Capacity          CAPUTIL     TCU          Capacity Utilization: Total (SA)
  Labour market     UNRATE      UNRATE       Civilian Unemployment Rate (SA)
  Employment        PAYEMS      PAYEMS       Total Nonfarm Payrolls (SA)
  Housing           HOUST       HOUST        Housing Starts: Total (SAAR)
  House prices      CSHPI       CSUSHPISA    Case-Shiller US National HPI (SA)
  Prices (CPI)      CPI         CPIAUCSL     CPI All Items (SA)
  Prices (PCE)      PCEPI       PCEPI        PCE Price Index (SA)
  Credit            NONREVSL    NONREVSL     Nonrevolving Consumer Credit (SA)
  Monetary policy   FFR         FEDFUNDS     Fed Funds Effective Rate (NSA, pct)

  All series are seasonally adjusted by the source (BLS / Census / BEA),
  except FEDFUNDS (not seasonally adjusted — standard for policy rates).
  All cover 1990–2026 on FRED without gaps.

  NOTE ON FFR: including the policy rate is necessary to avoid omitted-variable
  bias. Without it, any Fed response to a financial shock is absorbed into the
  VAR residuals, threatening the exogeneity of the external instrument Z_m
  (especially for the VIX-shock identification).

─────────────────────────────────────────────────────────────────────────────
TRANSFORMATIONS  (I(1) levels — BVAR convention)
─────────────────────────────────────────────────────────────────────────────
  Log levels  :  INDPRO, PAYEMS, HOUST, CSHPI, CPI, PCEPI, NRCREDIT
                 (indices / monetary amounts → log stabilises variance)
  Level       :  UNRATE, CAPUTIL, FFR  (already in pct pts, no transformation)

  Series are kept in I(1) form: the Minnesota prior in the BVAR shrinks
  toward unit-root (random-walk) behaviour, so stationarity is not required.
  Differencing is intentionally NOT applied.

─────────────────────────────────────────────────────────────────────────────
OUTPUTS
─────────────────────────────────────────────────────────────────────────────
  macro_monthly_raw.csv           raw levels, full sample 1990–2026
  macro_monthly_ready.csv         I(1) levels (log or raw), ready for MATLAB
  macro_unit_root_tests.csv       ADF + KPSS on raw and I(1) series
  macro_transformation_log.csv    which transformation was applied

"""

import os
import time
import pandas as pd
import numpy as np
from fredapi import Fred
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

# ── Runtime paths (run from any cwd) ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "04_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = "1990-01-01"
END_DATE   = "2026-01-01"

FRED_API_KEY = os.getenv("FRED_API_KEY", "fd44963c4c6cc9df732bb14fdfde790b")
fred = Fred(api_key=FRED_API_KEY)

# ── Variable definitions ─────────────────────────────────────────────────────
# (fred_code, variable_name, description, transformation)
#
# Transformations (I(1)-preserving — no differencing):
#   "log"   → ln(Xt)   natural log level   [indices, monetary amounts]
#   "level" → Xt       raw level           [rates already in pct pts]

VARIABLES = [
    # ── Real activity & capacity ──────────────────────────────────────────────
    ("INDPRO",    "INDPRO",   "Industrial Production Index (SA, 2017=100)",           "log"),
    ("TCU",       "CAPUTIL",  "Capacity Utilization: Total Industry (SA, pct)",       "level"),
    # ── Labour market ─────────────────────────────────────────────────────────
    ("UNRATE",    "UNRATE",   "Civilian Unemployment Rate (SA, pct)",                 "level"),
    ("PAYEMS",    "PAYEMS",   "Total Nonfarm Payrolls (SA, thousands)",               "log"),
    # ── Housing ───────────────────────────────────────────────────────────────
    ("HOUST",     "HOUST",    "Housing Starts: Total New (SAAR, thousands)",           "log"),
    ("CSUSHPISA", "CSHPI",    "S&P/Case-Shiller US National Home Price Index (SA)",   "log"),
    # ── Prices ────────────────────────────────────────────────────────────────
    ("CPIAUCSL",  "CPI",      "CPI All Items (SA, 1982-84=100)",                      "log"),
    ("PCEPI",     "PCEPI",    "PCE Price Index (SA, 2017=100)",                       "log"),
    # ── Credit channel (MA&R extended VAR, Fig. D.3) ─────────────────────────
    ("NONREVSL",  "NRCREDIT", "Nonrevolving Consumer Credit (SA, mil.$)",             "log"),
    # ── Monetary policy (omitted-variable control) ────────────────────────────
    ("FEDFUNDS",  "FFR",      "Federal Funds Effective Rate (NSA, pct)",              "level"),
]

# ── External (non-FRED) financial-market variables ────────────────────────────
# These three variables provide the financially-proximate channels for the
# three Kollo BVAR structural shocks used as Proxy-SVAR instruments

EXTERNAL_VARS = [
    # BEX Risk Aversion Index — dimensionless, spikes during equity stress
    ("BEX_Indices_Formatted.xlsx", "RAI_BEX",
     "BEX Risk Aversion Index — Bekaert, Engstrom & Xu (2022) [level]",
     "level", "Risk Aversion"),
    # BEX Uncertainty (annualised vol %) — option-implied uncertainty measure
    # Preferred over raw "Uncertainty" column (which is a variance-like unit);
    # "Annual Vol %" is in natural % units, comparable in scale to VIX/100.
    ("BEX_Indices_Formatted.xlsx", "UNC_BEX",
     "BEX Uncertainty Index, annualised vol % — Bekaert, Engstrom & Xu (2022) [level]",
     "level", "Uncertainty (Annual Vol %)"),
    # GZ Excess Bond Premium — corporate spread net of predicted default risk
    # Already in percentage-point units; mean-reverting and stationary (I(0))
    ("ebp_csv.csv",                "EBP",
     "GZ Excess Bond Premium — Gilchrist & Zakrajsek (2012), pct pts [level]",
     "level", "ebp"),
    # GZ Total Spread (gz_spread = EBP + predicted default premium).
    # Includes both the risk-premium and the default-risk components.
    ("ebp_csv.csv",                "GZ_SPR",
     "GZ Total Credit Spread — Gilchrist & Zakrajsek (2012), pct pts [level]",
     "level", "gz_spread"),
]

# Unit root test parameters (same as script 02)
ADF_MAXLAGS = 12          # monthly data → higher max lag than daily
KPSS_LAGS   = "auto"
ALPHA       = 0.05

print("=" * 80)
print("SCRIPT 04 — MONTHLY MACRO VARIABLES  (FRED + EXTERNAL)")
print("=" * 80)
print(f"Target range: {START_DATE} → {END_DATE}\n")

# ============================================================================
# HELPERS
# ============================================================================

def download_fred(series_id: str, name: str,
                  pause: float = 0.6, retries: int = 3) -> pd.Series:
    """Download one series from FRED with retry logic for rate-limit errors."""
    for attempt in range(1, retries + 1):
        try:
            print(f"  FRED  {name:<10s} ({series_id}) ... ", end="", flush=True)
            s = fred.get_series(series_id,
                                observation_start=START_DATE,
                                observation_end=END_DATE)
            s = s.dropna().sort_index()
            s.name = name
            print(f"✓  {len(s):,} obs  [{s.index[0].date()} → {s.index[-1].date()}]")
            time.sleep(pause)          # be polite to the FRED API
            return s
        except Exception as e:
            msg = str(e)
            print(f"✗  {msg}", end="")
            if "Too Many Requests" in msg and attempt < retries:
                wait = pause * 5 * attempt   # back-off: 3s, 6s …
                print(f"  → waiting {wait:.0f}s then retrying ({attempt}/{retries-1})")
                time.sleep(wait)
            else:
                print()
                return pd.Series(dtype=float, name=name)
    return pd.Series(dtype=float, name=name)


def apply_transform(series: pd.Series, method: str, name: str = "") -> pd.Series:
    """Apply I(1)-preserving transformation (no differencing)."""
    s = series.copy()
    if method == "log":
        # log is undefined for non-positive values — fail loudly instead of
        # silently clipping (which would inject ln(1e-10) ≈ -23 outliers)
        n_nonpos = int((s.dropna() <= 0).sum())
        if n_nonpos > 0:
            raise ValueError(
                f"{name or 'series'}: {n_nonpos} non-positive value(s) — "
                f"ln(·) cannot be applied. Check the raw data."
            )
        return np.log(s)
    elif method == "level":
        return s
    else:
        raise ValueError(f"Unknown transform: {method}")


def run_adf(series: pd.Series) -> dict:
    """ADF test. H0: unit root. Reject → stationary."""
    s = series.dropna()
    if len(s) < 20:
        return {"ADF_stat": np.nan, "ADF_pval": np.nan, "ADF_lags": np.nan,
                "ADF_crit5": np.nan, "ADF_reject": np.nan}
    try:
        stat, pval, lags, _, crit, _ = adfuller(s, maxlag=ADF_MAXLAGS, autolag="BIC")
        return {
            "ADF_stat":   round(stat, 4),
            "ADF_pval":   round(pval, 6),
            "ADF_lags":   lags,
            "ADF_crit5":  round(crit["5%"], 4),
            "ADF_reject": pval < ALPHA,
        }
    except Exception:
        return {"ADF_stat": np.nan, "ADF_pval": np.nan, "ADF_lags": np.nan,
                "ADF_crit5": np.nan, "ADF_reject": np.nan}


def run_kpss(series: pd.Series) -> dict:
    """KPSS test. H0: stationary. Reject → non-stationary."""
    s = series.dropna()
    if len(s) < 20:
        return {"KPSS_stat": np.nan, "KPSS_pval": np.nan, "KPSS_crit5": np.nan,
                "KPSS_reject": np.nan}
    try:
        stat, pval, _, crit = kpss(s, regression="c", nlags=KPSS_LAGS)
        return {
            "KPSS_stat":   round(stat, 4),
            "KPSS_pval":   round(pval, 4),
            "KPSS_crit5":  round(crit["5%"], 4),
            "KPSS_reject": pval < ALPHA,
        }
    except Exception:
        return {"KPSS_stat": np.nan, "KPSS_pval": np.nan, "KPSS_crit5": np.nan,
                "KPSS_reject": np.nan}


def stationarity_verdict(adf: dict, kpss_r: dict) -> str:
    """
    STATIONARY    : ADF rejects unit root  AND  KPSS does NOT reject stationarity
    NON-STATIONARY: ADF does not reject    AND  KPSS rejects stationarity
    AMBIGUOUS     : tests disagree
    """
    adf_ok  = adf.get("ADF_reject", False)
    kpss_ok = not kpss_r.get("KPSS_reject", True)
    if adf_ok and kpss_ok:
        return "STATIONARY"
    elif not adf_ok and not kpss_ok:
        return "NON-STATIONARY"
    else:
        return "AMBIGUOUS"


# ============================================================================
# STEP 1 — DOWNLOAD FROM FRED
# ============================================================================

print("─" * 80)
print("STEP 1 — DOWNLOAD")
print("─" * 80)

raw_series = {}
for fred_code, var_name, description, _ in VARIABLES:
    s = download_fred(fred_code, var_name)
    if len(s) > 0:
        raw_series[var_name] = s

# Merge into DataFrame
data_raw = pd.concat(raw_series.values(), axis=1).sort_index()

# Ensure monthly frequency (end-of-month alignment, standard for FRED monthly)
# FRED monthly series are dated on the 1st of the month;
# we keep that convention for consistency
data_raw.index = pd.to_datetime(data_raw.index)
data_raw.index.name = "Date"

print(f"\n✓ Raw merge: {data_raw.shape[0]:,} months × {data_raw.shape[1]} variables")
print(f"  Range: {data_raw.index.min().date()} → {data_raw.index.max().date()}")

# ============================================================================
# STEP 1b — EXTERNAL SERIES  (non-FRED)
# BEX Risk Aversion (Bekaert et al. 2022) and GZ Excess Bond Premium (2012)
# provide the financial-market channels needed for the monthly Proxy SVAR.
# ============================================================================

print("\n" + "─" * 80)
print("STEP 1b — EXTERNAL SERIES  (BEX Risk Aversion  |  GZ Excess Bond Premium)")
print("─" * 80)

for fname, var_name, desc, transform, col in EXTERNAL_VARS:
    fpath = os.path.join(SCRIPT_DIR, fname)
    print(f"  FILE  {var_name:<10s} ({fname}) ... ", end="")
    try:
        if fname.endswith(".xlsx"):
            df_ext = pd.read_excel(fpath, sheet_name="Monthly Data")
            df_ext["Date"] = pd.to_datetime(df_ext["Date"], format="%m/%Y")
            s_ext = df_ext.set_index("Date")[col].rename(var_name).sort_index()
        elif fname.endswith(".csv"):
            df_ext = pd.read_csv(fpath, parse_dates=["date"])
            s_ext = df_ext.set_index("date")[col].rename(var_name).sort_index()
        else:
            raise ValueError(f"Unsupported format: {fname}")
        # Align index to first-of-month (FRED convention)
        # to_timestamp() with no args = start of period = 1st of month
        s_ext.index = s_ext.index.to_period("M").to_timestamp()
        s_ext = s_ext[(s_ext.index >= START_DATE) & (s_ext.index <= END_DATE)].dropna()
        data_raw[var_name] = s_ext
        print(f"✓  {len(s_ext):,} obs  [{s_ext.index[0].date()} → {s_ext.index[-1].date()}]")
    except Exception as e:
        print(f"✗  ERROR: {e}")

# ── Guard: the monthly Proxy-SVAR instruments depend on the external series ──
missing_ext = [v for _, v, _, _, _ in EXTERNAL_VARS if v not in data_raw.columns]
if missing_ext:
    print(f"\n  [!] WARNING — external series failed to load: {missing_ext}")
    print("      These are the Proxy-SVAR instruments; check the source files "
          "before using the BVAR output.")

print(f"\n✓ Full raw merge: {data_raw.shape[0]:,} months × {data_raw.shape[1]} variables")

# Save raw (includes FRED + external)
data_raw.to_csv(os.path.join(OUTPUT_DIR, "macro_monthly_raw.csv"))
print("✓ macro_monthly_raw.csv")

# ============================================================================
# STEP 2 — UNIT ROOT TESTS ON RAW (LEVEL) SERIES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2 — UNIT ROOT TESTS ON RAW (LEVEL) SERIES")
print("=" * 80)
print(f"\n  {'Variable':<10} {'ADF stat':>10} {'ADF p':>10} {'KPSS stat':>10} {'KPSS p':>8}  Verdict")
print("  " + "─" * 62)

fred_names = [name for _, name, _, _ in VARIABLES]
ext_names  = [v   for _, v, _, _, _ in EXTERNAL_VARS]
available  = [v for v in fred_names + ext_names if v in data_raw.columns]

raw_results = []
for var in available:
    adf = run_adf(data_raw[var])
    kps = run_kpss(data_raw[var])
    vrd = stationarity_verdict(adf, kps)
    flag = "✓" if vrd == "STATIONARY" else ("?" if vrd == "AMBIGUOUS" else "✗")
    print(f"  {var:<10} {adf['ADF_stat']:>10.4f} {adf['ADF_pval']:>10.6f} "
          f"{kps['KPSS_stat']:>10.4f} {kps['KPSS_pval']:>8.4f}  {flag} {vrd}")
    raw_results.append({"Variable": var, "Stage": "RAW", "Transform": "level",
                        "Verdict": vrd, **adf, **kps})

# ============================================================================
# STEP 3 — APPLY TRANSFORMATIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3 — APPLY I(1)-PRESERVING TRANSFORMATIONS  (no differencing)")
print("=" * 80)

desc_map = {
    "log":   "natural log of level  (I(1) preserved)",
    "level": "raw level  (no transformation)",
}

transform_map = {name: tr for _, name, _, tr in VARIABLES}
# Add external variable transformations (all "level" — already usable as-is)
for _, var_name, _, transform, _ in EXTERNAL_VARS:
    transform_map[var_name] = transform

data_transf = pd.DataFrame(index=data_raw.index)
transf_log  = []

for var in available:
    method = transform_map[var]
    s_tr = apply_transform(data_raw[var], method, var)
    data_transf[var] = s_tr
    print(f"  {var:<10s} → {desc_map[method]}")
    transf_log.append({"Variable": var, "Transformation": method,
                        "Description": desc_map[method]})

# No row dropped: log/level transformations do not introduce leading NaN

# ============================================================================
# STEP 4 — UNIT ROOT TESTS ON TRANSFORMED SERIES
# ============================================================================

# Expected order of integration:
#   macro variables    → I(1)  (kept in levels; Minnesota prior handles the unit root)
#   financial proxies  → I(0)  (risk premia / spreads are mean-reverting by construction)
I0_EXPECTED = set(ext_names)

print("\n" + "=" * 80)
print("STEP 4 — UNIT ROOT TESTS ON TRANSFORMED SERIES")
print("=" * 80)
print("\n  Expected:  macro variables → NON-STATIONARY I(1)   |   "
      "financial proxies → STATIONARY I(0)")
print(f"\n  {'Variable':<10} {'Transform':<10} {'Exp':<5} {'ADF stat':>10} {'ADF p':>10} "
      f"{'KPSS stat':>10} {'KPSS p':>8}  Verdict")
print("  " + "─" * 77)

transf_results = []
for var in available:
    method   = transform_map[var]
    expected = "I(0)" if var in I0_EXPECTED else "I(1)"
    adf = run_adf(data_transf[var])
    kps = run_kpss(data_transf[var])
    vrd = stationarity_verdict(adf, kps)
    matches = ((expected == "I(1)" and vrd != "STATIONARY") or
               (expected == "I(0)" and vrd != "NON-STATIONARY"))
    flag = "✓" if matches else "!"
    print(f"  {var:<10} {method:<10} {expected:<5} {adf['ADF_stat']:>10.4f} {adf['ADF_pval']:>10.6f} "
          f"{kps['KPSS_stat']:>10.4f} {kps['KPSS_pval']:>8.4f}  {flag} {vrd}")
    transf_results.append({"Variable": var, "Stage": "TRANSFORMED",
                            "Transform": method, "Expected": expected,
                            "Verdict": vrd, **adf, **kps})

mismatches = [
    r["Variable"] for r in transf_results
    if (r["Expected"] == "I(1)" and r["Verdict"] == "STATIONARY")
    or (r["Expected"] == "I(0)" and r["Verdict"] == "NON-STATIONARY")
]
if mismatches:
    print(f"\n  [!] Order of integration differs from expectation (check): {mismatches}")
else:
    print("\n  All series match the expected order "
          "(macro I(1), financial proxies I(0)). ✓")

# ============================================================================
# STEP 5 — FINAL EXPORT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5 — FINAL EXPORT")
print("=" * 80)

data_transf = data_transf.dropna()
data_transf.index.name = "Date"

data_transf.to_csv(os.path.join(OUTPUT_DIR, "macro_monthly_ready.csv"))
print(f"✓ macro_monthly_ready.csv     →  {data_transf.shape[0]:,} obs × {data_transf.shape[1]} vars")
print(f"  Sample: {data_transf.index.min().date()} → {data_transf.index.max().date()}")

# Unit root test table
ur_df = pd.DataFrame(raw_results + transf_results)
ur_df.to_csv(os.path.join(OUTPUT_DIR, "macro_unit_root_tests.csv"), index=False)
print(f"✓ macro_unit_root_tests.csv   →  {len(ur_df)} rows")

# Transformation log
tl_df = pd.DataFrame(transf_log)
tl_df.to_csv(os.path.join(OUTPUT_DIR, "macro_transformation_log.csv"), index=False)
print(f"✓ macro_transformation_log.csv→  {len(tl_df)} variables")

# ============================================================================
# STEP 6 — SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
  Sample            : {data_transf.index.min().date()} → {data_transf.index.max().date()}
  Monthly obs       : {data_transf.shape[0]:,}
  Variables         : {data_transf.shape[1]}

  Transformations applied:""")
for entry in transf_log:
    print(f"    {entry['Variable']:<10s}  {entry['Description']}")

print(f"""
  Files:
    → macro_monthly_raw.csv            (raw levels)
    → macro_monthly_ready.csv          (I(1) levels: log or raw, no differencing)
    → macro_unit_root_tests.csv        (ADF + KPSS results)
    → macro_transformation_log.csv     (transformation documentation)

  MATLAB import:
    T     = readtable('macro_monthly_ready.csv');
    dates = datetime(T.Date, 'InputFormat', 'yyyy-MM-dd');
    Y     = T{{:, 2:end}};        % {data_transf.shape[0]:,} × {data_transf.shape[1]} matrix
    vars  = T.Properties.VariableNames(2:end);
""")
print("=" * 80)
print("SCRIPT 04 COMPLETE")
print("=" * 80)
