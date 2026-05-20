"""
SCRIPT 02: ALIGNMENT + CLEANING
================================
Loads financial_data_raw.csv (full-calendar output of script 01)
and produces a clean, NYSE-aligned, forward-filled dataset.

Transformation applied IN THIS SCRIPT (LEVELS only, no differencing):
  • Log-levels  [ln(x)]  : price-like and volatility variables
                           SP500, SP5EFIN, DAX, XLF, DXY, EURUSD, VIX, VXO
  • Raw levels  [x]      : everything else (rates, spreads, indices already
                           in basis-point or % units)

This script does NOT apply stationarity transformations (differencing /
log-returns) — that is script 03. The output is suitable for inspection,
plotting, or as direct input to any transformation pipeline.

─────────────────────────────────────────────────────────────────────────────
OPERATIONS (in order)
─────────────────────────────────────────────────────────────────────────────
  1. Load master daily CSV (full calendar from script 01)
  2. Filter to NYSE trading days (remove weekends + all NYSE holidays)
  3. Forward-fill on the trading-day grid only
     (fills gaps where a FRED series wasn't published on a given trading day,
      e.g. FEDFUNDS missing on a specific Thursday — does NOT create
      artificial weekend/holiday values, those dates are already excluded)
  4. Apply ln(·) to LOG_LEVEL_VARS; all other variables kept in raw levels
  5. Integrity check: report remaining NaN (series starting later)
  6. Export aligned dataset

─────────────────────────────────────────────────────────────────────────────
OUTPUTS
─────────────────────────────────────────────────────────────────────────────
  financial_data_aligned.csv    NYSE trading days, forward-filled,
                                 log-levels for price/vol vars, raw levels
                                 for rate/spread vars
  financial_data_aligned.xlsx   same content in Excel format
  alignment_report.csv          per-variable coverage after alignment
  ffill_audit.csv               per-variable forward-fill audit (NaN before/after)

─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "financial_data_raw.csv"

# ── Variable lists (same as script 01) ─────────────────────────────────────
BASELINE_VARS = [
    "SP500",     # Equity — broad market
    "SP5EFIN",   # Equity — financial sector
    "VIX",       # Volatility / uncertainty
    "CSPREAD",   # Credit risk — Moody's Baa−Aaa spread
    "AAA",       # Credit level anchor — Moody's Seasoned Aaa yield
    "DGS10",     # Long-term interest rate
    "SLOPE",     # Term spread — T10Y3M (10Y CMT − 3M CMT)
    "DXY",       # US Dollar Index
    "EURUSD",    # EUR/USD
    "FEDFUNDS",  # Policy rate
]

ROBUSTNESS_VARS = [
    "BAA",           # Moody's Seasoned Baa yield
    "DAX",           # Global equity — Europe
    "XLF",           # Financial Select Sector ETF
    "DGS2",          # 2Y yield
    "TED",           # TED spread (discontinued Jan 2023)
    "ICE_AAA",       # ICE BofA AAA corporate yield
    "ICE_BBB",       # ICE BofA BBB corporate yield
    "ICE_CSPREAD",   # ICE BofA BBB−AAA
    "SLOPE_10Y2Y",   # DGS10−DGS2
    "RAI_BEX",       # Risk Aversion Index by BEX
    "VXO",           # CBOE S&P 100 Volatility Index
    "DGS1",          # 1Y yield
    "YCSLOPE",       # DGS10−DGS1 (Miranda-Agrippino & Rey 2020)
    "DTB3",          # 3M T-Bill rate
    "MM_DIFF",       # DTB3 − FEDFUNDS (flight-to-safety in money market)
]

ALL_VARS = BASELINE_VARS + ROBUSTNESS_VARS

# ── Variables to be stored in LOG-LEVELS: ln(x) ────────────────────────────
#    Rationale:
#      • Equity indices (SP500, SP5EFIN, DAX, XLF) and FX rates (DXY, EURUSD)
#        are non-negative price processes; their natural scale is logarithmic.
#        ln(P_t) is the standard representation in the finance/macro literature
#        (cf. Kilian & Lütkepohl 2017, Ch. 2; Canova 2007, Ch. 3).
#      • Volatility indices (VIX, VXO) are strictly positive by construction;
#        the log transformation compresses the heavy right tail and stabilises
#        variance (Blair et al. 2001; Giot 2005).
#      All other variables (interest rates, spreads, risk indices) are already
#      expressed in economically interpretable units (%, bps) and are kept
#      in raw levels — taking logs of rates near zero is numerically unstable
#      and economically meaningless.
LOG_LEVEL_VARS = [
    "SP500",    # Equity index — broad market
    "SP5EFIN",  # Equity index — financial sector
    "DAX",      # Equity index — Europe
    "XLF",      # Equity ETF — financial sector
    "DXY",      # USD Index (trade-weighted price)
    "EURUSD",   # EUR/USD spot rate
    "VIX",      # CBOE Volatility Index (strictly positive)
    "VXO",      # CBOE S&P 100 Volatility Index (strictly positive)
]

# ============================================================================
# STEP 1 — LOAD MASTER DAILY (full calendar from script 01)
# ============================================================================

print("=" * 80)
print("SCRIPT 02 — ALIGNMENT + CLEANING")
print("=" * 80)
print(f"Input: {INPUT_FILE}\n")

print("=" * 80)
print("STEP 1 — LOAD MASTER DAILY (full calendar)")
print("=" * 80)

data = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
print(f"  Loaded: {data.shape[0]:,} calendar days × {data.shape[1]} variables")
print(f"  Range:  {data.index.min().date()} → {data.index.max().date()}")

available = [v for v in ALL_VARS if v in data.columns]
missing_v = [v for v in ALL_VARS if v not in data.columns]
if missing_v:
    print(f"\n  [!] Variables not found in master (check script 01): {missing_v}")
data = data[available].copy()
print(f"  Using {len(available)} of {len(ALL_VARS)} expected variables")

# ============================================================================
# STEP 2 — FILTER TO NYSE TRADING DAYS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2 — FILTER TO NYSE TRADING DAYS")
print("=" * 80)

nyse     = mcal.get_calendar("NYSE")
schedule = nyse.schedule(start_date=data.index.min(), end_date=data.index.max())
trading_days = schedule.index.tz_localize(None)

print(f"  Calendar days in master:      {data.shape[0]:,}")
print(f"  NYSE trading days in range:   {len(trading_days):,}")
print(f"  Weekends/holidays removed:    {data.shape[0] - len(trading_days):,}")

data.index = pd.to_datetime(data.index).tz_localize(None).normalize()

# Reindex to NYSE trading days ONLY
# (no full-calendar ffill — avoids propagating data onto weekends/holidays
#  before filtering, which is cleaner and avoids any potential bias)
data_aligned = data.reindex(trading_days)

# ============================================================================
# STEP 2b — SAMPLE CUT  [1990-01-01, 2026-01-01)
# ============================================================================
data_aligned = data_aligned.loc["1990-01-01":"2025-12-31"]
print(f"  Sample cut: {data_aligned.index.min().date()} → {data_aligned.index.max().date()}  "
      f"({data_aligned.shape[0]:,} trading days)")

# ============================================================================
# STEP 3 — FORWARD-FILL ON TRADING-DAY GRID
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3 — FORWARD-FILL ON TRADING-DAY GRID")
print("=" * 80)

# ── Per-variable NaN audit BEFORE ffill ────────────────────────────────────
nan_before_total = data_aligned.isna().sum().sum()
nan_before_var   = data_aligned.isna().sum()  # Series: var → count

# Apply forward-fill
data_aligned = data_aligned.ffill()
data_aligned.index.name = "Date"

# ── Per-variable NaN audit AFTER ffill ─────────────────────────────────────
nan_after_total = data_aligned.isna().sum().sum()
nan_after_var   = data_aligned.isna().sum()

print(f"\n  Aggregate:  {nan_before_total:,} NaN before  →  "
      f"{nan_after_total:,} NaN after  →  {nan_before_total - nan_after_total:,} cells filled")

# ── Detailed per-variable report ───────────────────────────────────────────
print(f"\n  {'Variable':<14} {'NaN_pre':>8} {'NaN_post':>9} {'Filled':>7} {'Reason'}")
print("  " + "─" * 70)

ffill_rows = []
for v in available:
    pre  = int(nan_before_var[v])
    post = int(nan_after_var[v])
    filled = pre - post

    # Classify the reason
    if pre == 0:
        reason = "no gaps — published every NYSE trading day"
    elif filled > 0 and post == 0:
        reason = f"all {filled} gaps filled (source skipped some trading days)"
    elif filled > 0 and post > 0:
        reason = (f"{filled} gaps filled + {post} residual NaN "
                  f"(series starts later than sample)")
    else:
        reason = f"{post} NaN — series starts later, no prior value to propagate"

    flag = "✓" if post == 0 else ("○" if post < 100 else "⚠")
    print(f"  {v:<14} {pre:>8,} {post:>9,} {filled:>7,}  {flag} {reason}")

    ffill_rows.append({
        "Variable":     v,
        "NaN_Before":   pre,
        "NaN_After":    post,
        "Cells_Filled": filled,
        "Reason":       reason,
    })

# Save ffill audit log
ffill_df = pd.DataFrame(ffill_rows)
ffill_df.to_csv("ffill_audit.csv", index=False)
print(f"\n  ✓ ffill_audit.csv — per-variable forward-fill audit saved")

print(f"\n  Result: {data_aligned.shape[0]:,} NYSE trading days × "
      f"{data_aligned.shape[1]} variables (pre-transformation)")

# ============================================================================
# STEP 4 — APPLY LOG-LEVELS TO PRICE/VOLATILITY VARIABLES
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4 — APPLY ln(·) TO PRICE/VOLATILITY VARIABLES")
print("=" * 80)
print("""
  Transformation rule:
    ln(x_t)   for price-like and volatility variables  → LOG_LEVEL_VARS
    x_t       for all other variables                  → raw levels
""")

log_transformed = []   # variables actually transformed in this run
log_skipped     = []   # LOG_LEVEL_VARS not present in the dataset

for v in LOG_LEVEL_VARS:
    if v not in data_aligned.columns:
        log_skipped.append(v)
        continue

    col = data_aligned[v]

    # Safety check: log is undefined for non-positive values
    n_nonpos = (col <= 0).sum()
    if n_nonpos > 0:
        raise ValueError(
            f"  [ERROR] {v} has {n_nonpos} non-positive value(s) — "
            f"ln(·) cannot be applied. Check the raw data."
        )

    data_aligned[v] = np.log(col)
    log_transformed.append(v)

# ── Report ──────────────────────────────────────────────────────────────────
print(f"  {'Variable':<14} {'Transform':<18} {'Note'}")
print("  " + "─" * 60)
for v in available:
    if v in log_transformed:
        note = "ln(x_t)"
        tag  = "LOG-LEVEL"
    else:
        note = "x_t"
        tag  = "RAW LEVEL"
    print(f"  {v:<14} {tag:<18} {note}")

if log_skipped:
    print(f"\n  [!] LOG_LEVEL_VARS not found in dataset (skipped): {log_skipped}")

print(f"\n  Transformed : {len(log_transformed)} variables → log-levels")
print(f"  Kept as-is  : {len(available) - len(log_transformed)} variables → raw levels")

# ============================================================================
# STEP 5 — INTEGRITY CHECK + COVERAGE REPORT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5 — INTEGRITY CHECK")
print("=" * 80)

# Report remaining NaN (series not yet available at sample start)
nan_report = data_aligned.isna().sum()
nan_vars   = nan_report[nan_report > 0]

if len(nan_vars) > 0:
    print("\n  Remaining NaN after ffill (series starting later than sample):")
    for v, n in nan_vars.items():
        first_valid = data_aligned[v].first_valid_index()
        fv_str = first_valid.date() if first_valid is not None else "N/A"
        print(f"    {v:<14s}: {n:,} NaN  ({n/len(data_aligned)*100:.1f}%)  "
              f"— first valid: {fv_str}")
else:
    print("  No remaining NaN after ffill. ✓")

# Check baseline completeness
baseline_cols = [v for v in BASELINE_VARS if v in data_aligned.columns]
baseline_nan  = data_aligned[baseline_cols].isna().sum()
baseline_nan  = baseline_nan[baseline_nan > 0]
if len(baseline_nan) == 0:
    print("  Baseline variables: fully complete ✓")
else:
    print("\n  [!] Baseline variables with NaN:")
    for v, n in baseline_nan.items():
        print(f"    {v}: {n:,} NaN")

# Build alignment report
report_rows = []
for v in available:
    col = data_aligned[v]
    n_valid = col.notna().sum()
    n_nan   = col.isna().sum()
    first   = col.first_valid_index()
    last    = col.last_valid_index()
    role    = "BASELINE" if v in BASELINE_VARS else "ROBUSTNESS"
    scale   = "log-level" if v in log_transformed else "raw level"
    report_rows.append({
        "Variable":    v,
        "Role":        role,
        "Scale":       scale,
        "N_Valid":     int(n_valid),
        "N_NaN":       int(n_nan),
        "NaN_Pct":     round(n_nan / len(data_aligned) * 100, 2),
        "First_Valid":  first.date() if first is not None else None,
        "Last_Valid":   last.date() if last is not None else None,
    })

report_df = pd.DataFrame(report_rows)

# ============================================================================
# STEP 6 — SAVE OUTPUTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6 — SAVE OUTPUTS")
print("=" * 80)

data_aligned.to_csv("financial_data_aligned.csv")
print("✓ financial_data_aligned.csv")
data_aligned.to_excel("financial_data_aligned.xlsx", engine="openpyxl")
print("✓ financial_data_aligned.xlsx")
report_df.to_csv("alignment_report.csv", index=False)
print("✓ alignment_report.csv")

# ============================================================================
# QUALITY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("DATA QUALITY REPORT — ALIGNED (NYSE TRADING DAYS)")
print("=" * 80)
print(f"\n  Aligned dataset:  {data_aligned.index.min().date()} → "
      f"{data_aligned.index.max().date()}  "
      f"({data_aligned.shape[0]:,} NYSE trading days × {data_aligned.shape[1]} vars)")

hdr = (f"  {'Var':<14} {'Role':<12} {'Scale':<12} {'N_valid':>7}  "
       f"{'NaN%':>6}  {'First':<12} {'Last':<12}")
print(hdr)
print("  " + "─" * (len(hdr) - 2))
for _, row in report_df.iterrows():
    flag = "✓" if row["NaN_Pct"] == 0 else ("○" if row["NaN_Pct"] < 5 else "⚠")
    print(f"  {row['Variable']:<14} {row['Role']:<12} {row['Scale']:<12} "
          f"{row['N_Valid']:>7,}  "
          f"{row['NaN_Pct']:>5.1f}%  "
          f"{str(row['First_Valid']):<12} {str(row['Last_Valid']):<12}  {flag}")

print("\n" + "=" * 80)
print("SCRIPT 02 COMPLETE — DATA ALIGNED TO NYSE TRADING DAYS")
print("=" * 80)
print("""
  Output files:
    → financial_data_aligned.csv    (NYSE trading days, ffilled)
    → financial_data_aligned.xlsx   (same, Excel format)
    → alignment_report.csv          (per-variable coverage + scale info)
    → ffill_audit.csv               (per-variable forward-fill audit)

  Variable scales in output:
    LOG-LEVEL  [ln(x_t)] : SP500, SP5EFIN, DAX, XLF, DXY, EURUSD, VIX, VXO
    RAW LEVEL  [x_t]     : CSPREAD, AAA, DGS10, SLOPE, FEDFUNDS, BAA,
                            DGS2, TED, ICE_AAA, ICE_BBB, ICE_CSPREAD,
                            SLOPE_10Y2Y, RAI_BEX, DGS1, YCSLOPE, DTB3

  Script 03 will:
    1. Load financial_data_aligned.csv
    2. Apply stationarity transformations:
         log-returns Δln(x_t) : SP500, SP5EFIN, DAX, XLF, DXY, EURUSD
                                 VIX, VXO  (simple first-diff of log-levels)
         first diff  Δx_t     : DGS10, SLOPE, FEDFUNDS, CSPREAD, AAA, ...
    3. ADF / KPSS unit-root tests
    4. Export var_ready_baseline.csv + var_ready_full.csv
""")