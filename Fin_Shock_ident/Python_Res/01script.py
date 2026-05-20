"""
SCRIPT 01: DATA DOWNLOAD  (FRED + LSEG EXCEL)
=============================================
Downloads daily financial variables for the thesis VAR model.

Sources: FRED (online) and LSEG Excel workbook only.

─────────────────────────────────────────────────────────────────────────────
FINAL VARIABLE SET
─────────────────────────────────────────────────────────────────────────────
BASELINE VAR — 10 variables

  Channel           Variable   Source    Notes
  ──────────────── ─────────  ────────  ──────────────────────────────────
  Equity (broad)   SP500      LSEG      S&P 500 index level
  Equity (fin.)    SP5EFIN    LSEG      S&P 500 Financial Sector index
  Volatility       VIX        FRED      CBOE VIX
  Credit risk      CSPREAD    FRED      Moody's Baa − Aaa yield (computed)
                   AAA        FRED      Moody's Seasoned Aaa Corporate Bond Yield (DAAA)
  Interest rates   DGS10      FRED      10-year Treasury yield
                   SLOPE      FRED      T10Y3M — 10Y CMT minus 3M CMT (direct from FRED)
  FX / Dollar      DXY        LSEG      US Dollar Index
                   EURUSD     LSEG      EUR/USD spot rate
  Money market     FEDFUNDS   FRED      Fed Funds rate (daily DFF)
  NOTE: DTB3 is in ROBUSTNESS — quasi-collinear with SLOPE (≈DGS10−DTB3) and FEDFUNDS

ROBUSTNESS — 15 variables (kept in master, excluded from baseline VAR)

  Variable      Source    Notes
  ───────────   ────────  ──────────────────────────────────────────────────
  BAA           FRED      Moody's Seasoned Baa yield (= AAA + CSPREAD, redundant in VAR)
  DAX           LSEG      German equity index (global dimension)
  XLF           LSEG      Financial Select Sector ETF (from 1998-12-22)
  DGS2          FRED      2-year Treasury (component of old SLOPE)
  TED           FRED      TED spread — DISCONTINUED Jan 2023; use 1990-2022 subsample only
  ICE_AAA       FRED      ICE BofA AAA corporate yield (BAMLC0A1CAAA)
  ICE_BBB       FRED      ICE BofA BBB corporate yield (BAMLC0A4CBBB)
  ICE_CSPREAD   FRED      ICE BofA BBB − AAA (computed)
  SLOPE_10Y2Y   FRED      DGS10 − DGS2 (old term spread, computed)
  RAI_BEX       LSEG      Risk Aversion Index by BEX (daily)
  VXO           LSEG      CBOE S&P 100 Volatility Index (VXO)
  DGS1          FRED      1-year Treasury constant maturity yield
  YCSLOPE       FRED      DGS10 − DGS1 (yield curve slope, Miranda-Agrippino & Rey 2020)
  DTB3          FRED      3-month T-Bill rate (moved from baseline: collinear with SLOPE+FEDFUNDS)      
  MM_DIFF       FRED     DTB3 − FEDFUNDS (flight-to-safety in money market, computed)
  ─────────────────────────────────────────────────────────────────────────────
SOURCE RULES
─────────────────────────────────────────────────────────────────────────────
  LSEG preferred  : SP500, SP5EFIN, DAX, DXY, EURUSD, XLF, RAI_BEX, VXO
  FRED fallback   : EURUSD → DEXUSEU (from 1999)
                    VXO    → VXOCLS (from 1986)
                    SP500, SP5EFIN, DAX, XLF, RAI_BEX have NO free fallback.
                    If their LSEG sheet is missing, the script warns and
                    leaves those columns as NaN.

─────────────────────────────────────────────────────────────────────────────
OUTPUTS
─────────────────────────────────────────────────────────────────────────────
  financial_data_raw.csv    raw merge on original (irregular) date index
  financial_data_raw.xlsx   same content in Excel format
  data_metadata.csv         coverage stats per variable
─────────────────────────────────────────────────────────────────────────────
NOTE ON SAMPLE
  The raw file preserves the full history from all sources.
  Sample restriction (1990–2026) is applied in script 02.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
from fredapi import Fred
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = "1990-01-02"
END_DATE   = "2026-01-02"

FRED_API_KEY = os.getenv("FRED_API_KEY", "fd44963c4c6cc9df732bb14fdfde790b")
fred = Fred(api_key=FRED_API_KEY)

LSEG_FILE = "LSEGdata1.xlsx"

# Map: LSEG sheet name → variable name in final dataset
# ──────────────────────────────────────────────────────
# Adjust sheet names below if LSEG workbook uses different labels
LSEG_SHEET_MAP = {
    "S&P500 daily":     "SP500",
    "S&P FinSect":      "SP5EFIN",
    "DAX":              "DAX",
    "DXY":              "DXY",
    "EURUSD":           "EURUSD",
    "XLF":              "XLF",
    "RAI BEX daily":    "RAI_BEX",   # Risk Aversion Index by BEX
    "VXO":              "VXO",       # CBOE S&P 100 Volatility Index
}

print("=" * 80)
print("SCRIPT 01 — DATA DOWNLOAD  (FRED + LSEG EXCEL)")
print("=" * 80)
print(f"Target range: {START_DATE} → {END_DATE}\n")

# ============================================================================
# HELPERS
# ============================================================================

def download_fred(series_id: str, name: str, start: str = START_DATE) -> pd.Series:
    """Download one series from FRED."""
    try:
        print(f"  FRED  {name:<14s} ({series_id}) ... ", end="")
        s = fred.get_series(series_id, observation_start=start, observation_end=END_DATE)
        s = s.dropna().sort_index()
        s.name = name
        print(f"✓  {len(s):,} obs  [{s.index[0].date()} → {s.index[-1].date()}]")
        return s
    except Exception as e:
        print(f"✗  ERROR: {e}")
        return pd.Series(dtype=float, name=name)


def read_lseg_sheet(path: str, sheet: str, name: str) -> pd.Series:
    """
    Read one sheet from the LSEG Excel workbook.

    Expected layout:
      Rows 1-5 : metadata  (Start / End / Frequency / Name / Code)
      After the row where col A == 'Code':
        col A : dates  (dd/mm/yy  or  dd/mm/yyyy)
        col B : values (may use comma as decimal separator)
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)

    # Locate the 'Code' header row
    col0 = df.iloc[:, 0].astype(str).str.strip().str.lower()
    code_rows = df.index[col0 == "code"].tolist()
    if not code_rows:
        raise ValueError(f"Sheet '{sheet}': row with 'Code' in column A not found.")
    data_start = code_rows[0] + 1

    dates = pd.to_datetime(df.iloc[data_start:, 0], dayfirst=True, errors="coerce")
    vals  = (
        df.iloc[data_start:, 1]
        .astype(str)
        .str.replace(r"\s", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    vals = pd.to_numeric(vals, errors="coerce")

    s = pd.Series(vals.values, index=dates, name=name, dtype="float64")
    s = s[~s.index.isna()].dropna().sort_index()
    return s


def load_lseg_workbook(path: str, sheet_map: dict) -> dict:
    """Load all mapped sheets from the LSEG workbook into a dict of Series."""
    out = {}
    if not os.path.exists(path):
        print(f"\n  [!] LSEG workbook not found: '{path}' — all LSEG series will be NaN.\n")
        return out

    print("\nLoading LSEG Excel workbook ...")
    xls       = pd.ExcelFile(path)
    available = set(xls.sheet_names)

    for sheet, name in sheet_map.items():
        if sheet not in available:
            print(f"  ○  Sheet '{sheet}' not found in workbook — {name} skipped.")
            continue
        try:
            s = read_lseg_sheet(path, sheet, name)
            out[name] = s
            print(f"  ✓  {name:<10s} from '{sheet}':  "
                  f"{s.index.min().date()} → {s.index.max().date()}  ({len(s):,} obs)")
        except Exception as e:
            print(f"  ✗  '{sheet}' → {name}:  {e}")
    return out


def pick_series(name: str, lseg_key: str, lseg: dict,
                fallback: pd.Series = None) -> pd.Series:
    """
    Return the LSEG series if available.
    Fall back to a FRED series with a warning if LSEG is missing.
    If neither is available, return an empty Series with a warning.
    """
    if lseg_key in lseg and len(lseg[lseg_key]) > 0:
        s = lseg[lseg_key].rename(name)
        print(f"  ✓  {name:<14s} → LSEG  ({len(s):,} obs)")
        return s
    if fallback is not None and len(fallback) > 0:
        print(f"  [!] {name:<14s} LSEG missing → FRED fallback  ({len(fallback):,} obs)")
        return fallback.rename(name)
    print(f"  [!] {name:<14s} LSEG missing, no fallback → column will be all-NaN.")
    return pd.Series(dtype=float, name=name)


# ============================================================================
# STEP 0 — LOAD LSEG EXCEL
# ============================================================================

lseg = load_lseg_workbook(LSEG_FILE, LSEG_SHEET_MAP)

# ============================================================================
# STEP 1 — FRED DOWNLOADS
# ============================================================================

print("\n" + "=" * 80)
print("FRED DOWNLOADS")
print("=" * 80)

# ── Volatility ──────────────────────────────────────────────────────────────
vix = download_fred("VIXCLS", "VIX")

# ── Interest rates ──────────────────────────────────────────────────────────
dgs10 = download_fred("DGS10", "DGS10")
dgs2  = download_fred("DGS2",  "DGS2")
dgs1  = download_fred("DGS1",  "DGS1")
dtb3  = download_fred("DTB3",  "DTB3")

# ── Baseline term spread: T10Y3M (10Y CMT minus 3M CMT, directly from FRED)
t10y3m = download_fred("T10Y3M", "SLOPE")

# ── OLD term spread → robustness
slope_10y2y = (dgs10 - dgs2).dropna().rename("SLOPE_10Y2Y")
print(f"  Computed  SLOPE_10Y2Y (10Y−2Y): {len(slope_10y2y):,} obs  [robustness]")

# ── yield-curve slope following Miranda-Agrippino & Rey (2020) → robustness
ycslope = (dgs10 - dgs1).dropna().rename("YCSLOPE")
print(f"  Computed  YCSLOPE (DGS10−DGS1, Miranda-Agrippino & Rey): {len(ycslope):,} obs  [robustness]")

# ── Credit spreads — NEW baseline: Moody's (available from 1986) ────────────
moody_baa = download_fred("DBAA", "BAA")          # Moody's Seasoned Baa
moody_aaa = download_fred("DAAA", "AAA")          # Moody's Seasoned Aaa
cspread   = (moody_baa - moody_aaa).dropna().rename("CSPREAD")
print(f"  Computed  CSPREAD (Moody's Baa−Aaa): {len(cspread):,} obs  [baseline]")

# ── Credit spreads — OLD: ICE BofA → robustness (available from end-1996) ──
ice_bbb     = download_fred("BAMLC0A4CBBB", "ICE_BBB", start="1996-12-31")
ice_aaa     = download_fred("BAMLC0A1CAAA", "ICE_AAA", start="1996-12-31")
ice_cspread = (ice_bbb - ice_aaa).dropna().rename("ICE_CSPREAD")
print(f"  Computed  ICE_CSPREAD (ICE BBB−AAA): {len(ice_cspread):,} obs  [robustness]")

# ── Federal Funds Effective Rate (Daily DFF) ────────────────────────────────
fedfunds = download_fred("DFF", "FEDFUNDS", start="1990-01-01").sort_index()
print(
    f"Loaded FEDFUNDS (DFF daily): {len(fedfunds)} obs "
    f"[{fedfunds.index[0].date()} → {fedfunds.index[-1].date()}]"
)

# ── TED spread — DISCONTINUED January 2023 ─────────────────────────────────
# Kept in master for robustness on 1990-2022 subsample only
ted = download_fred("TEDRATE", "TED", start="1986-01-02")

# ── MM_DIFF = DTB3 − FEDFUNDS  (flight-to-safety in money market) ───────────
mm_diff = (dtb3 - fedfunds).dropna().rename("MM_DIFF")
print(f"  Computed  MM_DIFF (DTB3−FEDFUNDS): {len(mm_diff):,} obs  [robustness]")

# ── FRED fallbacks for LSEG series ─────────────────────────────────────────
# EURUSD: starts 1999-01-04
eurusd_fred = download_fred("DEXUSEU", "EURUSD_FRED", start="1999-01-04")
# VXO: CBOE S&P 100 Volatility Index (from 1986)
vxo_fred    = download_fred("VXOCLS", "VXO_FRED", start="1986-01-02")

# ============================================================================
# STEP 2 — APPLY SOURCE RULES
# ============================================================================

print("\n" + "=" * 80)
print("SOURCE ASSIGNMENT  (LSEG preferred, FRED fallback where defined)")
print("=" * 80)

SP500   = pick_series("SP500",   "SP500",   lseg, fallback=None)
SP5EFIN = pick_series("SP5EFIN", "SP5EFIN", lseg, fallback=None)
DAX     = pick_series("DAX",     "DAX",     lseg, fallback=None)
DXY     = pick_series("DXY",     "DXY",     lseg, fallback=None)
EURUSD  = pick_series("EURUSD",  "EURUSD",  lseg, fallback=eurusd_fred)
XLF     = pick_series("XLF",     "XLF",     lseg, fallback=None)
RAI_BEX = pick_series("RAI_BEX", "RAI_BEX", lseg, fallback=None)
VXO     = pick_series("VXO",     "VXO",     lseg, fallback=vxo_fred)

# Always FRED — baseline
VIX      = vix.rename("VIX")
DGS10    = dgs10.rename("DGS10")
DTB3     = dtb3.rename("DTB3")
SLOPE    = t10y3m.rename("SLOPE")              # T10Y3M
AAA      = moody_aaa.rename("AAA")             # Moody's Aaa
BAA      = moody_baa.rename("BAA")             # Moody's Baa
CSPREAD  = cspread.rename("CSPREAD")           # Moody's Baa − Aaa
FEDFUNDS = fedfunds.rename("FEDFUNDS")

# Always FRED — robustness
DGS2         = dgs2.rename("DGS2")
DGS1         = dgs1.rename("DGS1")
TED          = ted.rename("TED")
ICE_AAA      = ice_aaa.rename("ICE_AAA")
ICE_BBB      = ice_bbb.rename("ICE_BBB")
ICE_CSPREAD  = ice_cspread.rename("ICE_CSPREAD")
SLOPE_10Y2Y  = slope_10y2y.rename("SLOPE_10Y2Y")
YCSLOPE      = ycslope.rename("YCSLOPE")

# ============================================================================
# STEP 3 — MERGE INTO MASTER DATASET
# ============================================================================

print("\n" + "=" * 80)
print("MERGING")
print("=" * 80)

# Baseline first (11), robustness after (11)
series_dict = {
    # ── BASELINE (10 vars for VAR) ───────────────────────────────────────────
    "SP500":       SP500,       # Equity — broad market
    "SP5EFIN":     SP5EFIN,     # Equity — financial sector
    "VIX":         VIX,         # Volatility / uncertainty
    "CSPREAD":     CSPREAD,     # Credit risk — Moody's Baa−Aaa
    "AAA":         AAA,         # Moody's Seasoned Aaa yield (credit level anchor)
    "DGS10":       DGS10,       # Long-term interest rate
    "SLOPE":       SLOPE,       # Term spread — T10Y3M (10Y CMT − 3M CMT)
    "DXY":         DXY,         # US Dollar Index
    "EURUSD":      EURUSD,      # EUR/USD exchange rate
    "FEDFUNDS":    FEDFUNDS,    # Monetary policy rate
    # ── ROBUSTNESS (14 vars) ─────────────────────────────────────────────────
    "BAA":         BAA,         # Moody's Seasoned Baa yield (= AAA + CSPREAD)
    "DAX":         DAX,         # Global equity — Europe
    "XLF":         XLF,         # Financial ETF (from 1998-12-22)
    "DGS2":        DGS2,        # 2Y yield (component of old SLOPE)
    "TED":         TED,         # Money market stress (discontinued 2023)
    "ICE_AAA":     ICE_AAA,     # ICE BofA AAA corporate yield
    "ICE_BBB":     ICE_BBB,     # ICE BofA BBB corporate yield
    "ICE_CSPREAD": ICE_CSPREAD, # ICE BofA BBB−AAA (old baseline spread)
    "SLOPE_10Y2Y": SLOPE_10Y2Y, # DGS10−DGS2 (old baseline term spread)
    "RAI_BEX":     RAI_BEX,     # Risk Aversion Index by BEX
    "VXO":         VXO,         # CBOE S&P 100 Volatility Index
    "DGS1":        DGS1,        # 1Y yield (component of YCSLOPE)
    "YCSLOPE":     YCSLOPE,     # DGS10−DGS1 (yield curve slope, Miranda-Agrippino & Rey 2020)
    "DTB3":        DTB3,        # 3M T-Bill (robustness: collinear with SLOPE+FEDFUNDS in baseline)
    "MM_DIFF":     mm_diff,     # DTB3 − FEDFUNDS (flight-to-safety in money market, computed)
}

N_BASELINE   = 10
N_ROBUSTNESS = 15

data_raw = pd.concat(series_dict.values(), axis=1).sort_index()
print(f"✓ Raw merge:    {data_raw.shape[0]:,} rows × {data_raw.shape[1]} variables")
print(f"  Date range:   {data_raw.index.min().date()} → {data_raw.index.max().date()}")

# ============================================================================
# STEP 4 — SAVE OUTPUTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING")
print("=" * 80)

data_raw.to_csv("financial_data_raw.csv")
print("✓ financial_data_raw.csv")
data_raw.to_excel("financial_data_raw.xlsx", engine="openpyxl")
print("✓ financial_data_raw.xlsx")

_roles   = ["BASELINE"] * N_BASELINE + ["ROBUSTNESS"] * N_ROBUSTNESS
_sources = [
    # Baseline (10): SP500, SP5EFIN, VIX, CSPREAD, AAA, DGS10, SLOPE, DXY, EURUSD, FEDFUNDS
    "LSEG", "LSEG", "FRED", "FRED", "FRED",
    "FRED", "FRED", "LSEG", "LSEG", "FRED",
    # Robustness (15): BAA, DAX, XLF, DGS2, TED, ICE_AAA, ICE_BBB, ICE_CSPREAD,
    #                  SLOPE_10Y2Y, RAI_BEX, VXO, DGS1, YCSLOPE, DTB3, MM_DIFF
    "FRED", "LSEG", "LSEG", "FRED", "FRED",
    "FRED", "FRED", "FRED", "FRED",
    "Online", "LSEG",
    "FRED", "FRED",  # DGS1, YCSLOPE (computed)
    "FRED", "FRED",           # DTB3, MM_DIFF (computed)
]
metadata = pd.DataFrame({
    "Variable":    list(series_dict.keys()),
    "Role":        _roles,
    "Source":      _sources,
    "First_Date":  [
        data_raw[c].dropna().index.min().date()
        if data_raw[c].notna().any() else None
        for c in series_dict
    ],
    "Last_Date":   [
        data_raw[c].dropna().index.max().date()
        if data_raw[c].notna().any() else None
        for c in series_dict
    ],
    "N_Obs":       [int(data_raw[c].notna().sum()) for c in series_dict],
    "Missing_Pct": [round(data_raw[c].isna().mean() * 100, 2) for c in series_dict],
})
metadata.to_csv("data_metadata.csv", index=False)
print("✓ data_metadata.csv")

# ============================================================================
# STEP 5 — QUALITY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("DATA QUALITY REPORT — RAW MERGE")
print("=" * 80)
print(f"\n  Raw dataset:  {data_raw.index.min().date()} → "
      f"{data_raw.index.max().date()}  "
      f"({data_raw.shape[0]:,} rows × {data_raw.shape[1]} vars)")

hdr = f"  {'Var':<14} {'Role':<12} {'Src':<5}  {'N_obs':>7}  {'Miss%':>6}  {'Start':<12} {'End':<12}"
print(hdr)
print("  " + "─" * (len(hdr) - 2))
for _, row in metadata.iterrows():
    flag = "✓" if row["Missing_Pct"] < 35 else ("○" if row["Missing_Pct"] < 55 else "⚠")
    print(f"  {row['Variable']:<14} {row['Role']:<12} {row['Source']:<5}  "
          f"{row['N_Obs']:>7,}  {row['Missing_Pct']:>5.1f}%  "
          f"{str(row['First_Date']):<12} {str(row['Last_Date']):<12}  {flag}")

print("\n" + "=" * 80)
print("SCRIPT 01 COMPLETE — DATA DOWNLOADED AND MERGED")
print("=" * 80)
print("""
  Output files:
    → financial_data_raw.csv    (raw merge, irregular index, in levels)
    → financial_data_raw.xlsx   (same, Excel format)
    → data_metadata.csv         (coverage statistics)

  Script 02 will:
    1. Load financial_data_raw.csv
    2. Restrict to sample window (1990–2026)
    3. Filter to NYSE trading days
    4. Forward-fill gaps on the trading-day grid
    5. Export financial_data_aligned.csv (still in levels, ready for script 03)
""")