#!/usr/bin/env python3
"""
14_extract_call_times_eu.py
Extract earnings call dates and times for EU ADR firms from
Refinitiv StreetEvents transcript files.

I use a two-pronged approach for cleaning the downloaded files. 
  1. Parse call DATE from the raw transcript filename, which encodes
     the exact call date: YYYY-Mon-DD-REFINITIVTICKER.EXCHANGE-ID-Transcript.txt
     e.g. 2022-Apr-21-ABBN.S-140169292426-Transcript.txt → Apr 21 2022
  2. Parse call TIME from the transcript header (same Refinitiv GMT
     format as US transcripts). If the header parse fails, time is
     recorded as NaN — this is acceptable because 09_download_taq_eu_adr_v2.py
     uses a FIXED open-reaction window (08:45–11:00 ET) and only needs
     the call DATE to identify the correct TAQ table, not the exact time.

All EU calls occur pre-market (07:00–09:00 ET); is_market_hours = 0
for all EU observations by construction.

Output: data/call_times_eu_extracted.csv
  Used by: 09_download_taq_eu_adr_v2.py

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang
# Advisor: Daniel Rigobon
"""

import os
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import pytz

base      = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
raw_dir   = base / "transcripts" / "europe" / "raw"
out_path  = base / "call_times_eu_extracted.csv"

if not raw_dir.exists():
    raise FileNotFoundError(f"EU raw transcripts not found: {raw_dir}")

txt_files = sorted(raw_dir.glob("*.txt"))
print(f"Found {len(txt_files)} EU transcript files")

# ── Refinitiv → NYSE ticker mapping (matches rename_eu_files_v2.py) ───────────
REFINITIV_TO_NYSE = {
    "ABBN":  "ABB",  "INGA":  "ING",  "NOKIA": "NOK",  "TTEF":  "TTE",
    "PHIA":  "PHG",  "STMPA": "STM",  "HSBA":  "HSBC", "BBV":   "BBVA",
    "SASY":  "SNY",  "NOVN":  "NVS",  "SAPG":  "SAP",  "CFR":   "CFR",
    "RIO":   "RIO",  "ERICB": "ERIC", "ENI":   "E",
    # Identity mappings
    "AZN":  "AZN",  "BP":    "BP",   "GSK":   "GSK",  "SHEL":  "SHEL",
    "SNY":  "SNY",  "SAP":   "SAP",  "ASML":  "ASML", "NVS":   "NVS",
    "ARGX": "ARGX", "BBVA":  "BBVA", "TEF":   "TEF",  "ERIC":  "ERIC",
    "E":    "E",    "RACE":  "RACE", "STM":   "STM",  "EQNR":  "EQNR",
    "NOK":  "NOK",  "ABB":   "ABB",  "ING":   "ING",  "PHG":   "PHG",
    "TTE":  "TTE",  "SAN":   "SAN",
}

SEMI_ANNUAL = {"DGE", "REL", "BATS", "ULVR", "UL", "RDS"}

MONTH_MAP = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12,
    # Header format (all-caps)
    "JANUARY":1,"FEBRUARY":2,"MARCH":3,"APRIL":4,"MAY":5,"JUNE":6,
    "JULY":7,"AUGUST":8,"SEPTEMBER":9,"OCTOBER":10,"NOVEMBER":11,"DECEMBER":12,
}

# Filename pattern: YYYY-Mon-DD-REFINITIVTICKER.EXCHANGE-ID-Transcript.txt
FNAME_PATTERN = re.compile(
    r'^(\d{4})-([A-Za-z]{3})-(\d{2})-([A-Za-z0-9.]+)-\d+-Transcript\.txt$'
)

# Header time pattern (same as US transcripts)
HEADER_PATTERN = re.compile(
    r'([A-Z]+)\s+(\d{1,2}),\s+(\d{4})\s*/\s*(\d{1,2}):(\d{2})(AM|PM)\s+GMT',
    re.IGNORECASE
)

eastern = pytz.timezone("US/Eastern")
gmt     = pytz.UTC

def date_to_quarter(dt: datetime) -> str:
    m = dt.month
    if   m <= 3:  return f"Q1_{dt.year}"
    elif m <= 6:  return f"Q2_{dt.year}"
    elif m <= 9:  return f"Q3_{dt.year}"
    else:         return f"Q4_{dt.year}"

def strip_exchange(ref_ticker: str) -> str:
    return ref_ticker.split(".")[0].upper()

def parse_header_time(content: str):
    """
    Try to extract GMT time from Refinitiv transcript header.
    Returns (hour_gmt, minute_gmt) or (None, None).
    """
    m = HEADER_PATTERN.search(content)
    if not m:
        return None, None
    hour   = int(m.group(4))
    minute = int(m.group(5))
    am_pm  = m.group(6).upper()
    if am_pm == "PM" and hour != 12:
        hour += 12
    elif am_pm == "AM" and hour == 12:
        hour = 0
    return hour, minute

results, errors = [], []

for fpath in txt_files:
    fname = fpath.name

    fm = FNAME_PATTERN.match(fname)
    if not fm:
        errors.append({"file": fname, "error": "Filename does not match pattern"})
        continue

    year_str   = fm.group(1)
    month_str  = fm.group(2)
    day_str    = fm.group(3)
    ref_ticker = strip_exchange(fm.group(4))

    # Skip semi-annual reporters
    if ref_ticker in SEMI_ANNUAL:
        continue

    # Map Refinitiv → NYSE
    nyse_ticker = REFINITIV_TO_NYSE.get(ref_ticker)
    if nyse_ticker is None:
        errors.append({"file": fname,
                       "error": f"No NYSE mapping for '{ref_ticker}'"})
        continue

    month = MONTH_MAP.get(month_str)
    if not month:
        errors.append({"file": fname, "error": f"Unknown month: {month_str}"})
        continue

    try:
        call_date = datetime(int(year_str), month, int(day_str))
    except ValueError as e:
        errors.append({"file": fname, "error": str(e)})
        continue

    quarter = date_to_quarter(call_date)

    # Pass 2: try to get exact time from header
    try:
        content   = fpath.read_text(encoding="utf-8", errors="ignore")[:5000]
        hr_gmt, mn_gmt = parse_header_time(content)
    except Exception:
        hr_gmt, mn_gmt = None, None

    if hr_gmt is not None:
        call_gmt = gmt.localize(
            datetime(call_date.year, call_date.month, call_date.day,
                     hr_gmt, mn_gmt))
        call_et      = call_gmt.astimezone(eastern)
        call_time_gmt = call_gmt.strftime("%H:%M:%S")
        call_time_et  = call_et.strftime("%H:%M:%S")
        call_dt_gmt   = call_gmt.strftime("%Y-%m-%d %H:%M:%S")
        call_dt_et    = call_et.strftime("%Y-%m-%d %H:%M:%S")
    else:
        call_time_gmt = None
        call_time_et  = None
        call_dt_gmt   = None
        call_dt_et    = None

    results.append({
        "ticker":            nyse_ticker,
        "quarter":           quarter,
        "call_date":         call_date.strftime("%Y-%m-%d"),
        "call_time_gmt":     call_time_gmt,
        "call_time_et":      call_time_et,
        "call_datetime_gmt": call_dt_gmt,
        "call_datetime_et":  call_dt_et,
        # All EU calls are pre-market by construction
        "is_market_hours":   0,
        "window_type":       "open_reaction",
    })

print(f"\nExtracted: {len(results)} EU firm-quarter call dates")
print(f"  With exact time: {sum(1 for r in results if r['call_time_gmt'])}")
print(f"  Date only:       {sum(1 for r in results if not r['call_time_gmt'])}")
print(f"Errors/skipped:  {len(errors)}")

if not results:
    print("ERROR")
    raise SystemExit(1)

df = (pd.DataFrame(results)
        .sort_values(["call_date", "ticker"])
        .drop_duplicates(subset=["ticker", "quarter"], keep="last")
        .reset_index(drop=True))

df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(df)} rows)")

print("\nQuarter distribution:")
print(df["quarter"].value_counts().sort_index().to_string())
print(f"\nFirms: {df['ticker'].nunique()}  —  "
      f"{sorted(df['ticker'].unique())}")

if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  {e['file']}: {e['error']}")
