#!/usr/bin/env python3
"""
rename_eu_files_v2.py
Rename Refinitiv StreetEvents EU audio (.mp3) and transcript (.txt)
files to the pipeline convention: TICKER_Qn_YYYY.{mp3,txt}

Both file types are delivered by Refinitiv into a single raw directory.
This script routes them to separate output directories.

Refinitiv audio format:   MMDDYY_REFINITIVTICKER_NUMERICALID.mp3
Refinitiv transcript fmt: YYYY-Mon-DD-REFINITIVTICKER.EXCHANGE-ID-Transcript.txt

Design choices:
  - Calendar-quarter assignment (Q1: Jan-Mar, Q2: Apr-Jun,
    Q3: Jul-Sep, Q4: Oct-Dec) by call date, matching US pipeline.
  - Semi-annual reporters excluded in code (DGE, REL, BATS, ULVR, UL, RDS).
  - Duplicate files per firm-quarter resolved by keeping largest file by size.
  - CFR and RIO retained in processed/ (excluded at regression stage only).

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang 
# Advisor: Daniel Rigobon
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

base     = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
raw_dir  = base / "transcripts" / "europe" / "raw"   # Refinitiv delivers both types here
a_proc   = base / "audio"       / "europe" / "processed"
t_proc   = base / "transcripts" / "europe" / "processed"

a_proc.mkdir(parents=True, exist_ok=True)
t_proc.mkdir(parents=True, exist_ok=True)

sample = pd.read_csv(base / "european_adr_sample.csv")
VALID_NYSE = set(sample["ticker"].tolist())

# ── Refinitiv → NYSE ticker mapping ───────────────────────────────────────────
# Audio files use short Refinitiv tickers (no exchange suffix).
# Transcript files use TICKER.EXCHANGE format; suffix stripped before lookup.
REFINITIV_TO_NYSE = {
    # Firms where Refinitiv ticker differs from NYSE ticker
    "ABBN"    : "ABB",
    "INGA"    : "ING",
    "NOKIA"   : "NOK",
    "TTEF"    : "TTE",
    "PHIA"    : "PHG",
    "STMPA"   : "STM",
    "HSBA"    : "HSBC",
    "BBV"     : "BBVA",
    "SAN"     : "SAN",    # Banco Santander — keep as SAN
    "SASY"    : "SNY",    # Sanofi (alt Refinitiv code)
    "NOVN"    : "NVS",
    "SAPG"    : "SAP",
    "CFR"     : "CFR",    # Retained; excluded at regression stage
    "RIO"     : "RIO",    # Retained; excluded at regression stage
    "ERICB"   : "ERIC",
    "ENI"     : "E",
    # Firms where Refinitiv ticker = NYSE ticker (identity mapping)
    "AZN"     : "AZN",
    "BP"      : "BP",
    "GSK"     : "GSK",
    "SHEL"    : "SHEL",
    "SNY"     : "SNY",
    "SAP"     : "SAP",
    "ASML"    : "ASML",
    "NVS"     : "NVS",
    "ARGX"    : "ARGX",
    "BBVA"    : "BBVA",
    "TEF"     : "TEF",
    "ERIC"    : "ERIC",
    "E"       : "E",
    "RACE"    : "RACE",
    "STM"     : "STM",
    "EQNR"    : "EQNR",
    "NOK"     : "NOK",
    "ABB"     : "ABB",
    "ING"     : "ING",
    "PHG"     : "PHG",
    "TTE"     : "TTE",
}

# Semi-annual reporters: excluded from processing
SEMI_ANNUAL = {"DGE", "REL", "BATS", "ULVR", "UL", "RDS",
               "HSBA_OLD"}   # safety catch for any old naming

MONTH_MAP = {
    "Jan":1, "Feb":2, "Mar":3,  "Apr":4,  "May":5,  "Jun":6,
    "Jul":7, "Aug":8, "Sep":9,  "Oct":10, "Nov":11, "Dec":12,
}

def date_to_quarter(dt: datetime) -> str:
    m = dt.month
    if   m <= 3:  return f"Q1_{dt.year}"
    elif m <= 6:  return f"Q2_{dt.year}"
    elif m <= 9:  return f"Q3_{dt.year}"
    else:         return f"Q4_{dt.year}"

def strip_exchange_suffix(ref_ticker: str) -> str:
    """ABBN.S -> ABBN | INGA.AS -> INGA | AZN.L -> AZN"""
    return ref_ticker.split(".")[0].upper()

def parse_audio(fname: str):
    """
    MMDDYY_REFINITIVTICKER_NUMERICALID.mp3
    Returns (datetime, refinitiv_ticker) or (None, None).
    """
    m = re.match(r'^(\d{2})(\d{2})(\d{2})_([A-Za-z0-9]+)_\d+\.mp3$', fname)
    if not m:
        return None, None
    try:
        dt = datetime(int(m.group(3)) + 2000,
                      int(m.group(1)),
                      int(m.group(2)))
    except ValueError:
        return None, None
    return dt, m.group(4).upper()

def parse_transcript(fname: str):
    """
    YYYY-Mon-DD-REFINITIVTICKER.EXCHANGE-NUMERICALID-Transcript.txt
    Returns (datetime, refinitiv_ticker_base) or (None, None).
    """
    m = re.match(
        r'^(\d{4})-([A-Za-z]{3})-(\d{2})-([A-Za-z0-9.]+)-\d+-Transcript\.txt$',
        fname)
    if not m:
        return None, None
    mon = MONTH_MAP.get(m.group(2))
    if not mon:
        return None, None
    try:
        dt = datetime(int(m.group(1)), mon, int(m.group(3)))
    except ValueError:
        return None, None
    ref_ticker = strip_exchange_suffix(m.group(4))
    return dt, ref_ticker

# ── Main renaming loop ─────────────────────────────────────────────────────────
# Track (output_path -> list of (source_path, file_size)) for dup resolution
audio_candidates = {}   # dest_path -> [(src_path, size)]
trans_candidates = {}

ok, skipped_semi, skipped_no_map, skipped_not_valid = 0, 0, 0, 0

for fpath in sorted(raw_dir.iterdir()):
    fname = fpath.name

    if fname.endswith(".mp3"):
        dt, ref_ticker = parse_audio(fname)
        candidates_dict = audio_candidates
        out_dir = a_proc
        ext = ".mp3"
    elif fname.endswith(".txt"):
        dt, ref_ticker = parse_transcript(fname)
        candidates_dict = trans_candidates
        out_dir = t_proc
        ext = ".txt"
    else:
        continue

    if dt is None:
        skipped_no_map += 1
        continue

    # Check semi-annual exclusion (by Refinitiv ticker)
    if ref_ticker in SEMI_ANNUAL or ref_ticker.split(".")[0] in SEMI_ANNUAL:
        skipped_semi += 1
        continue

    # Map Refinitiv -> NYSE
    nyse_ticker = REFINITIV_TO_NYSE.get(ref_ticker)
    if nyse_ticker is None:
        print(f"  WARN: no NYSE mapping for Refinitiv ticker '{ref_ticker}' ({fname})")
        skipped_no_map += 1
        continue

    # Validate against sample (CFR and RIO are in sample, so they pass)
    if nyse_ticker not in VALID_NYSE:
        skipped_not_valid += 1
        continue

    quarter  = date_to_quarter(dt)
    dest     = out_dir / f"{nyse_ticker}_{quarter}{ext}"
    size     = fpath.stat().st_size

    if dest not in candidates_dict:
        candidates_dict[dest] = []
    candidates_dict[dest].append((fpath, size))

# Resolve duplicates: keep largest file
for candidates_dict, label in [(audio_candidates, "audio"),
                                 (trans_candidates, "transcript")]:
    for dest, sources in candidates_dict.items():
        sources.sort(key=lambda x: x[1], reverse=True)  # largest first
        chosen, chosen_size = sources[0]
        if len(sources) > 1:
            print(f"  DUP ({label}): {dest.name} — "
                  f"keeping {chosen.name} ({chosen_size:,} bytes), "
                  f"dropping {[s[0].name for s in sources[1:]]}")
        shutil.copy2(chosen, dest)
        ok += 1

print(f"\nRename complete:")
print(f"  Copied:              {ok}")
print(f"  Skipped (semi-ann):  {skipped_semi}")
print(f"  Skipped (no map):    {skipped_no_map}")
print(f"  Skipped (not valid): {skipped_not_valid}")
print(f"\nAudio processed:      "
      f"{len(list(a_proc.glob('*.mp3')))}")
print(f"Transcripts processed: "
      f"{len(list(t_proc.glob('*.txt')))}")
