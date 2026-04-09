#!/usr/bin/env python3
"""
09_download_taq_eu_adr_v2.py
Download NYSE TAQ millisecond data for EU ADR firms.

Because all EU earnings calls occur pre-market (07:00-09:00 ET),
I use an open-reaction window anchored to NYSE open (09:30 ET)
rather than a call-synchronized window. This script downloads
a fixed daily window: 08:45-11:00 ET on the call date,
covering the pre-window (09:00-09:30) and event window (09:30-10:30)
plus a buffer.

Output: data/taq/eu/{TICKER}_{QUARTER}_taq.parquet

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang 
# Advisor: Daniel Rigobon
"""

import wrds
import pandas as pd
import numpy as np
import os
from pathlib import Path

data_dir = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
out_dir  = data_dir / "taq" / "eu"
out_dir.mkdir(parents=True, exist_ok=True)

db = wrds.Connection()

sample     = pd.read_csv(data_dir / "european_adr_sample.csv")
call_times = pd.read_csv(data_dir / "call_times_eu_extracted.csv")
# Expected columns: ticker, quarter, call_date, call_time_et (or call_time_gmt)

calls = sample.merge(call_times, on="ticker", how="left")

# Fixed open-reaction window bounds (ET)
# Pre-window:   09:00-09:30 ET
# Event window: 09:30-10:30 ET
# Download with buffer: 08:45-11:00 ET
WINDOW_START_ET = "08:45:00"
WINDOW_END_ET   = "11:00:00"

# EU calls are pre-market: all start before 09:30 ET.
# I use the call DATE only to identify the correct TAQ table;
# the time window is fixed to market-open reaction.

skipped, downloaded, errored = 0, 0, 0

for _, row in calls.iterrows():
    ticker  = row["ticker"]
    quarter = row["quarter"]

    out_path = out_dir / f"{ticker}_{quarter}_taq.parquet"
    if out_path.exists():
        print(f"  SKIP {ticker} {quarter} (exists)")
        skipped += 1
        continue

    # Determine the date for the TAQ table
    call_date = pd.Timestamp(row["call_date"])
    date_str  = call_date.strftime("%Y%m%d")

    try:
        trades = db.raw_sql(f"""
            SELECT date,
                   time_m   AS time,
                   sym_root AS ticker,
                   price,
                   size,
                   tr_corr,
                   tr_scond
            FROM taqmsec.ctm_{date_str}
            WHERE sym_root = '{ticker}'
              AND time_m BETWEEN '{WINDOW_START_ET}' AND '{WINDOW_END_ET}'
        """)

        quotes = db.raw_sql(f"""
            SELECT date,
                   time_m   AS time,
                   sym_root AS ticker,
                   bid,
                   ofr      AS ask,
                   bidsiz,
                   ofrsiz   AS asksiz,
                   qu_cond
            FROM taqmsec.cqm_{date_str}
            WHERE sym_root = '{ticker}'
              AND time_m BETWEEN '{WINDOW_START_ET}' AND '{WINDOW_END_ET}'
        """)

        trades["type"] = "trade"
        quotes["type"] = "quote"

        # Align columns for concat
        all_cols = sorted(set(trades.columns) | set(quotes.columns))
        combined = pd.concat(
            [trades.reindex(columns=all_cols),
             quotes.reindex(columns=all_cols)],
            ignore_index=True
        )

        combined.to_parquet(out_path, index=False)
        print(f"  OK  {ticker} {quarter}  "
              f"trades={len(trades):,}  quotes={len(quotes):,}")
        downloaded += 1

    except Exception as e:
        print(f"  ERR {ticker} {quarter}: {e}")
        errored += 1

print(f"\nDone — downloaded: {downloaded}  skipped: {skipped}  errors: {errored}")
db.close()
