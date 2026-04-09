#!/usr/bin/env python3
"""
27_sync_all.py
Lee-Ready signed order imbalance with DST-correct ET timestamps.

TIMEZONE BUG:
  2023 call timestamps were stored in UTC in source data but applied
  as ET, shifting all event windows +4h. Fix: 14_extract_call_times.py
  now uses pytz for DST-aware UTC→ET conversion and stores the result
  in call_datetime_et. This script reads call_datetime_et directly,
  avoiding the hardcoded offset that was incorrect for winter
  calls (Q4/Q1, when offset should be -5h for EST not -4h for EDT).

  Pre-fix OI std ~0.50; post-fix ~0.07-0.10, consistent.

Output: data/synchronized/full_synchronized.csv

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang, Princeton Senior Thesis
# Advisor: Daniel Rigobon
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

data_dir = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
taq_dir  = data_dir / "taq"
out_dir  = data_dir / "synchronized"
out_dir.mkdir(exist_ok=True)

sample     = pd.read_csv(data_dir / "selected_sample_40_FINAL.csv")
call_times = pd.read_csv(data_dir / "call_times_extracted.csv")
calls      = sample.merge(call_times, on=['ticker'], how='left')

PRE_WIN  = 30   # minutes before call
CALL_WIN = 60   # minutes during call

def load_taq(ticker, quarter):
    for ext in ['.csv.gz', '.parquet']:
        p = taq_dir / f"{ticker}_{quarter}_taq{ext}"
        if not p.exists():
            continue
        df = (pd.read_csv(p, compression='gzip') if ext == '.csv.gz'
              else pd.read_parquet(p))
        df['timestamp'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str),
            format='mixed', errors='coerce')
        return df[df['type'] == 'trade'].copy(), df[df['type'] == 'quote'].copy()
    return None, None

def sign_trades(trades, quotes):
    if trades.empty or quotes.empty:
        return trades.assign(sign=np.nan)

    quotes = quotes.sort_values('timestamp')
    quotes['midpoint'] = (quotes['bid'] + quotes['ofr']) / 2
    quotes = quotes[['timestamp', 'midpoint']].dropna()

    trades = trades.sort_values('timestamp')
    trades = pd.merge_asof(trades, quotes, on='timestamp',
                           direction='backward',
                           tolerance=pd.Timedelta('5s'))

    trades['sign'] = np.where(trades['price'] > trades['midpoint'],  1,
                     np.where(trades['price'] < trades['midpoint'], -1, np.nan))

    # Tick rule for midpoint ties
    tick = np.sign(trades['price'] - trades['price'].shift(1))
    trades['sign'] = np.where(trades['sign'].isna(), tick, trades['sign'])
    return trades

def order_imbalance(signed_trades):
    valid = signed_trades.dropna(subset=['sign'])
    if len(valid) == 0:
        return np.nan, 0
    return float(valid['sign'].sum() / len(valid)), len(valid)

all_results = []

for _, row in calls.iterrows():
    ticker  = row['ticker']
    quarter = row['quarter']

    # Use call_datetime_et directly (DST-aware, set by 14_extract_call_times.py)
    if pd.notna(row.get('call_datetime_et')):
        call_dt = pd.Timestamp(row['call_datetime_et'])
    else:
        # Fallback: reconstruct from GMT with hardcoded offset (EDT only)
        # Flag a warning so this is visible in logs
        print(f"  WARN {ticker} {quarter}: call_datetime_et missing, "
              f"falling back to GMT-4h (may be incorrect for winter calls)")
        call_dt = (pd.Timestamp(f"{row['call_date']} {row['call_time_gmt']}")
                   - pd.Timedelta(hours=4))

    trades, quotes = load_taq(ticker, quarter)
    if trades is None:
        print(f"  MISS {ticker} {quarter}: no TAQ file")
        continue

    trades = sign_trades(trades, quotes)

    pre_mask = ((trades['timestamp'] >= call_dt - pd.Timedelta(minutes=PRE_WIN)) &
                (trades['timestamp'] <  call_dt))
    dur_mask = ((trades['timestamp'] >= call_dt) &
                (trades['timestamp'] <  call_dt + pd.Timedelta(minutes=CALL_WIN)))

    oi_pre, n_pre = order_imbalance(trades[pre_mask])
    oi_dur, n_dur = order_imbalance(trades[dur_mask])

    oi_shift  = (float(oi_dur) - float(oi_pre)
                 if not (np.isnan(oi_pre) or np.isnan(oi_dur)) else np.nan)

    call_hour = call_dt.hour + call_dt.minute / 60
    all_results.append({
        'ticker':                  ticker,
        'quarter':                 quarter,
        'call_datetime_et':        str(call_dt),
        'call_hour':               call_hour,
        'is_market_hours':         int(9.5 <= call_hour < 16.0),
        'pre_30m_order_imbalance': oi_pre,
        'pre_30m_n_trades':        n_pre,
        'during_order_imbalance':  oi_dur,
        'during_n_trades':         n_dur,
        'oi_shift':                oi_shift,
    })

sync_df = pd.DataFrame(all_results)
sync_df.to_csv(out_dir / "full_synchronized.csv", index=False)
print(f"\nSynchronized {len(sync_df)} calls | "
      f"valid OI shift: {sync_df['oi_shift'].notna().sum()}")
print(f"OI shift std: {sync_df['oi_shift'].std():.4f}  "
      f"(expect ~0.07-0.10 post timezone fix)")
