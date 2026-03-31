
#!/usr/bin/env python3

"""

27_sync_all.py / 38_sync_2023.py

Lee-Ready signed OI with UTC->ET correction for 2023 calls.



TIMEZONE BUG HISTORY:

  2023 call timestamps stored in UTC but applied as ET, shifting windows +4h.

  Fix: subtract 4h from UTC before computing windows.

  Post-fix OI std ~0.07-0.10 (vs ~0.50 pre-fix).

"""

import pandas as pd, numpy as np, os

from pathlib import Path



data_dir = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

taq_dir  = data_dir / "taq"

out_dir  = data_dir / "synchronized"

out_dir.mkdir(exist_ok=True)



sample     = pd.read_csv(data_dir / "selected_sample_40_FINAL.csv")

call_times = pd.read_csv(data_dir / "call_times_extracted.csv")

calls      = sample.merge(call_times, on=['ticker'], how='left')



PRE_WIN  = 30

CALL_WIN = 60



def load_taq(ticker, quarter):

    for ext in ['.csv.gz', '.parquet']:

        p = taq_dir / f"{ticker}_{quarter}_taq{ext}"

        if not p.exists(): continue

        df = (pd.read_csv(p, compression='gzip') if ext == '.csv.gz'

              else pd.read_parquet(p))

        df['timestamp'] = pd.to_datetime(

            df['date'].astype(str) + ' ' + df['time'].astype(str),

            format='mixed', errors='coerce')

        return df[df['type']=='trade'].copy(), df[df['type']=='quote'].copy()

    return None, None



def sign_trades(trades, quotes):

    if trades.empty or quotes.empty:

        return trades.assign(sign=np.nan)

    quotes = quotes.sort_values('timestamp')

    quotes['midpoint'] = (quotes['bid'] + quotes['ofr']) / 2

    quotes = quotes[['timestamp','midpoint']].dropna()

    trades = trades.sort_values('timestamp')

    trades = pd.merge_asof(trades, quotes, on='timestamp',

                           direction='backward',

                           tolerance=pd.Timedelta('5s'))

    trades['sign'] = np.where(trades['price'] > trades['midpoint'],  1,

                     np.where(trades['price'] < trades['midpoint'], -1, np.nan))

    tick = np.sign(trades['price'] - trades['price'].shift(1))

    trades['sign'] = np.where(trades['sign'].isna(), tick, trades['sign'])

    return trades



def order_imbalance(signed_trades):

    valid = signed_trades.dropna(subset=['sign'])

    if len(valid) == 0: return np.nan, 0

    return valid['sign'].sum() / len(valid), len(valid)



all_results = []

for _, row in calls.iterrows():

    ticker  = row['ticker']

    quarter = row['quarter']

    call_dt = (pd.Timestamp(f"{row['call_date']} {row['call_time_gmt']}")

               - pd.Timedelta(hours=4))



    trades, quotes = load_taq(ticker, quarter)

    if trades is None: continue

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

        'ticker': ticker, 'quarter': quarter,

        'call_datetime_et': call_dt, 'call_hour': call_hour,

        'is_market_hours': int(9.5 <= call_hour < 16.0),

        'pre_30m_order_imbalance': oi_pre, 'pre_30m_n_trades': n_pre,

        'during_order_imbalance': oi_dur, 'during_n_trades': n_dur,

        'oi_shift': oi_shift,

    })



sync_df = pd.DataFrame(all_results)

sync_df.to_csv(out_dir / "full_synchronized.csv", index=False)

print(f"Synchronized {len(sync_df)} calls | valid: {sync_df['oi_shift'].notna().sum()}")

