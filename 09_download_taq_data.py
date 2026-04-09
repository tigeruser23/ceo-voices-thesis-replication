
#!/usr/bin/env python3

"""

09_download_taq_data.py / 15_download_taq_precise.py

Download NYSE TAQ millisecond data for each firm-quarter.

Six-hour window centered on call start time (UTC -> ET corrected).

"""



import wrds

import pandas as pd

import numpy as np

import os



db       = wrds.Connection()

data_dir = (f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

os.makedirs(f"{data_dir}/taq", exist_ok=True)



sample     = pd.read_csv(f"{data_dir}/selected_sample_40_FINAL.csv")

call_times = pd.read_csv(f"{data_dir}/call_times_extracted.csv")

calls      = sample.merge(call_times, on=['ticker'], how='left')



WINDOW_HOURS_BEFORE = 3

WINDOW_HOURS_AFTER  = 3



for _, row in calls.iterrows():

    ticker  = row['ticker']

    quarter = row['quarter']

    # CRITICAL: subtract 4h to convert UTC -> Eastern Time

    call_dt = (

        pd.Timestamp(f"{row['call_date']} {row['call_time_gmt']}")

        - pd.Timedelta(hours=4))



    start_dt = call_dt - pd.Timedelta(hours=WINDOW_HOURS_BEFORE)

    end_dt   = call_dt + pd.Timedelta(hours=WINDOW_HOURS_AFTER)

    out_path = f"{data_dir}/taq/{ticker}_{quarter}_taq.csv.gz"



    if os.path.exists(out_path):

        print(f"  SKIP {ticker} {quarter} (exists)")

        continue



    date_str = start_dt.strftime('%Y%m%d')

    time_s   = start_dt.strftime('%H:%M:%S')

    time_e   = end_dt.strftime('%H:%M:%S')



    try:

        taq = db.raw_sql(f"""

            SELECT date, time_m AS time, sym_root AS ticker,

                   price, size, tr_corr, tr_scond,

                   bid, ofr, bidsiz, ofrsiz, mode, 'trade' AS type

            FROM taqmsec.ctm_{date_str}

            WHERE sym_root = '{ticker}'

              AND time_m BETWEEN '{time_s}' AND '{time_e}'

            UNION ALL

            SELECT date, time_m AS time, sym_root AS ticker,

                   NULL, NULL, NULL, NULL,

                   bid, ofr, bidsiz, ofrsiz, qu_cond, 'quote' AS type

            FROM taqmsec.cqm_{date_str}

            WHERE sym_root = '{ticker}'

              AND time_m BETWEEN '{time_s}' AND '{time_e}'

        """)

        taq.to_csv(out_path, index=False, compression='gzip')

        print(f"  OK  {ticker} {quarter}  {len(taq):,} rows")

    except Exception as e:

        print(f"  ERR {ticker} {quarter}: {e}")

