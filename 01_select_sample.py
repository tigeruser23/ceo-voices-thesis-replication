
#!/usr/bin/env python3

"""

01_select_sample.py

Stratified random sampling from CRSP universe.

Sorts firms into 5 volatility quintiles, draws 8 per quintile.

"""



import wrds

import pandas as pd

import numpy as np



RANDOM_SEED    = 42

N_QUINTILES    = 5

N_PER_QUINTILE = 8



db = wrds.Connection()



crsp = db.raw_sql("""

    SELECT permno, ticker, date, ret, shrout, prc

    FROM crsp.msf

    WHERE date BETWEEN '2021-01-01' AND '2023-12-31'

      AND exchcd IN (1, 2, 3)

      AND ret IS NOT NULL

      AND ABS(prc) > 1

""")



study_quarters = pd.period_range('2022Q1', '2023Q4', freq='Q')

crsp['period']  = pd.to_datetime(crsp['date']).dt.to_period('Q')



coverage = (crsp[crsp['period'].isin(study_quarters)]

            .groupby('permno')['period']

            .nunique()

            .reset_index(name='n_quarters'))



full_coverage = coverage[coverage['n_quarters'] == 8]['permno']

crsp_full     = crsp[crsp['permno'].isin(full_coverage)].copy()



crsp_full['date'] = pd.to_datetime(crsp_full['date'])

crsp_full         = crsp_full.sort_values(['permno', 'date'])

vol_df = (crsp_full[crsp_full['date'] <= '2022-06-30']

          .groupby('permno')['ret']

          .std()

          .reset_index(name='hist_vol'))



tickers = crsp_full.groupby('permno')['ticker'].last().reset_index()

vol_df  = vol_df.merge(tickers, on='permno').dropna(subset=['hist_vol'])



vol_df['quintile'] = pd.qcut(

    vol_df['hist_vol'], q=5,

    labels=['Q1_Sleepy', 'Q2', 'Q3', 'Q4', 'Q5_Fragile'])



np.random.seed(RANDOM_SEED)

sample = (vol_df.groupby('quintile', group_keys=False)

          .apply(lambda g: g.sample(n=N_PER_QUINTILE,

                                    random_state=RANDOM_SEED)))



sample = (sample[['permno', 'ticker', 'quintile', 'hist_vol']]

          .reset_index(drop=True))

sample.to_csv('data/selected_sample_40_FINAL.csv', index=False)

print(f"Final sample: {len(sample)} firms")

print(sample.groupby('quintile').size())

