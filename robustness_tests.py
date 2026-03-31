
#!/usr/bin/env python3

"""

robustness_tests.py

SE specs, autocorrelation, White test.

"""

import os, pandas as pd, numpy as np, warnings

from pathlib import Path

import statsmodels.formula.api as smf

from statsmodels.stats.stattools import durbin_watson

from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white

warnings.filterwarnings('ignore')



base    = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

df      = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")

FORMULA = ("oi_shift ~ analyst_tone + stress_index"

           " + roa + lnmve + bm"

           " + is_market_hours + log_during_n_trades + is_2023")

keep    = ['oi_shift','analyst_tone','stress_index','roa','lnmve','bm',

           'is_market_hours','log_during_n_trades','is_2023','ticker','quarter']

data    = df[keep].dropna(

              subset=['oi_shift','analyst_tone','stress_index','roa',

                      'lnmve','bm','is_market_hours',

                      'log_during_n_trades','is_2023']).copy()



def stars(p):

    return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''



print(f"{'SE spec':<25} {'coef':>8}  {'se':>7}  {'p':>7}  sig")

print("-" * 55)



se_specs = [

    ("OLS",           {}),

    ("HC3 (primary)", {'cov_type':'HC3'}),

    ("Cluster(firm)", {'cov_type':'cluster',

                       'cov_kwds':{'groups':data['ticker']}}),

    ("Newey-West(4)", {'cov_type':'HAC','cov_kwds':{'maxlags':4}}),

    ("Firm FE",       {'cov_type':'HC3'}),

    ("Quarter FE",    {'cov_type':'HC3'}),

]



for spec, kw in se_specs:

    f = FORMULA

    if spec == "Firm FE":    f += " + C(ticker)"

    if spec == "Quarter FE":

        data['qtr'] = data['quarter'].str.split('_').str[0]

        f = FORMULA + " + C(qtr)"

    m  = smf.ols(f, data=data).fit(**kw) if kw else smf.ols(f, data=data).fit()

    c  = m.params['analyst_tone']

    se = m.bse['analyst_tone']

    p  = m.pvalues['analyst_tone']

    print(f"{spec:<25} {c:>8.4f}  {se:>7.4f}  {p:>7.4f}  {stars(p)}")



m_hc3  = smf.ols(FORMULA, data=data).fit(cov_type='HC3')

resids = m_hc3.resid

dw     = durbin_watson(resids)

bg1    = acorr_breusch_godfrey(m_hc3, nlags=1)

bg4    = acorr_breusch_godfrey(m_hc3, nlags=4)

ws, wp, _, _ = het_white(resids, m_hc3.model.exog)



print(f"\nDurbin-Watson:       {dw:.4f}  (2.0 = no autocorr)")

print(f"Breusch-Godfrey(1):  LM={bg1[0]:.3f}  p={bg1[1]:.4f}")

print(f"Breusch-Godfrey(4):  LM={bg4[0]:.3f}  p={bg4[1]:.4f}")

print(f"White test:          LM={ws:.1f}  p={wp:.4f}  "

      f"{'-> use HC3' if wp<0.05 else '-> homoskedastic'}")



if 'ticker' in data.columns:

    ac1 = (data.sort_values(['ticker'])

               .groupby('ticker')['analyst_tone']

               .apply(lambda x: x.autocorr(1) if len(x)>=3 else np.nan))

    print(f"Within-firm AC(1) of analyst_tone: {ac1.mean():.4f}")

