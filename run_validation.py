
#!/usr/bin/env python3

"""

run_validation.py

Four pre-specified validation tests.

"""

import os, pandas as pd, numpy as np, warnings

from pathlib import Path

import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')



base    = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

df      = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")

out_dir = base / "tables_v2"

out_dir.mkdir(exist_ok=True)



CTRL   = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"

N_PERM = 10_000

N_BOOT = 5_000



key_vars = (['oi_shift','analyst_tone','stress_index','pre_30m_order_imbalance']

            + CTRL.split(' + '))

perm_df  = df[key_vars].dropna().reset_index(drop=True)



def ols_coef(data, formula, var):

    try:

        return smf.ols(formula, data=data).fit(cov_type='HC3').params.get(var, np.nan)

    except:

        return np.nan



for var in ['analyst_tone','stress_index']:

    other   = 'stress_index' if var == 'analyst_tone' else 'analyst_tone'

    formula = f"pre_30m_order_imbalance ~ {var} + {other} + {CTRL}"

    m = smf.ols(formula, data=perm_df).fit(cov_type='HC3')

    c = m.params.get(var, np.nan); p = m.pvalues.get(var, np.nan)

    print(f"Placebo DV  {var:<20} coef={c:8.4f}  p={p:.4f}  "

          f"{'PASSES' if p>.10 else 'CONCERN'}")



obs_tone   = ols_coef(perm_df, f"oi_shift ~ analyst_tone + {CTRL}", 'analyst_tone')

obs_stress = ols_coef(perm_df,

                      f"oi_shift ~ stress_index + analyst_tone + {CTRL}",

                      'stress_index')

perm_t, perm_s = [], []

np.random.seed(99)

for _ in range(N_PERM):

    shuf = perm_df.copy()

    shuf['analyst_tone'] = np.random.permutation(perm_df['analyst_tone'].values)

    shuf['stress_index'] = np.random.permutation(perm_df['stress_index'].values)

    perm_t.append(ols_coef(shuf, f"oi_shift ~ analyst_tone + {CTRL}", 'analyst_tone'))

    perm_s.append(ols_coef(shuf,

                            f"oi_shift ~ stress_index + analyst_tone + {CTRL}",

                            'stress_index'))



print(f"Permutation  tone   p={np.mean(np.array(perm_t) <= obs_tone):.4f}")

print(f"Permutation  stress p={np.mean(np.abs(perm_s) >= np.abs(obs_stress)):.4f}")



boot_t, boot_s = [], []

np.random.seed(77)

for _ in range(N_BOOT):

    b = perm_df.sample(n=len(perm_df), replace=True)

    boot_t.append(ols_coef(b, f"oi_shift ~ analyst_tone + {CTRL}", 'analyst_tone'))

    boot_s.append(ols_coef(b,

                            f"oi_shift ~ stress_index + analyst_tone + {CTRL}",

                            'stress_index'))

for var, coefs, obs in [('analyst_tone',boot_t,obs_tone),

                         ('stress_index',boot_s,obs_stress)]:

    arr  = np.array(coefs); bias = arr.mean() - obs

    ci   = np.percentile(arr, [2.5, 97.5]) - bias

    print(f"Bootstrap 95% CI  {var:<20} [{ci[0]:.4f}, {ci[1]:.4f}]")



if 'ticker' not in perm_df.columns:

    perm_df['ticker'] = df.loc[perm_df.index, 'ticker'].values

np.random.seed(42)

wq_t, wq_s = [], []

for _ in range(1_000):

    shuf = perm_df.copy()

    for ticker in shuf['ticker'].unique():

        mask = shuf['ticker'] == ticker

        if mask.sum() < 2: continue

        shuf.loc[mask,'analyst_tone'] = np.random.permutation(

            shuf.loc[mask,'analyst_tone'].values)

        shuf.loc[mask,'stress_index'] = np.random.permutation(

            shuf.loc[mask,'stress_index'].values)

    wq_t.append(ols_coef(shuf, f"oi_shift ~ analyst_tone + {CTRL}", 'analyst_tone'))

    wq_s.append(ols_coef(shuf,

                          f"oi_shift ~ stress_index + analyst_tone + {CTRL}",

                          'stress_index'))



print(f"Wrong-quarter  tone   p={np.mean(np.array(wq_t) <= obs_tone):.4f}")

print(f"Wrong-quarter  stress p={np.mean(np.abs(wq_s) >= np.abs(obs_stress)):.4f}")

