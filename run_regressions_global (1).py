
#!/usr/bin/env python3

"""

run_regressions_global.py

G1-G5 cross-market specifications (US + European ADR).

"""

import os, pandas as pd, numpy as np, warnings

from pathlib import Path

import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')



base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

df   = pd.read_parquet(base / "analysis_dataset_GLOBAL.parquet")

tabs = base / "tables_v2"

tabs.mkdir(exist_ok=True)



CTRL_FULL    = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"

CTRL_REDUCED = "lnmve + is_market_hours + log_during_n_trades + is_2023"



def stars(p):

    return '***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''



def run_ols(formula, data, cov='HC3'):

    vars_ = [v.strip() for v in

             formula.replace('~',' ').replace('+',' ').replace('*',' ')

                    .replace('C(','').replace(')','').split()

             if v.strip() and not v.strip()[0].isdigit()]

    d = data.dropna(subset=vars_)

    try:

        return smf.ols(formula, data=d).fit(cov_type=cov), len(d)

    except:

        return None, 0



specs = {

    "G1": (f"oi_shift ~ analyst_tone + stress_index + {CTRL_FULL}",

           df[df['is_eu']==0]),

    "G2": (f"oi_shift ~ analyst_tone + stress_index + {CTRL_REDUCED}",

           df[df['is_eu']==1]),

    "G3": (f"oi_shift ~ analyst_tone + stress_index + is_eu + {CTRL_REDUCED}",

           df),

    "G4": (f"oi_shift ~ analyst_tone + is_eu + tone_x_eu + stress_index + {CTRL_REDUCED}",

           df),

    "G5": (f"oi_shift ~ analyst_tone + is_eu + is_2023 + tone_x_eu "

           f"+ tone_x_2023 + tone_x_eu_x_2023 + stress_index + lnmve "

           f"+ is_market_hours + log_during_n_trades",

           df),

}



results = []

for label, (formula, data) in specs.items():

    m, n = run_ols(formula, data)

    if m:

        for var in ['analyst_tone','stress_index','tone_x_eu',

                    'tone_x_2023','tone_x_eu_x_2023']:

            if var in m.params:

                results.append({

                    'spec': label, 'variable': var,

                    'coef': m.params[var],

                    'se':   m.bse[var],

                    'p':    m.pvalues[var],

                    'sig':  stars(m.pvalues[var]),

                    'n':    n, 'r2': m.rsquared

                })

        print(f"{label} N={n} R2={m.rsquared:.3f} "

              f"tone={m.params.get('analyst_tone',np.nan):.3f} "

              f"p={m.pvalues.get('analyst_tone',np.nan):.3f}")



pd.DataFrame(results).to_csv(tabs / "regression_results_global.csv", index=False)

print("Saved: regression_results_global.csv")

