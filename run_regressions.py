
#!/usr/bin/env python3

"""

run_regressions.py

M1-M5, regime split, Chow test, BH scan.

"""

import os, pandas as pd, numpy as np, warnings

from pathlib import Path

import statsmodels.formula.api as smf

from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')



base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

df   = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")

tabs = base / "tables_v2"

tabs.mkdir(exist_ok=True)



CTRL = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"



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

    "M1": f"oi_shift ~ {CTRL}",

    "M2": f"oi_shift ~ analyst_tone + {CTRL}",

    "M3": f"oi_shift ~ stress_index + analyst_tone + {CTRL}",

    "M4": f"oi_shift ~ z_F0 + z_jitter + z_shimmer + z_HNR + analyst_tone + {CTRL}",

    "M5": f"oi_shift ~ PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8 + analyst_tone + {CTRL}",

}

for label, formula in specs.items():

    m, n = run_ols(formula, df)



f_base = "oi_shift ~ analyst_tone + roa+lnmve+bm+is_market_hours+log_during_n_trades"

for yr in [None, 2022, 2023]:

    sub = df if yr is None else df[df['year']==yr]

    m, n = run_ols(f_base, sub)

    if m:

        c = m.params['analyst_tone']; p = m.pvalues['analyst_tone']

        print(f"  {str(yr):<6} N={n}  tone_coef={c:.4f}  p={p:.4f} {stars(p)}")



f_ctrl    = f"oi_shift ~ analyst_tone + {CTRL}"

m_pool, _ = run_ols(f_ctrl, df)

m_22, n22 = run_ols(f_base, df[df['year']==2022])

m_23, n23 = run_ols(f_base, df[df['year']==2023])

if all([m_pool, m_22, m_23]):

    k = len(m_22.params); ssr_r = m_22.ssr + m_23.ssr

    F = ((m_pool.ssr - ssr_r) / k) / (ssr_r / (n22 + n23 - 2*k))

    from scipy.stats import f as fdist

    ddf  = n22 + n23 - 2*k

    pval = 1 - fdist.cdf(F, k, ddf)

    print(f"  Chow F({k},{ddf}) = {F:.3f}  p = {pval:.4f}")



audio_cols = [c for c in df.columns if c.startswith('audio_')]

bh_base    = ['oi_shift','analyst_tone','roa','lnmve','bm',

              'is_market_hours','log_during_n_trades','is_2023']

bh_df      = df[bh_base + audio_cols].dropna(subset=bh_base)



bh_results = []

for feat in audio_cols:

    sub = bh_df[[feat]+bh_base].dropna()

    if len(sub) < 50: continue

    sub = sub.copy()

    sub[feat] = (sub[feat]-sub[feat].mean()) / sub[feat].std()

    try:

        formula = (f"oi_shift ~ {feat} + analyst_tone"

                   f" + roa+lnmve+bm+is_market_hours+log_during_n_trades+is_2023")

        m = smf.ols(formula, data=sub).fit(cov_type='HC3')

        bh_results.append({'feature':feat,'coef':m.params[feat],

                           'pval':m.pvalues[feat],'n':len(sub)})

    except: pass



bh_res = pd.DataFrame(bh_results).sort_values('pval')

reject, padj, _, _ = multipletests(bh_res['pval'], alpha=0.05, method='fdr_bh')

bh_res['pval_bh'] = padj; bh_res['reject_bh'] = reject

bh_res.to_csv(tabs / "bh_scan_results.csv", index=False)

print(f"BH survivors at q=0.05: {reject.sum()} / {len(bh_res)}")

