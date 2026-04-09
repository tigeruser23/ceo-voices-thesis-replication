#!/usr/bin/env python3
"""
run_regressions.py
M1-M5 main specifications, regime split, Chow test, BH feature scan.

Outputs:
  tables_v2/regression_results.csv   — M1-M5 full coefficient tables
  tables_v2/regime_results.csv       — regime split + Chow test
  tables_v2/bh_scan_results.csv      — BH-corrected p-values (88 features)
  tables_v2/quarter_by_quarter.csv   — quarter-by-quarter tone coefficients

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang 
# Advisor: Daniel Rigobon
"""

import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy.stats import f as fdist
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
df   = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")
tabs = base / "tables_v2"
tabs.mkdir(exist_ok=True)

CTRL = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"

def stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

def run_ols(formula: str, data: pd.DataFrame, cov: str = "HC3"):
    """Fit OLS with robust SEs; return (model, n) or (None, 0)."""
    clean = (formula.replace("~", " ").replace("+", " ").replace("*", " ")
                    .replace("C(", "").replace(")", ""))
    vars_ = [v.strip() for v in clean.split()
             if v.strip() and not v.strip()[0].isdigit()
             and v.strip() not in ("I",)]
    available = [v for v in vars_ if v in data.columns]
    d = data.dropna(subset=available)
    if len(d) < 20:
        return None, 0
    try:
        return smf.ols(formula, data=d).fit(cov_type=cov), len(d)
    except Exception as e:
        print(f"  OLS error: {e}")
        return None, 0

#  M1-M5 main specifications 
specs = {
    "M1": f"oi_shift ~ {CTRL}",
    "M2": f"oi_shift ~ analyst_tone + {CTRL}",
    "M3": f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
    "M4": (f"oi_shift ~ z_F0 + z_jitter + z_shimmer + z_HNR"
           f" + analyst_tone + {CTRL}"),
    "M5": (f"oi_shift ~ PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8"
           f" + analyst_tone + {CTRL}"),
}

main_results = []
for label, formula in specs.items():
    m, n = run_ols(formula, df)
    if m is None:
        print(f"  {label}: FAILED")
        continue
    for var in m.params.index:
        main_results.append({
            "spec":     label,
            "variable": var,
            "coef":     m.params[var],
            "se":       m.bse[var],
            "t":        m.tvalues[var],
            "p":        m.pvalues[var],
            "sig":      stars(m.pvalues[var]),
            "n":        n,
            "r2":       m.rsquared,
            "r2_adj":   m.rsquared_adj,
        })
    tone_c = m.params.get("analyst_tone", np.nan)
    tone_p = m.pvalues.get("analyst_tone", np.nan)
    stress_c = m.params.get("stress_index", np.nan)
    stress_p = m.pvalues.get("stress_index", np.nan)
    print(f"  {label}  N={n}  R2={m.rsquared:.3f}  "
          f"tone={tone_c:.4f}({stars(tone_p)})  "
          f"stress={stress_c:.4f}({stars(stress_p)})")

pd.DataFrame(main_results).to_csv(tabs / "regression_results.csv", index=False)
print(f"\nSaved: regression_results.csv")

#  Regime split: 2022 vs 2023 
f_base = ("oi_shift ~ analyst_tone"
          " + roa + lnmve + bm + is_market_hours + log_during_n_trades")

regime_results = []
for yr, label in [(None, "Pooled"), (2022, "2022"), (2023, "2023")]:
    sub = df if yr is None else df[df["year"] == yr]
    m, n = run_ols(f_base, sub)
    if m is None:
        continue
    c = m.params.get("analyst_tone", np.nan)
    s = m.bse.get("analyst_tone", np.nan)
    p = m.pvalues.get("analyst_tone", np.nan)
    print(f"  {label:<8} N={n}  coef={c:.4f}  se={s:.4f}  "
          f"p={p:.4f}  {stars(p)}")
    regime_results.append({
        "subsample": label, "n": n,
        "coef": c, "se": s, "p": p, "sig": stars(p),
        "r2": m.rsquared,
    })

# Chow structural break test
m_pool, _   = run_ols(f_base + " + is_2023", df)
m_22,   n22 = run_ols(f_base, df[df["year"] == 2022])
m_23,   n23 = run_ols(f_base, df[df["year"] == 2023])

if all([m_pool, m_22, m_23]):
    k      = len(m_22.params)
    ssr_r  = m_22.ssr + m_23.ssr
    ssr_u  = m_pool.ssr
    F      = ((ssr_u - ssr_r) / k) / (ssr_r / (n22 + n23 - 2 * k))
    ddf    = n22 + n23 - 2 * k
    p_chow = 1 - fdist.cdf(F, k, ddf)
    print(f"\n  Chow test: F({k},{ddf}) = {F:.3f}  p = {p_chow:.4f}  {stars(p_chow)}")
    regime_results.append({
        "subsample": "Chow_test",
        "n": n22 + n23,
        "coef": F,
        "se": np.nan,
        "p": p_chow,
        "sig": stars(p_chow),
        "r2": np.nan,
    })

# Tone x Year interaction
f_interact = f_base + " + is_2023 + analyst_tone:is_2023"
m_int, n_int = run_ols(f_interact, df)
if m_int:
    interaction_coef = m_int.params.get("analyst_tone:is_2023", np.nan)
    interaction_p    = m_int.pvalues.get("analyst_tone:is_2023", np.nan)
    print(f"  Tone×Year  coef={interaction_coef:.4f}  "
          f"p={interaction_p:.4f}  {stars(interaction_p)}")
    regime_results.append({
        "subsample": "Tone_x_2023_interaction",
        "n": n_int,
        "coef": interaction_coef,
        "se": m_int.bse.get("analyst_tone:is_2023", np.nan),
        "p": interaction_p,
        "sig": stars(interaction_p),
        "r2": m_int.rsquared,
    })

pd.DataFrame(regime_results).to_csv(tabs / "regime_results.csv", index=False)
print(f"Saved: regime_results.csv")

#  Quarter-by-quarter tone coefficients 

qxq_results = []
for qtr in sorted(df["quarter"].dropna().unique()):
    sub = df[df["quarter"] == qtr]
    m, n = run_ols(f_base, sub)
    if m is None or n < 15:
        continue
    c = m.params.get("analyst_tone", np.nan)
    p = m.pvalues.get("analyst_tone", np.nan)
    print(f"  {qtr}  N={n}  coef={c:.4f}  p={p:.4f}  {stars(p)}")
    qxq_results.append({"quarter": qtr, "n": n,
                         "coef": c, "p": p, "sig": stars(p)})

pd.DataFrame(qxq_results).to_csv(tabs / "quarter_by_quarter.csv", index=False)
print(f"Saved: quarter_by_quarter.csv")

#  Benjamini-Hochberg scan over 88 eGeMAPS features 

audio_cols = [c for c in df.columns if c.startswith("audio_")]
bh_base    = ["oi_shift", "analyst_tone", "roa", "lnmve", "bm",
              "is_market_hours", "log_during_n_trades", "is_2023"]
bh_df      = df[bh_base + audio_cols].dropna(subset=bh_base)

bh_results = []
for feat in audio_cols:
    sub = bh_df[[feat] + bh_base].dropna().copy()
    if len(sub) < 50:
        continue
    sub[feat] = (sub[feat] - sub[feat].mean()) / sub[feat].std()
    formula = (f"oi_shift ~ {feat} + analyst_tone"
               f" + roa + lnmve + bm"
               f" + is_market_hours + log_during_n_trades + is_2023")
    try:
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
        bh_results.append({
            "feature": feat,
            "coef":    m.params[feat],
            "pval":    m.pvalues[feat],
            "n":       len(sub),
        })
    except Exception:
        pass

bh_res = pd.DataFrame(bh_results).sort_values("pval")
reject, padj, _, _ = multipletests(bh_res["pval"], alpha=0.05, method="fdr_bh")
bh_res["pval_bh"]   = padj
bh_res["reject_bh"] = reject
bh_res.to_csv(tabs / "bh_scan_results.csv", index=False)

n_survivors = reject.sum()
print(f"BH survivors at q=0.05: {n_survivors} / {len(bh_res)}")
if n_survivors > 0:
    print(bh_res[bh_res["reject_bh"]][["feature", "coef", "pval", "pval_bh"]])
print(f"Saved: bh_scan_results.csv")
