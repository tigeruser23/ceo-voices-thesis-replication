#!/usr/bin/env python3
"""
run_regressions_global.py
Cross-market specifications G1-G6 (US + European ADR).

G1: US-only M3 replication (full controls)
G2: EU-only M3 (reduced controls)
G3: Pooled, no interaction (reduced controls)
G4: Pooled, Tone x EU interaction
G5: Pooled, Tone x EU x 2023 triple interaction
G6: EU regime split (2022 vs 2023), matching US regime analysis

All models: HC3 robust SEs.
Reduced control vector: lnmve + is_market_hours + log_during_n_trades + is_2023
(ROA and B/M unavailable for EU firms via WRDS.)

Output: tables_v2/regression_results_global.csv

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

warnings.filterwarnings("ignore")

base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
df   = pd.read_parquet(base / "analysis_dataset_GLOBAL.parquet")
tabs = base / "tables_v2"
tabs.mkdir(exist_ok=True)

CTRL_FULL    = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"
CTRL_REDUCED = "lnmve + is_market_hours + log_during_n_trades + is_2023"

def stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

def run_ols(formula: str, data: pd.DataFrame, cov: str = "HC3"):
    clean = (formula.replace("~", " ").replace("+", " ").replace("*", " ")
                    .replace("C(", "").replace(")", ""))
    vars_ = [v.strip() for v in clean.split()
             if v.strip() and not v.strip()[0].isdigit()]
    available = [v for v in vars_ if v in data.columns]
    d = data.dropna(subset=available)
    if len(d) < 20:
        return None, 0
    try:
        return smf.ols(formula, data=d).fit(cov_type=cov), len(d)
    except Exception as e:
        print(f"  OLS error: {e}")
        return None, 0

# ── G1-G5: pre-specified cross-market specifications ─────────────────────────
specs = {
    "G1": (f"oi_shift ~ analyst_tone + stress_index + {CTRL_FULL}",
           df[df["is_eu"] == 0]),
    "G2": (f"oi_shift ~ analyst_tone + stress_index + {CTRL_REDUCED}",
           df[df["is_eu"] == 1]),
    "G3": (f"oi_shift ~ analyst_tone + stress_index + is_eu + {CTRL_REDUCED}",
           df),
    "G4": (f"oi_shift ~ analyst_tone + is_eu + tone_x_eu"
           f" + stress_index + {CTRL_REDUCED}",
           df),
    "G5": (f"oi_shift ~ analyst_tone + is_eu + is_2023"
           f" + tone_x_eu + tone_x_2023 + tone_x_eu_x_2023"
           f" + stress_index + lnmve + is_market_hours + log_during_n_trades",
           df),
}

print("=" * 70)
print("Cross-market specifications G1-G5")
print("=" * 70)

results = []
models  = {}   # store for Chow test in G6

for label, (formula, data) in specs.items():
    m, n = run_ols(formula, data)
    if m is None:
        print(f"  {label}: FAILED")
        continue
    models[label] = (m, n, data)

    key_vars = ["analyst_tone", "stress_index", "tone_x_eu",
                "tone_x_2023", "tone_x_eu_x_2023", "is_eu"]
    for var in key_vars:
        if var in m.params:
            results.append({
                "spec":     label,
                "variable": var,
                "coef":     m.params[var],
                "se":       m.bse[var],
                "t":        m.tvalues[var],
                "p":        m.pvalues[var],
                "sig":      stars(m.pvalues[var]),
                "n":        n,
                "r2":       m.rsquared,
            })

    tone_c = m.params.get("analyst_tone", np.nan)
    tone_p = m.pvalues.get("analyst_tone", np.nan)
    txeu_c = m.params.get("tone_x_eu", np.nan)
    txeu_p = m.pvalues.get("tone_x_eu", np.nan)
    print(f"  {label}  N={n}  R2={m.rsquared:.3f}  "
          f"tone={tone_c:.4f}({stars(tone_p)})  "
          f"tone_x_eu={txeu_c:.4f}({stars(txeu_p)})")

# ── G6: EU regime split (2022 vs 2023) ────────────────────────────────────────
# FIX: G6 was missing entirely from prior version
print("\n" + "=" * 70)
print("G6: EU regime split (2022 vs 2023)")
print("=" * 70)

eu_df    = df[df["is_eu"] == 1].copy()
f_eu_base = (f"oi_shift ~ analyst_tone + stress_index + {CTRL_REDUCED}")

g6_regime = []
for yr, label in [(None, "EU_Pooled"), (2022, "EU_2022"), (2023, "EU_2023")]:
    sub = eu_df if yr is None else eu_df[eu_df["year"] == yr]
    m, n = run_ols(f_eu_base, sub)
    if m is None:
        continue
    c = m.params.get("analyst_tone", np.nan)
    s = m.bse.get("analyst_tone", np.nan)
    p = m.pvalues.get("analyst_tone", np.nan)
    print(f"  {label:<12} N={n}  coef={c:.4f}  se={s:.4f}  "
          f"p={p:.4f}  {stars(p)}")
    g6_regime.append({
        "spec": "G6", "variable": "analyst_tone",
        "subsample": label, "n": n,
        "coef": c, "se": s, "p": p, "sig": stars(p),
        "r2": m.rsquared,
    })
    results.append({
        "spec":     f"G6_{label}",
        "variable": "analyst_tone",
        "coef":     c, "se": s, "t": m.tvalues.get("analyst_tone", np.nan),
        "p":        p, "sig": stars(p),
        "n":        n, "r2": m.rsquared,
    })

# Chow test: EU 2022 vs EU 2023
m_eu_pool, _    = run_ols(f_eu_base, eu_df)
m_eu_22,  n_e22 = run_ols(f_eu_base, eu_df[eu_df["year"] == 2022])
m_eu_23,  n_e23 = run_ols(f_eu_base, eu_df[eu_df["year"] == 2023])

if all([m_eu_pool, m_eu_22, m_eu_23]):
    k       = len(m_eu_22.params)
    ssr_r   = m_eu_22.ssr + m_eu_23.ssr
    F       = ((m_eu_pool.ssr - ssr_r) / k) / (ssr_r / (n_e22 + n_e23 - 2 * k))
    ddf     = n_e22 + n_e23 - 2 * k
    p_chow  = 1 - fdist.cdf(F, k, ddf)
    print(f"\n  EU Chow test: F({k},{ddf}) = {F:.3f}  p = {p_chow:.4f}  "
          f"{stars(p_chow)}")
    results.append({
        "spec": "G6_Chow", "variable": "Chow_F",
        "coef": F, "se": np.nan, "t": np.nan,
        "p": p_chow, "sig": stars(p_chow),
        "n": n_e22 + n_e23, "r2": np.nan,
    })

# ── Save ───────────────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv(tabs / "regression_results_global.csv", index=False)
print(f"\nSaved: regression_results_global.csv  ({len(results)} rows)")
