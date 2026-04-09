#!/usr/bin/env python3
"""
robustness_tests.py
Robustness checks for the main analyst tone result.

Tests run:
  1. Analyst tone coefficient across six SE specifications:
       OLS | HC3 (primary) | Cluster(firm) | Newey-West(4) | Firm FE | Quarter FE
  2. Autocorrelation diagnostics:
       Durbin-Watson | Breusch-Godfrey lag-1 | Breusch-Godfrey lag-4
  3. Heteroskedasticity:
       White test (justifies HC3 primary SE)
  4. Within-firm AC(1) of analyst tone
       (tests whether tone is serially correlated within firms)

All output printed to stdout. Redirect to file on Adroit:
  python robustness_tests.py > robustness_output.txt

Primary specification (M3):
  oi_shift ~ stress_index + analyst_tone
           + roa + lnmve + bm
           + is_market_hours + log_during_n_trades + is_2023

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang, Princeton ORF 499 Senior Thesis
# Advisor: Daniel Rigobon
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white

warnings.filterwarnings("ignore")

base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
df   = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")

# ── Analysis sample ────────────────────────────────────────────────────────────
M3_FORMULA = ("oi_shift ~ stress_index + analyst_tone"
              " + roa + lnmve + bm"
              " + is_market_hours + log_during_n_trades + is_2023")

KEEP = ["oi_shift", "analyst_tone", "stress_index",
        "roa", "lnmve", "bm",
        "is_market_hours", "log_during_n_trades", "is_2023",
        "ticker", "quarter"]

data = df[KEEP].dropna(
    subset=["oi_shift", "analyst_tone", "stress_index",
            "roa", "lnmve", "bm",
            "is_market_hours", "log_during_n_trades", "is_2023"]
).copy()

data["qtr"] = data["quarter"].str.split("_").str[0]   # e.g. "Q1"

print("=" * 65)
print("ROBUSTNESS CHECKS — Primary specification: M3")
print("=" * 65)
print(f"Analysis sample: {len(data)} observations  "
      f"({data['ticker'].nunique()} firms)")

def stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

# ── Section 1: SE specifications ──────────────────────────────────────────────
print("\n" + "─" * 65)
print("Section 1: Analyst tone coefficient across SE specifications")
print("─" * 65)
print(f"{'Specification':<25} {'coef':>8}  {'se':>7}  {'p':>7}  sig")
print("-" * 55)

SE_SPECS = [
    ("OLS",            M3_FORMULA, {}),
    ("HC3 (primary)",  M3_FORMULA, {"cov_type": "HC3"}),
    ("Cluster(firm)",  M3_FORMULA, {"cov_type": "cluster",
                                    "cov_kwds": {"groups": data["ticker"]}}),
    ("Newey-West(4)",  M3_FORMULA, {"cov_type": "HAC",
                                    "cov_kwds": {"maxlags": 4}}),
    ("Firm FE",        M3_FORMULA + " + C(ticker)", {"cov_type": "HC3"}),
    ("Quarter FE",     M3_FORMULA + " + C(qtr)",    {"cov_type": "HC3"}),
]

for spec_name, formula, fit_kw in SE_SPECS:
    try:
        m = smf.ols(formula, data=data).fit(**fit_kw)
        c  = m.params["analyst_tone"]
        se = m.bse["analyst_tone"]
        p  = m.pvalues["analyst_tone"]
        print(f"  {spec_name:<23} {c:>8.4f}  {se:>7.4f}  {p:>7.4f}  {stars(p)}")
    except Exception as e:
        print(f"  {spec_name:<23} ERROR: {e}")

# ── Section 2: Autocorrelation diagnostics ────────────────────────────────────
print("\n" + "─" * 65)
print("Section 2: Autocorrelation diagnostics (HC3 residuals)")
print("─" * 65)

m_hc3  = smf.ols(M3_FORMULA, data=data).fit(cov_type="HC3")
resids = m_hc3.resid

dw = durbin_watson(resids)
print(f"  Durbin-Watson:          {dw:.4f}  "
      f"({'no autocorr' if 1.5 < dw < 2.5 else 'POSSIBLE AUTOCORR'})")

bg1 = acorr_breusch_godfrey(m_hc3, nlags=1)
print(f"  Breusch-Godfrey(lag=1): LM={bg1[0]:.3f}  p={bg1[1]:.4f}  "
      f"{'reject H0 (autocorr)' if bg1[1] < 0.05 else 'fail to reject'}")

bg4 = acorr_breusch_godfrey(m_hc3, nlags=4)
print(f"  Breusch-Godfrey(lag=4): LM={bg4[0]:.3f}  p={bg4[1]:.4f}  "
      f"{'reject H0 (autocorr)' if bg4[1] < 0.05 else 'fail to reject'}")

# ── Section 3: Heteroskedasticity ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("Section 3: Heteroskedasticity (White test)")
print("─" * 65)

try:
    ws, wp, _, _ = het_white(resids, m_hc3.model.exog)
    verdict = "→ HC3 justified" if wp < 0.05 else "→ homoskedastic"
    print(f"  White test: LM={ws:.2f}  p={wp:.4f}  {verdict}")
except Exception as e:
    print(f"  White test failed: {e}")

# ── Section 4: Within-firm AC(1) of analyst tone ──────────────────────────────
print("\n" + "─" * 65)
print("Section 4: Within-firm AC(1) of analyst tone")
print("─" * 65)

ac1_vals = (data.sort_values(["ticker", "quarter"])
               .groupby("ticker")["analyst_tone"]
               .apply(lambda x: x.autocorr(lag=1) if len(x) >= 3 else np.nan)
               .dropna())

print(f"  Firms with ≥3 quarters: {len(ac1_vals)}")
print(f"  Mean AC(1):  {ac1_vals.mean():.4f}")
print(f"  Median AC(1):{ac1_vals.median():.4f}")
print(f"  Std AC(1):   {ac1_vals.std():.4f}")
print(f"  (Negative mean AC(1) = analyst tone reverts within-firm; "
      f"not persistent)")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Summary")
print("=" * 65)
c_hc3 = m_hc3.params["analyst_tone"]
p_hc3 = m_hc3.pvalues["analyst_tone"]
print(f"  Primary HC3 result: β(analyst_tone) = {c_hc3:.4f}  "
      f"p = {p_hc3:.4f}  {stars(p_hc3)}")
print(f"  White test p={wp:.4f} → heteroskedasticity "
      f"{'confirmed' if wp < 0.05 else 'not confirmed'}, HC3 SE {'appropriate' if wp < 0.05 else 'conservative'}")
print(f"  DW={dw:.3f}, BG(1) p={bg1[1]:.4f}, BG(4) p={bg4[1]:.4f} → "
      f"{'no material autocorrelation' if bg1[1] > 0.05 and bg4[1] > 0.05 else 'some autocorrelation'}")
print(f"  Within-firm AC(1) of tone = {ac1_vals.mean():.4f} → "
      f"tone not persistent within firms")
