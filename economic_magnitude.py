#!/usr/bin/env python3
"""
economic_magnitude.py
Economic magnitude calculation (Section 5.10).

Translates the non-Q5 subsample analyst tone coefficient into
economically interpretable terms using TAQ trading statistics.

Key calculation (from thesis Section 5.10):
  β̂_tone (non-Q5) = -0.227, σ_tone = 0.173
  |∆OI| = 0.227 × 0.173 ≈ 0.039
  = 14.7% of one SD of OI shift (SD = 0.265)

  Median trade count during call window: N̄_trades = 3,599
  Upper bound (institutional order $50K): $7.0M
  Lower bound (child order $5K):          $0.70M

Also reports incremental R² decomposition (Section 5.3):
  Controls only → M3 (both channels) → isolate each channel.

Output: tables_v2/economic_magnitude.csv
        tables_v2/incremental_r2.csv

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are the author's own.
# Author: Olivia Yang, Princeton ORF 499 Senior Thesis (2024)
# Advisor: Daniel Rigobon
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
tabs = base / "tables_v2"
tabs.mkdir(exist_ok=True)

df = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")

CTRL      = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"
KEEP      = ["oi_shift","analyst_tone","stress_index","roa","lnmve","bm",
             "is_market_hours","log_during_n_trades","is_2023","quintile",
             "during_n_trades"]
data      = df[KEEP].dropna(subset=["oi_shift","analyst_tone","stress_index",
                                     "roa","lnmve","bm","is_market_hours",
                                     "log_during_n_trades","is_2023"])

def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

# ── Section 5.3: Incremental R² decomposition ─────────────────────────────────
print("=" * 65)
print("INCREMENTAL R² DECOMPOSITION (Figure 5.4)")
print("=" * 65)

m_ctrl  = smf.ols(f"oi_shift ~ {CTRL}",                          data=data).fit(cov_type="HC3")
m_tone  = smf.ols(f"oi_shift ~ analyst_tone + {CTRL}",           data=data).fit(cov_type="HC3")
m_stress= smf.ols(f"oi_shift ~ stress_index + {CTRL}",           data=data).fit(cov_type="HC3")
m_both  = smf.ols(f"oi_shift ~ analyst_tone + stress_index + {CTRL}", data=data).fit(cov_type="HC3")

r2_ctrl   = m_ctrl.rsquared
r2_tone   = m_tone.rsquared
r2_stress = m_stress.rsquared
r2_both   = m_both.rsquared

incr_tone   = r2_tone   - r2_ctrl
incr_stress = r2_stress - r2_ctrl
incr_joint  = r2_both   - r2_ctrl

print(f"\n  Controls only (M1):              R² = {r2_ctrl:.4f}")
print(f"  Controls + Tone (M2):            R² = {r2_tone:.4f}  (+{incr_tone*100:.2f}pp from controls)")
print(f"  Controls + Stress:               R² = {r2_stress:.4f}  (+{incr_stress*100:.2f}pp from controls)")
print(f"  Controls + Tone + Stress (M3):   R² = {r2_both:.4f}  (+{incr_joint*100:.2f}pp from controls)")
print(f"\n  Analyst tone unique contribution:  {incr_tone*100:.2f} percentage points")
print(f"  Stress index unique contribution:  {incr_stress*100:.4f} percentage points")
print(f"  Ratio (tone / stress):             {incr_tone/max(incr_stress,0.0001):.0f}x")

r2_df = pd.DataFrame([
    {"model": "Controls only",           "r2": r2_ctrl,   "incr_pp": 0},
    {"model": "Controls + Tone",         "r2": r2_tone,   "incr_pp": round(incr_tone*100,4)},
    {"model": "Controls + Stress",       "r2": r2_stress, "incr_pp": round(incr_stress*100,4)},
    {"model": "Controls + Both (M3)",    "r2": r2_both,   "incr_pp": round(incr_joint*100,4)},
])
r2_df.to_csv(tabs / "incremental_r2.csv", index=False)
print(f"\nSaved: incremental_r2.csv")

# ── Section 5.10: Economic Magnitude ──────────────────────────────────────────
print("\n" + "=" * 65)
print("ECONOMIC MAGNITUDE (Section 5.10)")
print("=" * 65)

# Non-Q5 subsample coefficient (the one used in thesis Section 5.10)
non_q5 = data[~data["quintile"].isin(["Q5_Fragile"])].copy()
m_nonq5 = smf.ols(f"oi_shift ~ analyst_tone + stress_index + {CTRL}",
                   data=non_q5).fit(cov_type="HC3")

beta_tone = m_nonq5.params["analyst_tone"]
se_tone   = m_nonq5.bse["analyst_tone"]
p_tone    = m_nonq5.pvalues["analyst_tone"]

print(f"\nNon-Q5 subsample (N={len(non_q5)}):")
print(f"  β̂_tone = {beta_tone:.4f}  SE={se_tone:.4f}  p={p_tone:.4f} {stars(p_tone)}")

# Key distributional statistics
sigma_tone    = df["analyst_tone"].std()
sigma_oi      = data["oi_shift"].std()
median_trades = df["during_n_trades"].median()
mean_trades   = df["during_n_trades"].mean()

print(f"\nKey statistics:")
print(f"  σ_Tone (analyst tone SD):        {sigma_tone:.4f}")
print(f"  σ_OI (OI shift SD, full sample): {sigma_oi:.4f}")
print(f"  Median trades during call:        {median_trades:.0f}")
print(f"  Mean trades during call:          {mean_trades:.0f}")

# Primary calculation (thesis Equation 5.3)
delta_oi    = abs(beta_tone) * sigma_tone
pct_of_sd   = delta_oi / sigma_oi * 100

print(f"\nPrimary calculation (1-SD increase in analyst tone):")
print(f"  |∆OI| = |{beta_tone:.3f}| × {sigma_tone:.3f} = {delta_oi:.3f}")
print(f"  = {pct_of_sd:.1f}% of one SD of OI shift")

# Dollar translation (thesis Equations 5.4 and 5.5)
N_trades = median_trades
upper_order_size = 50_000   # institutional notional ($)
lower_order_size =  5_000   # child order ($)

upper_flow = N_trades * upper_order_size * delta_oi
lower_flow = N_trades * lower_order_size * delta_oi

print(f"\nDollar translation (N̄_trades = {N_trades:.0f}):")
print(f"  Upper bound (institutional $50K/trade): ${upper_flow/1e6:.1f}M")
print(f"  Lower bound (child order $5K/trade):    ${lower_flow/1e6:.2f}M")
print(f"  True effect lies in [${lower_flow/1e6:.2f}M, ${upper_flow/1e6:.1f}M] range")

# Save
mag_df = pd.DataFrame([{
    "beta_tone_nonq5":    round(beta_tone, 4),
    "se_tone_nonq5":      round(se_tone, 4),
    "p_tone_nonq5":       round(p_tone, 4),
    "sigma_tone":         round(sigma_tone, 4),
    "sigma_oi":           round(sigma_oi, 4),
    "delta_oi_1sd":       round(delta_oi, 4),
    "pct_of_oi_sd":       round(pct_of_sd, 2),
    "median_trades":      int(median_trades),
    "upper_flow_usd_M":   round(upper_flow/1e6, 2),
    "lower_flow_usd_M":   round(lower_flow/1e6, 3),
    "n_nonq5":            len(non_q5),
}])
mag_df.to_csv(tabs / "economic_magnitude.csv", index=False)
print(f"\nSaved: economic_magnitude.csv")
print(f"Saved: incremental_r2.csv")
