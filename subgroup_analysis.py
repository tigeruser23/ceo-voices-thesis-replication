#!/usr/bin/env python3
"""
subgroup_analysis.py
Subgroup analysis (Table 5.3) and SVB natural experiment (Table 5.8).

SUBGROUP ANALYSIS (Section 5.4 / Table 5.3):
  Pre-specified subgroups testing three theoretical hypotheses:
  1. Volatility: non-Q5 firms (Q1-Q4) vs Q5 highest-volatility
  2. Call timing: during market hours vs after-hours
  3. Earnings outcome: beat (SUE >= 0) vs miss (SUE < 0)
  M3 control vector throughout; HC3 robust SEs.

SVB NATURAL EXPERIMENT (Section 5.8 / Table 5.8):
  Compares acoustic stress features during Q1 2023 (April-May 2023,
  immediately after the SVB collapse on March 10, 2023) vs all other
  quarters. Two-sample t-tests with unequal variances.

Outputs:
  tables_v2/subgroup_results.csv
  tables_v2/svb_experiment.csv

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
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
tabs = base / "tables_v2"
tabs.mkdir(exist_ok=True)

df = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")

# M3 control vector — must match run_regressions.py exactly
CTRL   = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"
F_TONE = f"oi_shift ~ analyst_tone + stress_index + {CTRL}"
F_BASE = f"oi_shift ~ stress_index + {CTRL}"

def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

def run_m3_subgroup(label, mask, formula=F_TONE):
    """Run M3 on a subsample; return result dict."""
    sub = df[mask].copy()
    needed = ["oi_shift","analyst_tone","stress_index",
              "roa","lnmve","bm","is_market_hours",
              "log_during_n_trades","is_2023"]
    sub = sub.dropna(subset=needed)
    if len(sub) < 15:
        print(f"  {label}: insufficient obs ({len(sub)})")
        return []
    try:
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
    except Exception as e:
        print(f"  {label}: model failed ({e})")
        return []

    rows = []
    for var in ["analyst_tone", "stress_index"]:
        if var not in m.params:
            continue
        c = m.params[var]; p = m.pvalues[var]
        rows.append({
            "subgroup":  label,
            "variable":  var,
            "n":         len(sub),
            "coef":      round(c, 4),
            "se":        round(m.bse[var], 4),
            "p":         round(p, 4),
            "sig":       stars(p),
            "r2":        round(m.rsquared, 4),
        })
    tone_c = m.params.get("analyst_tone", np.nan)
    tone_p = m.pvalues.get("analyst_tone", np.nan)
    stress_c = m.params.get("stress_index", np.nan)
    stress_p = m.pvalues.get("stress_index", np.nan)
    print(f"  {label:<40} N={len(sub):3d}  "
          f"tone={tone_c:+.3f}(p={tone_p:.3f}{stars(tone_p)})  "
          f"stress={stress_c:+.4f}(p={stress_p:.3f})")
    return rows

# ── Section 5.4: Subgroup Analysis ────────────────────────────────────────────
print("=" * 65)
print("SUBGROUP ANALYSIS (Table 5.3)")
print("=" * 65)

all_rows = []

# Full sample for reference
print("\nFull sample reference:")
all_rows.extend(run_m3_subgroup("Full sample (M3)", pd.Series([True]*len(df), index=df.index)))

# 1. Volatility quintile: non-Q5 vs Q5 only (matches thesis exactly)
print("\nVolatility quintile:")
all_rows.extend(run_m3_subgroup(
    "Non-Q5 (Q1-Q4, lower vol.)",
    ~df["quintile"].isin(["Q5_Fragile"])
))
all_rows.extend(run_m3_subgroup(
    "Q5 only (highest volatility)",
    df["quintile"] == "Q5_Fragile"
))

# 2. Call timing: market hours vs after-hours
# Market hours = calls starting 9:30-16:00 ET (is_market_hours == 1)
print("\nCall timing:")
all_rows.extend(run_m3_subgroup(
    "During market hours",
    df["is_market_hours"] == 1
))
all_rows.extend(run_m3_subgroup(
    "After market hours",
    df["is_market_hours"] == 0
))

# 3. Earnings outcome: beat vs miss using SUE from I/B/E/S
# Note: SUE column from financial_controls_all.csv, merged in rebuild_master_v2
print("\nEarnings outcome (stress index only — small N):")
if "surp" in df.columns:
    beat_mask = df["surp"] >= 0
    miss_mask = df["surp"] <  0

    # Earnings beat/miss: thesis only shows stress_index for these (tone n too small)
    beat_sub = df[beat_mask].dropna(subset=["oi_shift","stress_index","roa","lnmve","bm",
                                             "is_market_hours","log_during_n_trades","is_2023"])
    miss_sub = df[miss_mask].dropna(subset=["oi_shift","stress_index","roa","lnmve","bm",
                                             "is_market_hours","log_during_n_trades","is_2023"])

    for label, sub in [("Earnings beat (SUE >= 0)", beat_sub),
                        ("Earnings miss (SUE < 0)",  miss_sub)]:
        if len(sub) < 10:
            print(f"  {label}: insufficient obs ({len(sub)})")
            continue
        try:
            f_stress = f"oi_shift ~ stress_index + {CTRL}"
            m = smf.ols(f_stress, data=sub).fit(cov_type="HC3")
            c = m.params["stress_index"]
            p = m.pvalues["stress_index"]
            print(f"  {label:<40} N={len(sub):3d}  stress={c:+.4f}(p={p:.3f}{stars(p)})")
            all_rows.append({
                "subgroup": label, "variable": "stress_index",
                "n": len(sub), "coef": round(c,4),
                "se": round(m.bse["stress_index"],4),
                "p": round(p,4), "sig": stars(p), "r2": round(m.rsquared,4),
            })
        except Exception as e:
            print(f"  {label}: {e}")

    # Beat-minus-miss difference in stress coefficient
    if len(beat_sub) > 10 and len(miss_sub) > 10:
        try:
            f_s = f"oi_shift ~ stress_index + {CTRL}"
            m_b = smf.ols(f_s, data=beat_sub).fit(cov_type="HC3")
            m_m = smf.ols(f_s, data=miss_sub).fit(cov_type="HC3")
            diff   = m_m.params["stress_index"] - m_b.params["stress_index"]
            se_diff = np.sqrt(m_m.bse["stress_index"]**2 + m_b.bse["stress_index"]**2)
            t_diff  = diff / se_diff
            p_diff  = 2 * (1 - stats.t.cdf(abs(t_diff),
                           df=len(beat_sub)+len(miss_sub)-2))
            print(f"  Beat-miss difference (stress): {diff:+.4f}  "
                  f"(SE={se_diff:.4f})  p={p_diff:.3f}{stars(p_diff)}")
            all_rows.append({
                "subgroup": "Beat-miss difference", "variable": "stress_index",
                "n": len(beat_sub)+len(miss_sub),
                "coef": round(diff,4), "se": round(se_diff,4),
                "p": round(p_diff,4), "sig": stars(p_diff), "r2": np.nan,
            })
        except Exception as e:
            print(f"  Beat-miss difference: {e}")
else:
    print("  surp column not found — run 29_financial_controls.py first")

results_df = pd.DataFrame(all_rows)
results_df.to_csv(tabs / "subgroup_results.csv", index=False)
print(f"\nSaved: subgroup_results.csv  ({len(results_df)} rows)")

# ── Section 5.8: SVB Natural Experiment ───────────────────────────────────────
print("\n" + "=" * 65)
print("SVB NATURAL EXPERIMENT (Table 5.8)")
print("SVB quarter = Q1 2023 (calls April-May 2023, after March 10 collapse)")
print("=" * 65)

# Stress features to compare
stress_vars = {
    "stress_index": "Composite Stress Index",
    "z_jitter":     "z-Jitter",
    "z_F0":         "z-F0 (pitch)",
    "z_shimmer":    "z-Shimmer",
    "z_HNR":        "z-HNR",
}

svb_mask    = df["quarter"] == "Q1_2023"
nonsvb_mask = df["quarter"] != "Q1_2023"

print(f"\nSVB quarter (Q1_2023):     {svb_mask.sum()} calls")
print(f"Non-SVB quarters:          {nonsvb_mask.sum()} calls")
print(f"\n{'Feature':<28} {'SVB Mean':>10} {'Non-SVB':>10} {'Diff':>8} {'p-val':>7}")
print("-" * 68)

svb_rows = []
for col, label in stress_vars.items():
    if col not in df.columns:
        print(f"  {label}: column not found")
        continue
    svb_vals    = df.loc[svb_mask,    col].dropna()
    nonsvb_vals = df.loc[nonsvb_mask, col].dropna()
    if len(svb_vals) < 3 or len(nonsvb_vals) < 3:
        continue
    diff = svb_vals.mean() - nonsvb_vals.mean()
    t, p = stats.ttest_ind(svb_vals, nonsvb_vals, equal_var=False)
    print(f"  {label:<26} {svb_vals.mean():>+10.3f} {nonsvb_vals.mean():>+10.3f} "
          f"{diff:>+8.3f} {p:>7.3f} {stars(p)}")
    svb_rows.append({
        "feature":       label,
        "svb_mean":      round(svb_vals.mean(), 4),
        "nonsvb_mean":   round(nonsvb_vals.mean(), 4),
        "difference":    round(diff, 4),
        "t_stat":        round(t, 3),
        "p_value":       round(p, 3),
        "sig":           stars(p),
        "n_svb":         len(svb_vals),
        "n_nonsvb":      len(nonsvb_vals),
    })

svb_df = pd.DataFrame(svb_rows)
svb_df.to_csv(tabs / "svb_experiment.csv", index=False)
print(f"\nSaved: svb_experiment.csv")
print("\nNote: Composite stress index directionally elevated in SVB quarter")
print("(p > 0.05 expected — small N single quarter; directional, not conclusive)")
