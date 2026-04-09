#!/usr/bin/env python3
"""
run_validation.py
Four pre-specified validation tests.

1. Placebo dependent variable (pre-call OI as DV)
2. Permutation test (10,000 shuffles)
3. Bias-corrected bootstrap confidence intervals (5,000 resamples)
4. Wrong-quarter shuffle within firm (1,000 shuffles)

FIX: ticker column is now included in key_vars from the start,
     avoiding the index-alignment bug introduced by reset_index(drop=True).

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
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

base    = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
df      = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")
out_dir = base / "tables_v2"
out_dir.mkdir(exist_ok=True)

CTRL   = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"
N_PERM = 10_000
N_BOOT = 5_000

# FIX: include 'ticker' in key_vars from the start to avoid index
# misalignment in the wrong-quarter shuffle (prior version used
# reset_index(drop=True) then tried to re-attach ticker by index).
key_vars = (["oi_shift", "analyst_tone", "stress_index",
              "pre_30m_order_imbalance", "ticker"]
            + CTRL.split(" + "))

perm_df = df[key_vars].dropna(
    subset=["oi_shift", "analyst_tone", "stress_index",
            "pre_30m_order_imbalance"] + CTRL.split(" + ")
).reset_index(drop=True)

print(f"Validation sample: {len(perm_df)} complete observations")

def ols_coef(data: pd.DataFrame, formula: str, var: str) -> float:
    try:
        return (smf.ols(formula, data=data)
                   .fit(cov_type="HC3")
                   .params.get(var, np.nan))
    except Exception:
        return np.nan

# ── Test 1: Placebo dependent variable ────────────────────────────────────────
print("\n" + "=" * 60)
print("Test 1: Placebo DV (pre_30m_order_imbalance as DV)")
print("=" * 60)

placebo_results = []
for var in ["analyst_tone", "stress_index"]:
    other   = "stress_index" if var == "analyst_tone" else "analyst_tone"
    formula = f"pre_30m_order_imbalance ~ {var} + {other} + {CTRL}"
    m       = smf.ols(formula, data=perm_df).fit(cov_type="HC3")
    c       = m.params.get(var, np.nan)
    p       = m.pvalues.get(var, np.nan)
    outcome = "PASSES (p > .10)" if p > .10 else "CONCERN (p <= .10)"
    print(f"  {var:<25} coef={c:8.4f}  p={p:.4f}  {outcome}")
    placebo_results.append({
        "test": "placebo_dv", "variable": var,
        "coef": c, "p": p, "outcome": outcome,
    })

# ── Test 2: Permutation test ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Test 2: Permutation test ({N_PERM:,} shuffles)")
print("=" * 60)

obs_tone   = ols_coef(perm_df,
                      f"oi_shift ~ analyst_tone + {CTRL}",
                      "analyst_tone")
obs_stress = ols_coef(perm_df,
                      f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
                      "stress_index")
print(f"  Observed: tone={obs_tone:.4f}  stress={obs_stress:.4f}")

np.random.seed(99)
perm_t, perm_s = [], []
for i in range(N_PERM):
    shuf = perm_df.copy()
    shuf["analyst_tone"] = np.random.permutation(perm_df["analyst_tone"].values)
    shuf["stress_index"] = np.random.permutation(perm_df["stress_index"].values)
    perm_t.append(ols_coef(shuf,
                            f"oi_shift ~ analyst_tone + {CTRL}",
                            "analyst_tone"))
    perm_s.append(ols_coef(shuf,
                            f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
                            "stress_index"))
    if (i + 1) % 2000 == 0:
        print(f"  ... {i+1}/{N_PERM}")

perm_p_tone   = np.mean(np.array(perm_t) <= obs_tone)
perm_p_stress = np.mean(np.abs(perm_s) >= np.abs(obs_stress))
print(f"  Permutation p  tone={perm_p_tone:.4f}  stress={perm_p_stress:.4f}")

# ── Test 3: Bias-corrected bootstrap CIs ─────────────────────────────────────
print("\n" + "=" * 60)
print(f"Test 3: Bias-corrected bootstrap ({N_BOOT:,} resamples)")
print("=" * 60)

np.random.seed(77)
boot_t, boot_s = [], []
for i in range(N_BOOT):
    b = perm_df.sample(n=len(perm_df), replace=True)
    boot_t.append(ols_coef(b,
                            f"oi_shift ~ analyst_tone + {CTRL}",
                            "analyst_tone"))
    boot_s.append(ols_coef(b,
                            f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
                            "stress_index"))
    if (i + 1) % 1000 == 0:
        print(f"  ... {i+1}/{N_BOOT}")

boot_results = []
for var, coefs, obs in [("analyst_tone", boot_t, obs_tone),
                         ("stress_index", boot_s, obs_stress)]:
    arr  = np.array(coefs)
    bias = arr.mean() - obs
    ci   = np.percentile(arr, [2.5, 97.5]) - bias
    print(f"  {var:<25} BC 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]  "
          f"(bias={bias:.4f})")
    boot_results.append({
        "test": "bootstrap_ci", "variable": var,
        "obs_coef": obs, "bias": bias,
        "ci_lo": ci[0], "ci_hi": ci[1],
    })

# ── Test 4: Wrong-quarter shuffle within firm ─────────────────────────────────
print("\n" + "=" * 60)
print("Test 4: Wrong-quarter shuffle within firm (1,000 shuffles)")
print("=" * 60)

# FIX: ticker is already in perm_df, no index re-attachment needed
np.random.seed(42)
wq_t, wq_s = [], []
for i in range(1_000):
    shuf = perm_df.copy()
    for ticker in shuf["ticker"].unique():
        mask = shuf["ticker"] == ticker
        if mask.sum() < 2:
            continue
        shuf.loc[mask, "analyst_tone"] = np.random.permutation(
            shuf.loc[mask, "analyst_tone"].values)
        shuf.loc[mask, "stress_index"] = np.random.permutation(
            shuf.loc[mask, "stress_index"].values)
    wq_t.append(ols_coef(shuf,
                          f"oi_shift ~ analyst_tone + {CTRL}",
                          "analyst_tone"))
    wq_s.append(ols_coef(shuf,
                          f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
                          "stress_index"))

wq_p_tone   = np.mean(np.array(wq_t) <= obs_tone)
wq_p_stress = np.mean(np.abs(wq_s) >= np.abs(obs_stress))
print(f"  Wrong-quarter p  tone={wq_p_tone:.4f}  stress={wq_p_stress:.4f}")

# ── Save all validation results ────────────────────────────────────────────────
val_summary = []
val_summary.extend(placebo_results)
val_summary.extend(boot_results)
val_summary.append({
    "test": "permutation", "variable": "analyst_tone",
    "obs_coef": obs_tone, "p": perm_p_tone,
})
val_summary.append({
    "test": "permutation", "variable": "stress_index",
    "obs_coef": obs_stress, "p": perm_p_stress,
})
val_summary.append({
    "test": "wrong_quarter", "variable": "analyst_tone",
    "obs_coef": obs_tone, "p": wq_p_tone,
})
val_summary.append({
    "test": "wrong_quarter", "variable": "stress_index",
    "obs_coef": obs_stress, "p": wq_p_stress,
})

pd.DataFrame(val_summary).to_csv(out_dir / "validation_results.csv", index=False)
print(f"\nSaved: validation_results.csv")
