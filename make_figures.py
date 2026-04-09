#!/usr/bin/env python3
"""
make_figures.py
Generate all thesis figures (Figures 5.1-5.7).

Requires prior completion of:
  run_regressions.py   → tables_v2/regression_results.csv
                          tables_v2/regime_results.csv
                          tables_v2/quarter_by_quarter.csv
                          tables_v2/bh_scan_results.csv
  run_validation.py    → tables_v2/validation_results.csv
  robustness_tests.py  → (run inline here)
  economic_magnitude.py→ tables_v2/incremental_r2.csv

Figures produced (saved to figures/):
  fig5_1_coef_m1_m5.pdf         M1-M5 coefficient plot
  fig5_2_regime_split.pdf        Regime split + quarter-by-quarter
  fig5_3_oi_distributions.pdf    OI shift histograms by year
  fig5_4_incremental_r2.pdf      Incremental R² decomposition
  fig5_5_permutation.pdf         Permutation null distributions
  fig5_6_bootstrap_ci.pdf        Bootstrap confidence intervals
  fig5_7_se_specs.pdf            SE specifications coefficient plot

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
import matplotlib
matplotlib.use("Agg")   # headless backend for Adroit HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

base    = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
tabs    = base / "tables_v2"
fig_dir = base / "figures"
fig_dir.mkdir(exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

df = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")

CTRL = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"
KEEP = ["oi_shift","analyst_tone","stress_index","roa","lnmve","bm",
        "is_market_hours","log_during_n_trades","is_2023","year","quarter"]
data = df[KEEP].dropna(subset=["oi_shift","analyst_tone","stress_index",
                                "roa","lnmve","bm","is_market_hours",
                                "log_during_n_trades","is_2023"])

def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

print(f"Analysis sample: {len(data)} observations")

# ── Figure 5.1: M1-M5 coefficient plot ────────────────────────────────────────
print("Generating Figure 5.1...")

specs = {
    "M1": f"oi_shift ~ {CTRL}",
    "M2": f"oi_shift ~ analyst_tone + {CTRL}",
    "M3": f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
    "M4": f"oi_shift ~ z_F0 + z_jitter + z_shimmer + z_HNR + analyst_tone + {CTRL}",
    "M5": f"oi_shift ~ PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8 + analyst_tone + {CTRL}",
}

tone_coefs, tone_lo, tone_hi = [], [], []
stress_coefs, stress_lo, stress_hi = [], [], []
labels = []

for label, formula in specs.items():
    needed = [v.strip() for v in
              formula.replace("~"," ").replace("+"," ").split()
              if v.strip() and not v.strip()[0].isdigit()]
    sub = data.dropna(subset=[v for v in needed if v in data.columns])
    try:
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
    except Exception:
        continue
    labels.append(label)
    for coefs, lo_list, hi_list, var in [
        (tone_coefs, tone_lo, tone_hi, "analyst_tone"),
        (stress_coefs, stress_lo, stress_hi, "stress_index"),
    ]:
        c  = m.params.get(var, np.nan)
        se = m.bse.get(var, np.nan)
        coefs.append(c)
        lo_list.append(c - 1.96*se)
        hi_list.append(c + 1.96*se)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 4))

ax.errorbar(x - 0.1, tone_coefs,
            yerr=[np.array(tone_coefs)-np.array(tone_lo),
                  np.array(tone_hi)-np.array(tone_coefs)],
            fmt="o", color="#2166ac", capsize=4, label="Analyst Tone")
ax.errorbar(x + 0.1, stress_coefs,
            yerr=[np.array(stress_coefs)-np.array(stress_lo),
                  np.array(stress_hi)-np.array(stress_coefs)],
            fmt="s", color="#d6604d", capsize=4, label="Stress Index")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlabel("Specification"); ax.set_ylabel("Coefficient estimate")
ax.set_title("Figure 5.1: Analyst Tone and Stress Index Coefficients (M1–M5)", pad=12)
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(fig_dir / "fig5_1_coef_m1_m5.pdf")
plt.close()
print("  Saved: fig5_1_coef_m1_m5.pdf")

# ── Figure 5.2: Regime split + quarter-by-quarter ────────────────────────────
print("Generating Figure 5.2...")

f_base = ("oi_shift ~ analyst_tone"
          " + roa + lnmve + bm + is_market_hours + log_during_n_trades")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: full / 2022 / 2023
regime_labels, regime_coefs, regime_lo, regime_hi, regime_cols = [], [], [], [], []
for yr, col in [(None,"#555555"),(2022,"#2166ac"),(2023,"#d6604d")]:
    sub = data if yr is None else data[data["year"]==yr]
    sub = sub.dropna(subset=["oi_shift","analyst_tone","roa","lnmve","bm",
                              "is_market_hours","log_during_n_trades"])
    try:
        m = smf.ols(f_base, data=sub).fit(cov_type="HC3")
        c = m.params["analyst_tone"]; se = m.bse["analyst_tone"]
        regime_labels.append("Full" if yr is None else str(yr))
        regime_coefs.append(c); regime_cols.append(col)
        regime_lo.append(c - 1.96*se); regime_hi.append(c + 1.96*se)
    except Exception:
        pass

x = np.arange(len(regime_labels))
axes[0].bar(x, regime_coefs, color=regime_cols, alpha=0.75, width=0.5)
axes[0].errorbar(x, regime_coefs,
                 yerr=[np.array(regime_coefs)-np.array(regime_lo),
                       np.array(regime_hi)-np.array(regime_coefs)],
                 fmt="none", color="black", capsize=5)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_xticks(x); axes[0].set_xticklabels(regime_labels)
axes[0].set_ylabel("Analyst Tone Coefficient")
axes[0].set_title("Regime Split (Full / 2022 / 2023)")

# Right: quarter-by-quarter
qtrs = sorted(data["quarter"].dropna().unique())
qxq_c, qxq_lo, qxq_hi, qxq_labels, qxq_cols = [], [], [], [], []
for qtr in qtrs:
    sub = data[data["quarter"]==qtr].dropna(
        subset=["oi_shift","analyst_tone","roa","lnmve","bm",
                "is_market_hours","log_during_n_trades"])
    if len(sub) < 10: continue
    try:
        m = smf.ols(f_base, data=sub).fit(cov_type="HC3")
        c = m.params["analyst_tone"]; se = m.bse["analyst_tone"]
        yr = int(qtr.split("_")[1])
        qxq_c.append(c); qxq_lo.append(c-1.96*se); qxq_hi.append(c+1.96*se)
        qxq_labels.append(qtr.replace("_","\n"))
        qxq_cols.append("#2166ac" if yr==2022 else "#d6604d")
    except Exception:
        pass

xq = np.arange(len(qxq_c))
axes[1].bar(xq, qxq_c, color=qxq_cols, alpha=0.75, width=0.6)
axes[1].errorbar(xq, qxq_c,
                 yerr=[np.array(qxq_c)-np.array(qxq_lo),
                       np.array(qxq_hi)-np.array(qxq_c)],
                 fmt="none", color="black", capsize=4)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_xticks(xq); axes[1].set_xticklabels(qxq_labels, fontsize=8)
axes[1].set_ylabel("Analyst Tone Coefficient")
axes[1].set_title("Quarter-by-Quarter")
blue_p  = mpatches.Patch(color="#2166ac", alpha=0.75, label="2022")
red_p   = mpatches.Patch(color="#d6604d", alpha=0.75, label="2023")
axes[1].legend(handles=[blue_p, red_p], framealpha=0.9)

fig.suptitle("Figure 5.2: Analyst Tone Regime Split", y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / "fig5_2_regime_split.pdf", bbox_inches="tight")
plt.close()
print("  Saved: fig5_2_regime_split.pdf")

# ── Figure 5.3: OI shift distributions by year ───────────────────────────────
print("Generating Figure 5.3...")
oi_data = df[df["oi_shift"].notna()].copy()
oi_22 = oi_data[oi_data["year"]==2022]["oi_shift"]
oi_23 = oi_data[oi_data["year"]==2023]["oi_shift"]

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-1.2, 1.2, 50)
ax.hist(oi_22, bins=bins, alpha=0.6, color="#2166ac",
        label=f"2022 (N={len(oi_22)}, SD={oi_22.std():.3f})", density=True)
ax.hist(oi_23, bins=bins, alpha=0.6, color="#d6604d",
        label=f"2023 (N={len(oi_23)}, SD={oi_23.std():.3f})", density=True)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("OI Shift (during − pre)")
ax.set_ylabel("Density")
ax.set_title("Figure 5.3: Order Imbalance Shift Distributions by Year", pad=12)
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(fig_dir / "fig5_3_oi_distributions.pdf")
plt.close()
print("  Saved: fig5_3_oi_distributions.pdf")

# ── Figure 5.4: Incremental R² decomposition ─────────────────────────────────
print("Generating Figure 5.4...")
r2_path = tabs / "incremental_r2.csv"
if r2_path.exists():
    r2_df = pd.read_csv(r2_path)
    r2_labels = r2_df["model"].tolist()
    r2_vals   = r2_df["r2"].tolist()
else:
    # Compute on the fly if economic_magnitude.py hasn't been run
    m_ctrl  = smf.ols(f"oi_shift ~ {CTRL}", data=data).fit(cov_type="HC3")
    m_tone  = smf.ols(f"oi_shift ~ analyst_tone + {CTRL}", data=data).fit(cov_type="HC3")
    m_stress= smf.ols(f"oi_shift ~ stress_index + {CTRL}", data=data).fit(cov_type="HC3")
    m_both  = smf.ols(f"oi_shift ~ analyst_tone + stress_index + {CTRL}", data=data).fit(cov_type="HC3")
    r2_labels = ["Controls only", "Controls + Tone", "Controls + Stress", "Controls + Both (M3)"]
    r2_vals   = [m_ctrl.rsquared, m_tone.rsquared, m_stress.rsquared, m_both.rsquared]

colors = ["#aaaaaa","#2166ac","#d6604d","#4dac26"]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(len(r2_labels)), [v*100 for v in r2_vals],
              color=colors, alpha=0.8)
ax.set_xticks(range(len(r2_labels)))
ax.set_xticklabels([l.replace(" + ","\n+\n") for l in r2_labels], fontsize=9)
ax.set_ylabel("R² (%)")
ax.set_title("Figure 5.4: Incremental R² Decomposition", pad=12)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val*100:.2f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(fig_dir / "fig5_4_incremental_r2.pdf")
plt.close()
print("  Saved: fig5_4_incremental_r2.pdf")

# ── Figures 5.5 and 5.6: Permutation and bootstrap ───────────────────────────
print("Generating Figures 5.5 and 5.6 (permutation + bootstrap)...")

val_path = tabs / "validation_results.csv"
if val_path.exists():
    val_df = pd.read_csv(val_path)
    # Load pre-computed permutation null distribution if saved
    # (run_validation.py saves summary stats, not full distribution)
    # Re-run permutation for figures (500 shuffles — enough for visualization)
    print("  Re-running permutation (500 shuffles for figure)...")
else:
    print("  validation_results.csv not found — run run_validation.py first")

np.random.seed(99)
N_PERM = 500
perm_data = data.dropna(subset=["oi_shift","analyst_tone","stress_index",
                                  "roa","lnmve","bm","is_market_hours",
                                  "log_during_n_trades","is_2023"]).copy()

obs_tone   = smf.ols(f"oi_shift ~ analyst_tone + {CTRL}",
                      data=perm_data).fit(cov_type="HC3").params["analyst_tone"]
obs_stress = smf.ols(f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
                      data=perm_data).fit(cov_type="HC3").params["stress_index"]

perm_t, perm_s = [], []
for _ in range(N_PERM):
    shuf = perm_data.copy()
    shuf["analyst_tone"] = np.random.permutation(perm_data["analyst_tone"].values)
    shuf["stress_index"] = np.random.permutation(perm_data["stress_index"].values)
    try:
        perm_t.append(smf.ols(f"oi_shift ~ analyst_tone + {CTRL}",
                               data=shuf).fit().params["analyst_tone"])
        perm_s.append(smf.ols(f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
                               data=shuf).fit().params["stress_index"])
    except Exception:
        pass

# Figure 5.5
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, null, obs, color, label in [
    (axes[0], perm_t, obs_tone,   "#2166ac", "Analyst Tone"),
    (axes[1], perm_s, obs_stress, "#d6604d", "Stress Index"),
]:
    p_val = np.mean(np.array(null) <= obs)
    ax.hist(null, bins=30, color="#cccccc", edgecolor="white", density=True)
    ax.axvline(obs, color=color, linewidth=2,
               label=f"True: {obs:.3f} (p={p_val:.3f})")
    ax.set_xlabel("Permuted coefficient")
    ax.set_ylabel("Density")
    ax.set_title(label)
    ax.legend(framealpha=0.9)

fig.suptitle("Figure 5.5: Permutation Test Null Distributions vs True Coefficients", y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / "fig5_5_permutation.pdf", bbox_inches="tight")
plt.close()
print("  Saved: fig5_5_permutation.pdf")

# Figure 5.6: Bootstrap CIs
print("  Running bootstrap (500 resamples for figure)...")
np.random.seed(77)
N_BOOT = 500
boot_t, boot_s = [], []
for _ in range(N_BOOT):
    b = perm_data.sample(n=len(perm_data), replace=True)
    try:
        boot_t.append(smf.ols(f"oi_shift ~ analyst_tone + {CTRL}",
                               data=b).fit().params["analyst_tone"])
        boot_s.append(smf.ols(f"oi_shift ~ stress_index + analyst_tone + {CTRL}",
                               data=b).fit().params["stress_index"])
    except Exception:
        pass

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, boot, obs, color, label, ci_level in [
    (axes[0], boot_t, obs_tone,   "#2166ac", "Analyst Tone",  [5, 95]),
    (axes[1], boot_s, obs_stress, "#d6604d", "Stress Index",  [2.5, 97.5]),
]:
    arr  = np.array(boot)
    bias = arr.mean() - obs
    ci   = np.percentile(arr, ci_level) - bias
    ax.hist(arr, bins=30, color="#cccccc", edgecolor="white", density=True)
    ax.axvline(obs, color=color, linewidth=2, label=f"Estimate: {obs:.3f}")
    ax.axvline(ci[0], color=color, linewidth=1.5, linestyle="--",
               label=f"BC CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.axvline(ci[1], color=color, linewidth=1.5, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Bootstrap coefficient")
    ax.set_ylabel("Density")
    ax.set_title(label)
    ax.legend(framealpha=0.9, fontsize=9)

fig.suptitle("Figure 5.6: Bias-Corrected Bootstrap Confidence Intervals", y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / "fig5_6_bootstrap_ci.pdf", bbox_inches="tight")
plt.close()
print("  Saved: fig5_6_bootstrap_ci.pdf")

# ── Figure 5.7: SE specifications ─────────────────────────────────────────────
print("Generating Figure 5.7...")

se_specs = [
    ("OLS",           {"cov_type": "nonrobust"}),
    ("HC3",           {"cov_type": "HC3"}),
    ("Cluster(firm)", {"cov_type": "cluster",
                       "cov_kwds": {"groups": perm_data["ticker"]}
                       if "ticker" in perm_data.columns
                       else {}}),
    ("NW(4)",         {"cov_type": "HAC", "cov_kwds": {"maxlags": 4}}),
]

# Firm FE and Quarter FE need special handling
perm_data2 = perm_data.copy()
if "ticker" not in perm_data2.columns:
    perm_data2["ticker"] = df.loc[perm_data2.index, "ticker"]
perm_data2["qtr"] = perm_data2["quarter"].str.split("_").str[0]

se_labels, se_coefs, se_lo, se_hi, se_ps = [], [], [], [], []
for spec_name, fit_kw in se_specs:
    try:
        formula = f"oi_shift ~ analyst_tone + stress_index + {CTRL}"
        m = smf.ols(formula, data=perm_data2).fit(**fit_kw)
        c  = m.params["analyst_tone"]
        se = m.bse["analyst_tone"]
        p  = m.pvalues["analyst_tone"]
        se_labels.append(spec_name)
        se_coefs.append(c); se_ps.append(p)
        se_lo.append(c - 1.96*se); se_hi.append(c + 1.96*se)
    except Exception as e:
        print(f"  {spec_name}: {e}")

# Firm FE and Quarter FE
for fe_name, fe_formula in [
    ("Firm FE",    f"oi_shift ~ analyst_tone + stress_index + {CTRL} + C(ticker)"),
    ("Quarter FE", f"oi_shift ~ analyst_tone + stress_index + {CTRL} + C(qtr)"),
]:
    try:
        m = smf.ols(fe_formula, data=perm_data2).fit(cov_type="HC3")
        c  = m.params["analyst_tone"]
        se = m.bse["analyst_tone"]
        p  = m.pvalues["analyst_tone"]
        se_labels.append(fe_name)
        se_coefs.append(c); se_ps.append(p)
        se_lo.append(c - 1.96*se); se_hi.append(c + 1.96*se)
    except Exception as e:
        print(f"  {fe_name}: {e}")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(se_labels))
colors_se = ["#d6604d" if p > 0.10 else "#f4a582" if p > 0.05 else "#2166ac"
             for p in se_ps]
ax.scatter(x, se_coefs, color=colors_se, s=80, zorder=3)
ax.errorbar(x, se_coefs,
            yerr=[np.array(se_coefs)-np.array(se_lo),
                  np.array(se_hi)-np.array(se_coefs)],
            fmt="none", color="gray", capsize=5, alpha=0.7)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x); ax.set_xticklabels(se_labels, rotation=20, ha="right")
ax.set_ylabel("Analyst Tone Coefficient")
ax.set_title("Figure 5.7: Analyst Tone Coefficient Across SE Specifications", pad=12)
for xi, (c, p) in enumerate(zip(se_coefs, se_ps)):
    ax.text(xi, c + 0.01, f"p={p:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(fig_dir / "fig5_7_se_specs.pdf")
plt.close()
print("  Saved: fig5_7_se_specs.pdf")

print(f"\nAll figures saved to: {fig_dir}")
print("Files:")
for f in sorted(fig_dir.glob("fig5_*.pdf")):
    print(f"  {f.name}")
