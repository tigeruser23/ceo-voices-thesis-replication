#!/usr/bin/env python3
"""
run_vix_interaction.py
VIX-Continuous Interaction Robustness Check.

Replaces the binary 2022/2023 regime split with a continuous VIX
interaction to verify that the regime result is not an artifact of
ad hoc year discretisation (see appendix methods table, Section sec:vix).

Specifications:
  V1: oi_shift ~ analyst_tone + vix_call + controls
      (VIX main effect only; confirms no mechanical collinearity)
  V2: oi_shift ~ analyst_tone * vix_call + controls
      (interaction with raw VIX level)
  V3: oi_shift ~ analyst_tone * vix_z + controls
      (interaction with standardised VIX; coefficient interpretable
       as effect of 1-SD increase in uncertainty)
  V4: oi_shift ~ analyst_tone * vix_z + is_2023 + controls_no_yr
      (standardised VIX interaction WITH explicit year dummy but
       WITHOUT year dummy inside CTRL, to test whether continuous VIX
       and discrete year dummy are jointly significant or collinear)

  NOTE: V4 differs from V3 by using CTRL_NO2023 + explicit is_2023.
  V3 uses CTRL (which already includes is_2023). V4 lets the year 
  dummy coefficient to be read separately from the VIX interaction 
  coefficient.

VIX DATA SOURCE:
  vix_daily.csv must be downloaded from FRED before running:
    Series: VIXCLS (CBOE Volatility Index, daily close)
    URL: https://fred.stlouisfed.org/series/VIXCLS
    Download as CSV; rename columns to: date, vix
    Save to: data/vix_daily.csv

Outputs:
  tables_v2/vix_interaction_results.csv
  tables_v2/vix_interaction_table.tex

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang
# Advisor: Daniel Rigobon
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
TABS = BASE / "tables_v2"
TABS.mkdir(exist_ok=True)

#  Load data 
df  = pd.read_parquet(BASE / "analysis_dataset_MASTER.parquet")
vix = pd.read_csv(BASE / "vix_daily.csv", parse_dates=["date"])
vix["date"] = pd.to_datetime(vix["date"])
vix["vix"]  = pd.to_numeric(vix["vix"], errors="coerce")
vix = vix.dropna(subset=["vix"]).sort_values("date").reset_index(drop=True)

print(f"Master dataset:  {len(df)} rows")
print(f"VIX data:        {len(vix)} trading days "
      f"({vix['date'].min().date()} – {vix['date'].max().date()})")

#  Merge VIX onto call dates 
def quarter_to_approx_date(quarter: str) -> pd.Timestamp:
    """Approximate call date as midpoint of earnings season."""
    qmap = {"Q1": "05-01", "Q2": "08-01", "Q3": "11-01", "Q4": "02-15"}
    parts = quarter.split("_")
    q, yr = parts[0], int(parts[1])
    mo_day = qmap[q]
    yr = yr + 1 if q == "Q4" else yr
    return pd.Timestamp(f"{yr}-{mo_day}")

if "call_datetime_et" in df.columns:
    df["call_date_dt"] = pd.to_datetime(
        df["call_datetime_et"], errors="coerce").dt.normalize()
    mask = df["call_date_dt"].isna()
    df.loc[mask, "call_date_dt"] = df.loc[mask, "quarter"].apply(
        quarter_to_approx_date)
else:
    df["call_date_dt"] = df["quarter"].apply(quarter_to_approx_date)

df_sorted = df.sort_values("call_date_dt").reset_index(drop=True)
merged = pd.merge_asof(
    df_sorted,
    vix.rename(columns={"date": "call_date_dt", "vix": "vix_call"}),
    on="call_date_dt",
    direction="backward",
    tolerance=pd.Timedelta("5 days"),
)

n_matched = merged["vix_call"].notna().sum()
print(f"VIX matched:     {n_matched} / {len(merged)} calls")

vix_mean = merged["vix_call"].mean()
vix_std  = merged["vix_call"].std()
merged["vix_z"] = (merged["vix_call"] - vix_mean) / vix_std
print(f"VIX stats:       mean={vix_mean:.1f}  SD={vix_std:.1f}  "
      f"min={merged['vix_call'].min():.1f}  max={merged['vix_call'].max():.1f}")

#  Control vectors 
CTRL        = "roa + lnmve + bm + is_market_hours + log_during_n_trades + is_2023"
CTRL_NO2023 = "roa + lnmve + bm + is_market_hours + log_during_n_trades"

def stars(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""

def run_ols(formula: str, data: pd.DataFrame, cov: str = "HC3"):
    needed = [v.strip() for v in
              formula.replace("~"," ").replace("+"," ").replace("*"," ")
                     .replace(":"," ").split()
              if v.strip() and not v.strip()[0].isdigit()]
    d = data.dropna(subset=[c for c in needed if c in data.columns])
    if len(d) < 20:
        return None, 0
    try:
        return smf.ols(formula, data=d).fit(cov_type=cov), len(d)
    except Exception as e:
        print(f"  Model failed: {e}")
        return None, 0

#  Specifications 
specs = {
    # V1: VIX main effect only (no interaction)
    "V1": (f"oi_shift ~ analyst_tone + vix_call + {CTRL}", merged),
    # V2: Interaction with raw VIX
    "V2": (f"oi_shift ~ analyst_tone * vix_call + {CTRL}", merged),
    # V3: Interaction with standardised VIX (CTRL includes is_2023)
    "V3": (f"oi_shift ~ analyst_tone * vix_z + {CTRL}", merged),
    # V4: Standardised VIX + explicit is_2023 WITHOUT is_2023 buried in CTRL.
    #     FIX: V3 and V4 were identical in prior version (both used CTRL).
    #     V4 now uses CTRL_NO2023 + explicit is_2023 to pit continuous VIX
    #     against discrete year dummy in the same specification.
    "V4": (f"oi_shift ~ analyst_tone * vix_z + is_2023 + {CTRL_NO2023}", merged),
}

print("VIX INTERACTION REGRESSIONS")

results = {}
for label, (formula, data) in specs.items():
    m, n = run_ols(formula, data)
    if m is None:
        print(f"  {label}: FAILED")
        continue
    results[label] = (m, n)

    tone_c = m.params.get("analyst_tone", np.nan)
    tone_p = m.pvalues.get("analyst_tone", np.nan)
    vix_c  = m.params.get("vix_call", m.params.get("vix_z", np.nan))
    vix_p  = m.pvalues.get("vix_call", m.pvalues.get("vix_z", np.nan))
    inter  = next((k for k in m.params.index if ":" in k), None)
    int_c  = m.params.get(inter, np.nan) if inter else np.nan
    int_p  = m.pvalues.get(inter, np.nan) if inter else np.nan
    yr_c   = m.params.get("is_2023", np.nan)
    yr_p   = m.pvalues.get("is_2023", np.nan)

    print(f"\n{label}  (N={n}  R²={m.rsquared:.4f})")
    print(f"  analyst_tone    {tone_c:+.4f}  p={tone_p:.4f} {stars(tone_p)}")
    if not np.isnan(vix_c):
        print(f"  vix             {vix_c:+.4f}  p={vix_p:.4f} {stars(vix_p)}")
    if inter:
        print(f"  tone × vix      {int_c:+.4f}  p={int_p:.4f} {stars(int_p)}")
    if not np.isnan(yr_c):
        print(f"  is_2023         {yr_c:+.4f}  p={yr_p:.4f} {stars(yr_p)}")

    # Economic interpretation at low/high VIX
    if inter and not np.isnan(int_c):
        for pctile, label_p in [(0.10, "10th"), (0.90, "90th")]:
            vz    = merged["viz_z"].quantile(pctile) if "vix_z" in merged else np.nan
            if np.isnan(vz):
                vz = merged["vix_z"].quantile(pctile)
            vraw  = vz * vix_std + vix_mean
            te    = tone_c + int_c * vz
            print(f"  Implied tone @ VIX={vraw:.0f} ({label_p} pctile): {te:+.4f}")

#  Save CSV 
rows = []
for label, (m, n) in results.items():
    for var in m.params.index:
        rows.append({
            "spec": label, "variable": var,
            "coef": m.params[var], "se": m.bse[var],
            "pval": m.pvalues[var], "sig": stars(m.pvalues[var]),
            "n": n, "r2": m.rsquared,
        })
pd.DataFrame(rows).to_csv(TABS / "vix_interaction_results.csv", index=False)

# LaTeX table 
def fmt_coef(coef, se, p):
    s = stars(p)
    sup = f"^{{{s}}}" if s else ""
    return f"${coef:+.3f}{sup}$", f"$({se:.3f})$"

present = [l for l in ["V1","V2","V3","V4"] if l in results]
col_labels = {
    "V1": "Tone+VIX",
    "V2": r"Tone$\times$VIX",
    "V3": r"Tone$\times$VIX$_z$",
    "V4": r"Tone$\times$VIX$_z$+yr",
}

tex = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{VIX-Continuous Interaction: Robustness to Continuous Macro Uncertainty}",
    r"\label{tab:vix_interaction}",
    r"\begin{threeparttable}",
    r"\small",
    r"\begin{tabular}{l" + "c" * len(present) + "}",
    r"\toprule",
    " & " + " & ".join(present) + r" \\",
    " & " + " & ".join(col_labels[l] for l in present) + r" \\",
    r"\midrule",
]

for var, label_tex in [
    ("analyst_tone", "Analyst Tone"),
    ("vix_call",     "VIX"),
    ("vix_z",        r"VIX (standardised)"),
    ("is_2023",      "2023 dummy"),
]:
    coef_cells, se_cells, any_val = [], [], False
    for l in present:
        m, _ = results[l]
        if var in m.params.index:
            c, s = fmt_coef(m.params[var], m.bse[var], m.pvalues[var])
            coef_cells.append(c); se_cells.append(s); any_val = True
        else:
            coef_cells.append("---"); se_cells.append("")
    if any_val:
        tex += [" & ".join([label_tex] + coef_cells) + r" \\",
                " & ".join([""] + se_cells) + r" \\",
                r"\addlinespace"]

# Interaction row
inter_cells = []
for l in present:
    m, _ = results[l]
    inter = next((k for k in m.params.index if ":" in k), None)
    if inter:
        c, _ = fmt_coef(m.params[inter], m.bse[inter], m.pvalues[inter])
        inter_cells.append(c)
    else:
        inter_cells.append("---")
tex += [r"Tone $\times$ VIX & " + " & ".join(inter_cells) + r" \\",
        r"\addlinespace",
        "Controls & " + " & ".join(["incl."] * len(present)) + r" \\",
        r"$R^2$ & " + " & ".join(
            f"{results[l][0].rsquared:.4f}" for l in present) + r" \\",
        r"$N$ & " + " & ".join(
            str(results[l][1]) for l in present) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item HC3 robust SEs in parentheses. "
        r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
        rf"VIX$_z$ standardised (mean={vix_mean:.1f}, SD={vix_std:.1f}). "
        r"V4 includes the binary 2023 dummy alongside the continuous VIX "
        r"interaction to test collinearity between the two uncertainty proxies. "
        r"VIX data: CBOE Volatility Index (VIXCLS) from FRED.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
]

tex_out = TABS / "vix_interaction_table.tex"
tex_out.write_text("\n".join(tex))

print(f"\nSaved:")
print(f"  {TABS / 'vix_interaction_results.csv'}")
print(f"  {tex_out}")

m_v3 = results.get("V3")
if m_v3:
    m3, _ = m_v3
    inter = next((k for k in m3.params.index if ":" in k), None)
    if inter:
        c = m3.params[inter]; p = m3.pvalues[inter]
        direction = "amplifies" if c < 0 else "attenuates"
        print(f"\nKey result (V3 Tone×VIX_z): β={c:+.4f}  p={p:.4f} {stars(p)}")
        print(f"  1-SD increase in VIX {direction} the tone→OI channel by {abs(c):.4f}")
