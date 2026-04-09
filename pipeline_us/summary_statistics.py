#!/usr/bin/env python3
"""
summary_statistics.py
Generate Table 3.4: Summary Statistics by Sample.

Three panels matching the thesis exactly:
  Panel A: US sample (40 firms, Q1 2022–Q4 2023)
  Panel B: European ADR sample (23 firms, open-reaction window)
  Panel C: Pooled US + EU

Variables reported: N, Mean, SD, P25, Median, P75.
Matches the exact variable list and sample sizes in Table 3.4.

Output: tables_v2/summary_statistics.csv
        tables_v2/summary_statistics_latex.tex

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang, Princeton Senior Thesis 
# Advisor: Daniel Rigobon
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
tabs = base / "tables_v2"
tabs.mkdir(exist_ok=True)

us  = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")
gl  = pd.read_parquet(base / "analysis_dataset_GLOBAL.parquet")
eu  = gl[gl["is_eu"] == 1].copy()

def describe_var(series, label):
    """Compute summary statistics for one variable."""
    s = series.dropna()
    return {
        "variable": label,
        "n":        len(s),
        "mean":     round(s.mean(), 3),
        "sd":       round(s.std(), 3),
        "p25":      round(s.quantile(0.25), 3),
        "median":   round(s.median(), 3),
        "p75":      round(s.quantile(0.75), 3),
    }

rows = []

#  Panel A: US Sample 
panel_a_vars = [
    ("oi_shift",            "OI Shift"),
    ("analyst_tone",        "Analyst Tone (FinBERT)"),
    ("stress_index",        "Stress Index"),
    ("z_jitter",            "z-Jitter"),
    ("z_shimmer",           "z-Shimmer"),
    ("z_HNR",               "z-HNR"),
    ("z_F0",                "z-F0"),
    ("roa",                 "ROA"),
    ("lnmve",               "ln(MVE)"),
    ("bm",                  "Book-to-Market"),
    ("log_during_n_trades", "ln(Trade Count)"),
    ("is_market_hours",     "Market Hours (=1)"),
    ("is_2023",             "2023 Indicator"),
]

print("Panel A: US Sample")
for col, label in panel_a_vars:
    if col in us.columns:
        r = describe_var(us[col], label)
        r["panel"] = "A: US Sample"
        rows.append(r)
        print(f"  {label:<35} N={r['n']:3d}  "
              f"Mean={r['mean']:7.3f}  SD={r['sd']:7.3f}  "
              f"P25={r['p25']:7.3f}  Med={r['median']:7.3f}  P75={r['p75']:7.3f}")
    else:
        print(f"  {label:<35} MISSING COLUMN: {col}")

#  Panel B: European ADR Sample 
panel_b_vars = [
    ("oi_shift",            "OI Shift (open-reaction window)"),
    ("analyst_tone",        "Analyst Tone (FinBERT)"),
    ("stress_index",        "Stress Index (eGeMAPS)"),
    ("during_n_trades",     "During-window N trades"),
]

print("\nPanel B: European ADR Sample")
print("-" * 70)
for col, label in panel_b_vars:
    if col in eu.columns:
        r = describe_var(eu[col], label)
        r["panel"] = "B: EU Sample"
        rows.append(r)
        print(f"  {label:<35} N={r['n']:3d}  "
              f"Mean={r['mean']:9.3f}  SD={r['sd']:9.3f}  "
              f"Med={r['median']:9.3f}")
    else:
        print(f"  {label:<35} MISSING COLUMN: {col}")

#  Panel C: Pooled US + EU 
panel_c_vars = [
    ("oi_shift",     "OI Shift"),
    ("analyst_tone", "Analyst Tone"),
]

print("\nPanel C: Pooled US + EU")
print("-" * 70)
for col, label in panel_c_vars:
    if col in gl.columns:
        r = describe_var(gl[col], label)
        r["panel"] = "C: Pooled"
        rows.append(r)
        print(f"  {label:<35} N={r['n']:3d}  "
              f"Mean={r['mean']:7.3f}  SD={r['sd']:7.3f}  "
              f"Med={r['median']:7.3f}")

#  Save CSV 
stats_df = pd.DataFrame(rows)[
    ["panel","variable","n","mean","sd","p25","median","p75"]
]
stats_df.to_csv(tabs / "summary_statistics.csv", index=False)
print(f"\nSaved: summary_statistics.csv  ({len(stats_df)} rows)")

tex = [
    r"\begin{table}[H]",
    r"\centering",
    r"\caption{Summary Statistics by Sample}",
    r"\label{tab:sumstats}",
    r"\small",
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Variable & $N$ & Mean & SD & P25 & Median & P75 \\",
    r"\midrule",
    r"\multicolumn{7}{l}{\textit{Panel A: US Sample (during-call window)}} \\",
    r"\addlinespace",
]

for r in rows:
    if r["panel"] != "A: US Sample":
        continue
    p25 = f"{r['p25']:.3f}" if not (r['p25'] == 0 and r['p75'] == 0) else "0"
    p75 = f"{r['p75']:.3f}" if not (r['p25'] == 0 and r['p75'] == 0) else "0"
    tex.append(
        f"  {r['variable']} & {r['n']} & {r['mean']:.3f} & "
        f"{r['sd']:.3f} & {p25} & {r['median']:.3f} & {p75} \\\\"
    )

tex += [
    r"\addlinespace",
    r"\multicolumn{7}{l}{\textit{Panel B: European ADR Sample}} \\",
    r"\addlinespace",
]
for r in rows:
    if r["panel"] != "B: EU Sample":
        continue
    tex.append(
        f"  {r['variable']} & {r['n']} & {r['mean']:.3f} & "
        f"{r['sd']:.3f} & {r['p25']:.3f} & {r['median']:.3f} & {r['p75']:.3f} \\\\"
    )

tex += [
    r"\addlinespace",
    r"\multicolumn{7}{l}{\textit{Panel C: Pooled US + EU}} \\",
    r"\addlinespace",
]
for r in rows:
    if r["panel"] != "C: Pooled":
        continue
    tex.append(
        f"  {r['variable']} & {r['n']} & {r['mean']:.3f} & "
        f"{r['sd']:.3f} & {r['p25']:.3f} & {r['median']:.3f} & {r['p75']:.3f} \\\\"
    )

tex += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

tex_path = tabs / "summary_statistics_latex.tex"
tex_path.write_text("\n".join(tex))
print(f"Saved: summary_statistics_latex.tex")
