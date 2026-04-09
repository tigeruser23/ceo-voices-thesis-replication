#!/usr/bin/env python3
"""
rebuild_master_v2.py
Merge all US sources into analysis_dataset_MASTER.parquet.

Merges three OI sources (Q1/Q2 2022, Q3/Q4 2022, 2023 timezone-corrected),
audio features, FinBERT tone, and financial controls. Performs
standardisation, PCA (PC1-PC8), winsorisation at 1st/99th percentile,
and derives all regression variables.

Output: data/analysis_dataset_MASTER.parquet  (310 x 156; ~282 complete)

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang 
# Advisor: Daniel Rigobon
"""

import os                                      
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

# Order imbalance: merge three sources
oi_q1q2 = pd.read_csv(base / "taq" / "synchronized_q1q2_2022.csv")
oi_q3q4 = pd.read_csv(base / "synchronized" / "full_synchronized.csv")
oi_2023 = pd.read_csv(base / "taq_2023_oi_patch.csv")

for df in [oi_q1q2, oi_q3q4, oi_2023]:
    if "oi_shift_fixed" in df.columns:
        df.rename(columns={
            "oi_shift_fixed":         "oi_shift",
            "during_n_trades_fixed":  "during_n_trades",
            "is_market_hours_fixed":  "is_market_hours",
        }, inplace=True)

oi = (pd.concat([oi_q1q2, oi_q3q4, oi_2023], ignore_index=True)
        .drop_duplicates(subset=["ticker", "quarter"], keep="last"))

print(f"OI rows (deduplicated): {len(oi)}")

#  Audio features 
audio      = pd.read_csv(base / "audio_features" / "audio_features_all308.csv")
audio_cols = [c for c in audio.columns if c not in ["ticker", "quarter"]]

audio[audio_cols] = StandardScaler().fit_transform(audio[audio_cols])
audio.rename(columns={c: f"audio_{c}" for c in audio_cols}, inplace=True)
audio_cols = [f"audio_{c}" for c in audio_cols]

def find_col(patterns, cols):
    for p in patterns:
        matches = [c for c in cols if p in c]
        if matches:
            return matches[0]
    return None

jit  = find_col(["jitterLocal_sma3nz_amean", "jitter"],     audio_cols)
shim = find_col(["shimmerLocaldB_sma3nz_amean", "shimmer"],  audio_cols)
hnr  = find_col(["HNRdBACF_sma3nz_amean", "HNR"],           audio_cols)
f0   = find_col(["F0semitoneFrom27.5Hz_sma3nz_amean", "F0"], audio_cols)

if all([jit, shim, hnr]):
    audio["stress_index"] = audio[jit] + audio[shim] - audio[hnr]

for col, alias in [(jit,  "z_jitter"),
                   (shim, "z_shimmer"),
                   (hnr,  "z_HNR"),
                   (f0,   "z_F0")]:
    if col:
        audio[alias] = audio[col]

pcs = PCA(n_components=8, random_state=42).fit_transform(
    audio[audio_cols].fillna(0))
for i in range(8):
    audio[f"PC{i+1}"] = pcs[:, i]

#  FinBERT tone 
tone = pd.read_csv(
    base / "finbert" / "finbert_tone_results_all.csv"
)[["ticker", "quarter", "analyst_tone"]]

#  Financial controls 
ctrl = pd.read_csv(base / "financial_controls_all.csv")
ctrl["lnmve"] = np.log(ctrl["mve"].clip(lower=0.001))

#  Merge 
df = oi.merge(audio, on=["ticker", "quarter"], how="left")
df = df.merge(tone,  on=["ticker", "quarter"], how="left")
df = df.merge(ctrl,  on=["ticker", "quarter"], how="left")

df["log_during_n_trades"] = np.log1p(df["during_n_trades"].fillna(0))
df["year"]    = df["quarter"].str.extract(r"_(\d{4})").astype(float)
df["is_2023"] = (df["year"] == 2023).astype(int)

#  Winsorise 
for col in ["oi_shift", "analyst_tone", "stress_index", "roa", "lnmve", "bm"]:
    if col in df.columns and df[col].notna().sum() > 10:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

#  Save 
df.to_parquet(base / "analysis_dataset_MASTER.parquet", index=False)

complete = df.dropna(subset=["oi_shift", "analyst_tone",
                              "stress_index", "roa"]).shape[0]
print(f"MASTER: {df.shape[0]} rows x {df.shape[1]} cols | "
      f"complete cases: {complete}")
