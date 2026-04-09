#!/usr/bin/env python3
"""
rebuild_master_global_v2.py
Assemble the global dataset: US MASTER + EU ADR extension.

Steps:
  1. Load validated US MASTER (analysis_dataset_MASTER.parquet)
  2. Load EU OI (eu_adr_synchronized.csv)
  3. Load EU audio features; standardise on combined US+EU sample
  4. Load EU FinBERT tone (finbert_tone_eu.csv)
  5. Attach CRSP-based EU financial controls (lnmve only;
     ROA and B/M unavailable via WRDS for EU firms)
  6. Construct interaction columns:
       is_eu, is_2023, tone_x_eu, tone_x_2023, tone_x_eu_x_2023
  7. Write analysis_dataset_GLOBAL.parquet (US + EU)
     and analysis_dataset_EU_only.parquet (EU subset)

Reduced control vector for pooled models:
  lnmve + is_market_hours + log_during_n_trades + is_2023
  (ROA and B/M omitted: unavailable for EU via WRDS)

Clean rewrite (v2): no iterative patches from prior runs.

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang 
# Advisor: Daniel Rigobon
"""

import pandas as pd
import numpy as np
import os
import wrds
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

# ── 1. Load US MASTER ─────────────────────────────────────────────────────────
print("Loading US MASTER...")
us = pd.read_parquet(base / "analysis_dataset_MASTER.parquet")
us["is_eu"] = 0
print(f"  US MASTER: {us.shape[0]} rows x {us.shape[1]} cols")

# ── 2. Load EU OI ─────────────────────────────────────────────────────────────
print("Loading EU OI...")
eu_oi = pd.read_csv(base / "synchronized" / "eu_adr_synchronized.csv")

# Rename to match US column convention
eu_oi = eu_oi.rename(columns={
    "pre_oi":         "pre_30m_order_imbalance",
    "pre_n_trades":   "pre_30m_n_trades",
    "event_oi":       "during_order_imbalance",
    "event_n_trades": "during_n_trades",
})

# Exclude CFR and RIO at assembly stage
EXCLUDE_EU = {"CFR", "RIO"}
eu_oi = eu_oi[~eu_oi["ticker"].isin(EXCLUDE_EU)].copy()
print(f"  EU OI rows (post-exclusion): {len(eu_oi)}")

# ── 3. Load EU audio features ─────────────────────────────────────────────────
print("Loading EU audio features...")
eu_audio_path = base / "audio_features" / "europe" / "eu_audio_features_all.csv"
eu_audio = pd.read_csv(eu_audio_path)
eu_audio = eu_audio[~eu_audio["ticker"].isin(EXCLUDE_EU)].copy()

# Identify the 88 eGeMAPS feature columns
audio_cols = [c for c in eu_audio.columns if c not in ["ticker", "quarter"]]

# Standardise EU audio on combined US+EU means/stds
# Load US audio to get combined distribution
us_audio_path = base / "audio_features" / "audio_features_all308.csv"
us_audio = pd.read_csv(us_audio_path)
us_audio_feat = [c for c in us_audio.columns if c not in ["ticker", "quarter"]]

# Use only columns present in both
common_audio = [c for c in audio_cols if c in us_audio_feat]
print(f"  Common eGeMAPS features: {len(common_audio)}")

combined_audio = pd.concat([
    us_audio[common_audio],
    eu_audio[common_audio]
], ignore_index=True)

scaler = StandardScaler().fit(combined_audio)
eu_audio[common_audio] = scaler.transform(eu_audio[common_audio])
eu_audio.rename(columns={c: f"audio_{c}" for c in common_audio}, inplace=True)
audio_cols_renamed = [f"audio_{c}" for c in common_audio]

# Construct EU stress index from standardised features
def find_col(patterns, cols):
    for p in patterns:
        matches = [c for c in cols if p in c]
        if matches:
            return matches[0]
    return None

jit  = find_col(["jitterLocal_sma3nz_amean", "jitter"],  audio_cols_renamed)
shim = find_col(["shimmerLocaldB_sma3nz_amean", "shimmer"], audio_cols_renamed)
hnr  = find_col(["HNRdBACF_sma3nz_amean", "HNR"],        audio_cols_renamed)
f0   = find_col(["F0semitoneFrom27.5Hz_sma3nz_amean", "F0"], audio_cols_renamed)

if all([jit, shim, hnr]):
    eu_audio["stress_index"] = eu_audio[jit] + eu_audio[shim] - eu_audio[hnr]
    print(f"  Stress index constructed: {jit}, {shim}, {hnr}")
for col, alias in [(jit, "z_jitter"), (shim, "z_shimmer"),
                   (hnr, "z_HNR"), (f0, "z_F0")]:
    if col:
        eu_audio[alias] = eu_audio[col]

# PCA on EU audio (fitted on combined sample for comparability)
us_audio_std = us_audio[common_audio].copy()
us_audio_std_vals = scaler.transform(us_audio_std)
us_audio_std = pd.DataFrame(us_audio_std_vals, columns=common_audio)
us_audio_std.rename(columns={c: f"audio_{c}" for c in common_audio}, inplace=True)

combined_for_pca = pd.concat([
    us_audio_std[audio_cols_renamed].fillna(0),
    eu_audio[audio_cols_renamed].fillna(0)
], ignore_index=True)

pca = PCA(n_components=8, random_state=42).fit(combined_for_pca)
eu_pcs = pca.transform(eu_audio[audio_cols_renamed].fillna(0))
for i in range(8):
    eu_audio[f"PC{i+1}"] = eu_pcs[:, i]

# ── 4. Load EU FinBERT tone ───────────────────────────────────────────────────
print("Loading EU FinBERT tone...")
eu_tone = pd.read_csv(base / "finbert" / "finbert_tone_eu.csv")
eu_tone = eu_tone[~eu_tone["ticker"].isin(EXCLUDE_EU)][
    ["ticker", "quarter", "analyst_tone", "n_sentences"]
].copy()
print(f"  EU tone rows: {len(eu_tone)}  "
      f"non-missing: {eu_tone['analyst_tone'].notna().sum()}")

# ── 5. EU financial controls (CRSP-based lnmve only) ─────────────────────────
print("Fetching EU financial controls from CRSP...")
eu_sample = pd.read_csv(base / "european_adr_sample.csv")
eu_sample = eu_sample[~eu_sample["ticker"].isin(EXCLUDE_EU)]
eu_permnos = eu_sample["permno"].tolist()
permno_str = ", ".join(map(str, eu_permnos))

db = wrds.Connection()
crsp_mkt = db.raw_sql(f"""
    SELECT permno, date, shrout, prc
    FROM crsp.msf
    WHERE permno IN ({permno_str})
      AND date BETWEEN '2021-10-01' AND '2024-03-31'
""")
db.close()

crsp_mkt["mve"] = crsp_mkt["shrout"].abs() * crsp_mkt["prc"].abs() / 1e3
crsp_mkt["date"]   = pd.to_datetime(crsp_mkt["date"])
crsp_mkt["period"] = crsp_mkt["date"].dt.to_period("Q")

# Quarter-end MVE: last observation per permno-quarter
qmve = (crsp_mkt.sort_values("date")
        .groupby(["permno", "period"])["mve"]
        .last()
        .reset_index())

qmve["quarter"] = qmve["period"].astype(str).str.replace("Q", "Q")
# Convert "2022Q1" -> "Q1_2022"
qmve["quarter"] = qmve["period"].apply(
    lambda p: f"Q{p.quarter}_{p.year}")
qmve["lnmve"] = np.log(qmve["mve"].clip(lower=0.001))

eu_ctrl = (qmve.merge(
    eu_sample[["permno", "ticker"]], on="permno")
    [["ticker", "quarter", "lnmve"]])
print(f"  EU controls rows: {len(eu_ctrl)}")

# ── 6. Assemble EU frame ──────────────────────────────────────────────────────
print("Assembling EU frame...")
eu = eu_oi.merge(eu_audio,  on=["ticker", "quarter"], how="left")
eu = eu.merge(eu_tone,  on=["ticker", "quarter"], how="left")
eu = eu.merge(eu_ctrl,  on=["ticker", "quarter"], how="left")

eu["log_during_n_trades"] = np.log1p(eu["during_n_trades"].fillna(0))
eu["year"]    = eu["quarter"].str.extract(r"_(\d{4})").astype(float)
eu["is_2023"] = (eu["year"] == 2023).astype(int)

# Open-reaction window: is_market_hours always 1 for EU (09:30-10:30 ET)
eu["is_market_hours"] = 1

# Winsorise key variables at 1/99th percentile
for col in ["oi_shift", "analyst_tone", "stress_index", "lnmve"]:
    if col in eu.columns and eu[col].notna().sum() > 10:
        lo, hi = eu[col].quantile([0.01, 0.99])
        eu[col] = eu[col].clip(lo, hi)

eu["is_eu"] = 1
print(f"  EU frame: {eu.shape[0]} rows x {eu.shape[1]} cols")

# ── 7. Pool US + EU ───────────────────────────────────────────────────────────
print("Pooling US + EU...")

# Align columns: add missing cols as NaN in each frame
all_cols = sorted(set(us.columns) | set(eu.columns))
us_aligned = us.reindex(columns=all_cols)
eu_aligned = eu.reindex(columns=all_cols)

global_df = pd.concat([us_aligned, eu_aligned], ignore_index=True)

# Interaction columns for pooled regressions
global_df["tone_x_eu"]      = global_df["analyst_tone"] * global_df["is_eu"]
global_df["tone_x_2023"]    = global_df["analyst_tone"] * global_df["is_2023"]
global_df["tone_x_eu_x_2023"] = (global_df["analyst_tone"]
                                   * global_df["is_eu"]
                                   * global_df["is_2023"])

print(f"  Global dataset: {global_df.shape[0]} rows x {global_df.shape[1]} cols")

# Reduced-control complete cases
reduced_ctrl = ["oi_shift", "analyst_tone", "lnmve",
                "is_market_hours", "log_during_n_trades", "is_2023"]
n_complete = global_df.dropna(subset=reduced_ctrl).shape[0]
print(f"  Complete (reduced controls): {n_complete}")

# ── 8. Save ───────────────────────────────────────────────────────────────────
global_df.to_parquet(base / "analysis_dataset_GLOBAL.parquet", index=False)
print(f"Saved: analysis_dataset_GLOBAL.parquet "
      f"({global_df.shape[0]} x {global_df.shape[1]})")

eu_only = global_df[global_df["is_eu"] == 1].copy()
eu_only.to_parquet(base / "analysis_dataset_EU_only.parquet", index=False)
print(f"Saved: analysis_dataset_EU_only.parquet "
      f"({eu_only.shape[0]} x {eu_only.shape[1]})")
