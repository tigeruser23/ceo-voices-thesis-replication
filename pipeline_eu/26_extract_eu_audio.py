#!/usr/bin/env python3
"""
26_extract_eu_audio.py
Extract 88-dimensional eGeMAPS v02 acoustic features from EU ADR
earnings call MP3s using OpenSMILE.

Auto-discovers all .mp3 files from audio/europe/processed/.
Skips files already extracted (skip-existing guard).
Produces one CSV per call, then consolidates to a single output file.

Output: data/audio_features/europe/eu_audio_features_all.csv

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang 
# Advisor: Daniel Rigobon
"""

import opensmile
import pandas as pd
import os
from pathlib import Path

data_dir  = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
audio_dir = data_dir / "audio"    / "europe" / "processed"
out_dir   = data_dir / "audio_features" / "europe"
out_dir.mkdir(parents=True, exist_ok=True)

consolidated_path = out_dir / "eu_audio_features_all.csv"

#  OpenSMILE configuration 
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

#  Load already-extracted files for skip-existing 
existing_keys = set()
if consolidated_path.exists():
    existing = pd.read_csv(consolidated_path)
    existing_keys = set(
        zip(existing["ticker"], existing["quarter"])
    )
    print(f"Found {len(existing_keys)} already-extracted EU calls. Skipping.")

#  Extract 
results, errors = [], []

mp3_files = sorted(audio_dir.glob("*.mp3"))
print(f"Found {len(mp3_files)} EU MP3 files to process.")

for mp3_path in mp3_files:
    stem  = mp3_path.stem          # e.g. ABB_Q1_2022
    parts = stem.split("_")
    if len(parts) < 3:
        print(f"  SKIP malformed filename: {stem}")
        continue

    ticker  = parts[0]
    quarter = "_".join(parts[1:])  # e.g. Q1_2022

    if (ticker, quarter) in existing_keys:
        continue

    try:
        feats = smile.process_file(str(mp3_path))
        row   = {"ticker": ticker, "quarter": quarter}
        row.update(feats.iloc[0].to_dict())
        results.append(row)
        print(f"  OK  {ticker} {quarter}  ({len(feats.columns)} features)")
    except Exception as e:
        errors.append({"file": stem, "error": str(e)})
        print(f"  ERR {stem}: {e}")

#  Consolidate with any previously extracted results 
new_df = pd.DataFrame(results)

if consolidated_path.exists() and len(results) > 0:
    old_df = pd.read_csv(consolidated_path)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "quarter"], keep="last")
elif consolidated_path.exists():
    combined = pd.read_csv(consolidated_path)
else:
    combined = new_df

combined.to_csv(consolidated_path, index=False)

#  Summary 
n_features = combined.shape[1] - 2 
print(f"\nExtracted: {len(results)} new calls")
print(f"Total EU calls in file: {len(combined)}")
print(f"eGeMAPS features: {n_features}")
print(f"Errors: {len(errors)}")
if errors:
    for e in errors:
        print(f"  {e['file']}: {e['error']}")
