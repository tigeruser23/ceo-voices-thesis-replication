
#!/usr/bin/env python3

"""

rebuild_master_v2.py

Merge all sources into analysis_dataset_MASTER.parquet.

"""

import pandas as pd, numpy as np

from pathlib import Path

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



base = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")



oi_q1q2 = pd.read_csv(base / "taq/synchronized_q1q2_2022.csv")

oi_q3q4 = pd.read_csv(base / "synchronized/full_synchronized.csv")

oi_2023 = pd.read_csv(base / "taq_2023_oi_patch.csv")



for df in [oi_q1q2, oi_q3q4, oi_2023]:

    if 'oi_shift_fixed' in df.columns:

        df.rename(columns={'oi_shift_fixed':'oi_shift',

                           'during_n_trades_fixed':'during_n_trades',

                           'is_market_hours_fixed':'is_market_hours'},

                  inplace=True)



oi = (pd.concat([oi_q1q2, oi_q3q4, oi_2023], ignore_index=True)

        .drop_duplicates(subset=['ticker','quarter'], keep='last'))



audio      = pd.read_csv(base / "audio_features/audio_features_all308.csv")

audio_cols = [c for c in audio.columns if c not in ['ticker','quarter']]

audio[audio_cols] = StandardScaler().fit_transform(audio[audio_cols])

audio.rename(columns={c: f"audio_{c}" for c in audio_cols}, inplace=True)

audio_cols = [f"audio_{c}" for c in audio_cols]



def fc(patterns):

    for p in patterns:

        cols = [c for c in audio.columns if p in c]

        if cols: return cols[0]



jit  = fc(['jitterLocal_sma3nz_amean', 'jitter'])

shim = fc(['shimmerLocaldB_sma3nz_amean', 'shimmer'])

hnr  = fc(['HNRdBACF_sma3nz_amean', 'HNR'])

f0   = fc(['F0semitoneFrom27.5Hz_sma3nz_amean', 'F0'])



if all([jit, shim, hnr]):

    audio['stress_index'] = audio[jit] + audio[shim] - audio[hnr]

for col, alias in [(jit,'z_jitter'),(shim,'z_shimmer'),(hnr,'z_HNR'),(f0,'z_F0')]:

    if col: audio[alias] = audio[col]



pcs = PCA(n_components=8, random_state=42).fit_transform(audio[audio_cols].fillna(0))

for i in range(8):

    audio[f'PC{i+1}'] = pcs[:, i]



tone = pd.read_csv(base / "finbert/finbert_tone_results_all.csv")[

           ['ticker','quarter','analyst_tone']]

ctrl = pd.read_csv(base / "financial_controls_all.csv")

ctrl['lnmve'] = np.log(ctrl['mve'].clip(lower=0.001))



df = oi.merge(audio, on=['ticker','quarter'], how='left')

df = df.merge(tone,  on=['ticker','quarter'], how='left')

df = df.merge(ctrl,  on=['ticker','quarter'], how='left')



df['log_during_n_trades'] = np.log1p(df['during_n_trades'].fillna(0))

df['year']    = df['quarter'].str.extract(r'_(\d{4})').astype(float)

df['is_2023'] = (df['year'] == 2023).astype(int)



for col in ['oi_shift','analyst_tone','stress_index','roa','lnmve','bm']:

    if col in df.columns:

        lo, hi = df[col].quantile([0.01, 0.99])

        df[col] = df[col].clip(lo, hi)



df.to_parquet(base / "analysis_dataset_MASTER.parquet", index=False)

complete = df.dropna(subset=['oi_shift','analyst_tone','stress_index','roa']).shape[0]

print(f"MASTER: {df.shape[0]} rows x {df.shape[1]} cols | complete: {complete}")

