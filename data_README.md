# Data Directory Structure

This file documents the expected directory layout under
`/scratch/network/oy3009/thesis_week1/data/`.

All data files are excluded from the GitHub repository due to
WRDS licensing restrictions and file size. This document is the
reference a replicator needs to understand what to expect at
each pipeline stage.

---

## Root-level files

These small metadata files ARE committed to the repo:

```
data/
├── selected_sample_40_FINAL.csv       # 40-firm stratified US sample
├── european_adr_sample.csv            # 23-firm EU ADR sample
├── call_times_extracted.csv           # US call dates/times (from 14_extract_call_times.py)
├── call_times_eu_extracted.csv        # EU call dates (from 14_extract_call_times_eu.py)
├── financial_controls_all.csv         # ROA, MVE, B/M per firm-quarter (from 29_financial_controls.py)
├── vix_daily.csv                      # CBOE VIX daily close from FRED (manual download)
│                                      # Series: VIXCLS, URL: https://fred.stlouisfed.org/series/VIXCLS
│                                      # Columns: date, vix
├── analysis_dataset_MASTER.parquet    # US analysis dataset 310 × 156 (from rebuild_master_v2.py)
├── analysis_dataset_GLOBAL.parquet    # US + EU pooled 490 × 257 (from rebuild_master_global_v2.py)
├── analysis_dataset_EU_only.parquet   # EU subset 196 × 257 (from rebuild_master_global_v2.py)
└── taq_2023_oi_patch.csv              # 2023 OI with timezone correction (from 27_sync_all.py)
```

---

## TAQ data

```
data/taq/
├── synchronized_q1q2_2022.csv         # OI shift for Q1/Q2 2022
├── {TICKER}_{QUARTER}_taq.csv.gz      # US firm-quarter TAQ files (40 firms × 8 qtrs)
│                                      # e.g. AAPL_Q1_2022_taq.csv.gz
└── eu/
    └── {TICKER}_{QUARTER}_taq.parquet # EU firm-quarter TAQ files (23 firms × 8 qtrs)
                                       # e.g. ABB_Q1_2022_taq.parquet
```

---

## Synchronized OI data

```
data/synchronized/
├── full_synchronized.csv              # OI shift for Q3/Q4 2022 (US)
└── eu_adr_synchronized.csv            # OI shift for EU (open-reaction window)
```

---

## Audio files

```
data/audio/
├── raw/                               # Original Refinitiv MP3 files (US)
├── processed/                         # Renamed to TICKER_Qn_YYYY.mp3
└── europe/
    ├── raw/                           # Original Refinitiv EU MP3 files
    │                                  # NOTE: Refinitiv delivers both .mp3 and
    │                                  # .txt files to this same directory
    └── processed/                     # Renamed EU MP3s: TICKER_Qn_YYYY.mp3
```

---

## Transcripts

```
data/transcripts/
├── raw/                               # Original Refinitiv transcript files (US)
│                                      # Format: YYYY-Mon-DD-TICKER.EXCHANGE-ID-Transcript.txt
├── processed/                         # Renamed to TICKER_Qn_YYYY.txt
└── europe/
    ├── raw/                           # Original Refinitiv EU transcripts (and MP3s)
    │                                  # Format: YYYY-Mon-DD-TICKER.EXCHANGE-ID-Transcript.txt
    └── processed/                     # Renamed EU transcripts: TICKER_Qn_YYYY.txt
```

---

## Acoustic features

```
data/audio_features/
├── audio_features_all308.csv          # US eGeMAPS: 307 calls × 90 cols
│                                      # (88 features + ticker + quarter)
└── europe/
    └── eu_audio_features_all.csv      # EU eGeMAPS: ~163 calls × 90 cols
```

---

## FinBERT tone scores

```
data/finbert/
├── finbert_tone_results_all.csv       # US analyst tone: 310 rows
└── finbert_tone_eu.csv                # EU analyst tone: ~184 rows
```

---

## Regression outputs

```
data/tables_v2/
├── regression_results.csv             # M1–M5 full coefficient tables
├── regime_results.csv                 # Regime split + Chow test
├── bh_scan_results.csv                # BH-corrected p-values (88 features)
├── quarter_by_quarter.csv             # Quarter-by-quarter tone coefficients
├── validation_results.csv             # Permutation, bootstrap, wrong-quarter
├── regression_results_global.csv      # G1–G6 cross-market results
├── vix_interaction_results.csv        # V1–V4 VIX robustness results
└── vix_interaction_table.tex          # LaTeX table for VIX robustness
```

---

## Intermediate / scratch files

These are generated during the pipeline but not used in final analysis:

```
data/
├── compustat_controls.csv             # Raw Compustat pull (intermediate)
├── ibes_surprise.csv                  # Raw I/B/E/S pull (intermediate)
└── eu_adr_candidates.csv              # EU seed list before CRSP/TAQ filtering
```

---

## Replication notes

1. **WRDS access required:** TAQ, Compustat, CRSP, and I/B/E/S data
   require a valid WRDS subscription. Princeton credentials configured
   in `~/.pgpass` (see `slurm_financial_controls.sh`).

2. **Refinitiv StreetEvents required:** MP3 audio and transcript files
   are licensed data accessed via the Princeton library subscription.
   They cannot be redistributed.

3. **VIX data:** Download manually from FRED before running
   `run_vix_interaction.py`. Save as `data/vix_daily.csv` with
   columns `date` and `vix`.

4. **HuggingFace cache:** The FinBERT model (~440 MB) is cached at
   `/scratch/network/$USER/huggingface_cache/` to avoid home-directory
   quota issues on Adroit. Set before running FinBERT scripts:
   ```bash
   export HF_HOME=/scratch/network/$USER/huggingface_cache
   export TRANSFORMERS_CACHE=/scratch/network/$USER/huggingface_cache
   ```

5. **Log directory:** SLURM scripts write logs to
   `/scratch/network/$USER/thesis_week1/logs/`.
   Created automatically by each SLURM script on first run.

6. **Timezone:** All timestamps in this pipeline are US Eastern Time
   unless explicitly noted as GMT/UTC. The 2023 TAQ timezone bug
   (UTC stored as ET) is corrected in `27_sync_all.py` and documented
   in `09_download_taq_data.py`.
