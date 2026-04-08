# Replication Code: Do CEO Voices Move Markets?

**Olivia Yang — Princeton Senior Thesis**  
Advisor: Daniel Rigobon

> **Note:** Portions of the pipeline code were debugged with assistance from
> Claude AI (Anthropic). Core statistical design and all empirical choices
> are my own.

## Overview

Complete replication pipeline for *"Do CEO Voices Move Markets?
Vocal Stress and Analyst Tone in Earnings Calls as Predictors
of High-Frequency Order Flow."*

The pipeline integrates five data sources: NYSE TAQ (millisecond
trade/quote data), Refinitiv StreetEvents (MP3 audio + transcripts),
WRDS Compustat (ROA, B/M), CRSP (log market equity), and I/B/E/S
(analyst consensus). The European ADR extension adds 23 EU firms
across 11 countries using the same TAQ infrastructure.

---

## Requirements

- Python 3.11
- WRDS database access (Princeton subscription)
- OpenSMILE with eGeMAPS v02 feature set
- HuggingFace `transformers` (ProsusAI/finbert)
- Packages: `wrds`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`,
  `scipy`, `torch`, `opensmile`, `pathlib`

---

## Execution Order

### US Pipeline (A.1–A.10)

| Script | Purpose | Key Output |
|--------|---------|------------|
| `01_select_sample.py` | Stratified CRSP sampling (40 firms, 5 volatility quintiles) | `selected_sample_40_FINAL.csv` |
| `09_download_taq_data.py` | NYSE TAQ millisecond download (6h window, UTC→ET corrected) | `taq/{TICKER}_{QTR}_taq.csv.gz` |
| `35_rename_2023.py` | Audio/transcript standardisation to `TICKER_Qn_YYYY` | `audio/processed/`, `transcripts/processed/` |
| `26_extract_all_audio.py` | eGeMAPS 88-feature extraction via OpenSMILE | `audio_features_all308.csv` |
| `28_run_finbert.py` | FinBERT analyst tone scoring (Q&A turns only) | `finbert_tone_results_all.csv` |
| `27_sync_all.py` | Lee-Ready OI computation; timezone fix for 2023 | `full_synchronized.csv` |
| `rebuild_master_v2.py` | US master dataset assembly (310 × 156) | `analysis_dataset_MASTER.parquet` |
| `run_regressions.py` | M1–M5, regime split, Chow test, BH scan | `regression_results.csv`, `regime_results.csv`, `bh_scan_results.csv` |
| `run_validation.py` | Placebo, permutation, bootstrap, wrong-quarter | `validation_results.csv` |
| `robustness_tests.py` | SE specifications, autocorrelation diagnostics | stdout |

### European ADR Extension (A.12–A.19)

| Script | Purpose | Key Output |
|--------|---------|------------|
| `01_eu_adr_sample_v2.py` | EU ADR sample selection (23 firms, 11 countries) | `european_adr_sample.csv` |
| `09_download_taq_eu_adr_v2.py` | NYSE TAQ download for EU firms (open-reaction window) | `taq/eu/{TICKER}_{QTR}_taq.parquet` |
| `rename_eu_files_v2.py` | Refinitiv→pipeline filename convention; dup resolution | `audio/europe/processed/`, `transcripts/europe/processed/` |
| `26_extract_eu_audio.py` | eGeMAPS extraction for EU (auto-discovers, skip-existing) | `audio_features/europe/eu_audio_features_all.csv` |
| `28_run_finbert_eu.py` | FinBERT analyst tone for EU (skip-existing guard) | `finbert/finbert_tone_eu.csv` |
| `38_sync_eu_adr.py` | Lee-Ready OI with open-reaction window for EU ADRs | `synchronized/eu_adr_synchronized.csv` |
| `rebuild_master_global_v2.py` | Global dataset assembly (US + EU) | `analysis_dataset_GLOBAL.parquet` |
| `run_regressions_global.py` | G1–G6 cross-market specifications | `regression_results_global.csv` |

---

## Key Outputs

| File | Dimensions | Description |
|------|-----------|-------------|
| `analysis_dataset_MASTER.parquet` | 310 × 156 | US sample; 281–282 complete cases |
| `analysis_dataset_GLOBAL.parquet` | 490 × 257 | US + EU pooled; 438 complete (reduced controls) |
| `analysis_dataset_EU_only.parquet` | 196 × 257 | EU subset; 164 tone+OI complete cases |

---

## Critical Notes

### Timezone Correction (2023 TAQ data)
2023 TAQ timestamps were stored in UTC in the source data but were
incorrectly applied as Eastern Time, shifting all event windows by +4 hours.
**Fix:** subtract 4 hours before computing pre/during windows.
Pre-fix OI standard deviation ≈ 0.50; post-fix ≈ 0.07–0.10.
See `27_sync_all.py` for implementation.

### EU Open-Reaction Window
All European ADR calls occur 07:00–09:00 ET (pre-market). NYSE TAQ
is illiquid pre-open, so EU OI is measured using a fixed open-reaction
window: pre-window 09:00–09:30 ET, event window 09:30–10:30 ET.
The `is_eu` indicator in pooled regressions absorbs any level
difference from this window choice.

### Semi-Annual Reporter Exclusions
CFR (Richemont) and RIO (Rio Tinto) are retained through data
collection but excluded at the regression stage. DGE, REL, BATS,
and ULVR are excluded during file renaming.

### Refinitiv → NYSE Ticker Mapping
Key mappings used in `rename_eu_files_v2.py`:
ABBN→ABB, INGA→ING, NOKIA→NOK, TTEF→TTE, PHIA→PHG,
STMPA→STM, HSBA→HSBC.

---

## Data Sources

All data accessed via Princeton WRDS subscription:
- **NYSE TAQ:** `taqmsec.ctm_YYYYMMDD` (trades), `taqmsec.cqm_YYYYMMDD` (quotes)
- **CRSP:** `crsp.msf` (monthly security file)
- **Compustat:** `comp.fundq` (quarterly fundamentals)
- **I/B/E/S:** analyst consensus data
- **Refinitiv StreetEvents:** MP3 audio and matched transcripts
  (accessed via Princeton library subscription)

---

## Replication Environment

All scripts executed on Princeton's Adroit HPC cluster.  
Working directory: `/scratch/network/oy3009/thesis_week1/`  
Conda environment: `thesis_env_scratch`  
Activate with:
```bash
cd /scratch/network/oy3009/thesis_week1/code
source /scratch/network/oy3009/thesis_env_scratch/bin/activate
```
