
# Replication Code: Do CEO Voices Move Markets?



**Olivia Yang — Princeton ORF 499 Senior Thesis (2024)**

Advisor: Daniel Rigobon



## Overview

Complete replication pipeline for "Do CEO Voices Move Markets?

Vocal Stress and Analyst Tone in Earnings Calls as Predictors

of High-Frequency Order Flow."



## Requirements

- Python 3.11

- WRDS database access (Princeton subscription)

- OpenSMILE (eGeMAPS v02)

- HuggingFace transformers (ProsusAI/finbert)



## Execution Order



| Script | Purpose |

|--------|---------|

| `01_select_sample.py` | Stratified CRSP sampling |

| `09_download_taq_data.py` | NYSE TAQ millisecond download |

| `35_rename_2023.py` | Audio/transcript standardisation |

| `26_extract_all_audio.py` | eGeMAPS acoustic features |

| `28_run_finbert.py` | FinBERT analyst tone |

| `27_sync_all.py` | Lee-Ready OI computation |

| `rebuild_master_v2.py` | Master dataset assembly |

| `run_regressions.py` | M1-M5, regime analysis, BH scan |

| `run_validation.py` | Placebo, permutation, bootstrap |

| `robustness_tests.py` | SE specifications, diagnostics |

| `run_regressions_global.py` | G1-G5 cross-market specs |



## Key Outputs

- `analysis_dataset_MASTER.parquet` — 310 x 156, US sample

- `analysis_dataset_GLOBAL.parquet` — 430 x 254, US + EU pooled



## Critical Note: Timezone Correction

2023 TAQ timestamps stored in UTC must be converted to Eastern

Time (UTC-4). See `27_sync_all.py` for implementation.

