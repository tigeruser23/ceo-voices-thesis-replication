#!/usr/bin/env python3
"""
29_financial_controls.py
Pull financial controls from WRDS Compustat and I/B/E/S.

Computes per firm-quarter:
  - ROA      = ibq / atq
  - mve      = cshoq * prccq  (raw market equity; lnmve derived in master)
  - bm       = ceqq / mve
  - leverage = ltq / atq
  - surp     = eps_actual - eps_consensus  (I/B/E/S earnings surprise)

Matching convention: most recent Compustat quarter with datadate
within 120 days before the call date.

Output: data/financial_controls_all.csv
  Columns: ticker, quarter, roa, mve, bm, leverage, surp, n_analysts

NOTE: BHAR (30/60/90-day buy-and-hold abnormal returns) is computed
by a separate script and is not part of the main regression pipeline.

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang
# Advisor: Daniel Rigobon
"""

import os
import warnings
import numpy as np
import pandas as pd
import wrds
from pathlib import Path

warnings.filterwarnings("ignore")

base     = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
out_path = base / "financial_controls_all.csv"

# ── Load sample and call times (not analysis_dataset.csv — that comes later) ──
sample     = pd.read_csv(base / "selected_sample_40_FINAL.csv")
call_times = pd.read_csv(base / "call_times_extracted.csv")

calls = sample.merge(call_times, on="ticker", how="left")
calls["call_date"] = pd.to_datetime(calls["call_date"])
calls = calls.dropna(subset=["call_date"])

tickers    = calls["ticker"].unique().tolist()
ticker_str = "'" + "','".join(tickers) + "'"

print(f"Sample: {len(calls)} firm-quarters across {len(tickers)} firms")

# ── Connect to WRDS (no hardcoded username) ────────────────────────────────────
print("Connecting to WRDS...")
db = wrds.Connection()
print("Connected.")

# ── Section 1: Compustat quarterly fundamentals ───────────────────────────────
print("\nQuerying Compustat...")
comp = db.raw_sql(f"""
    SELECT
        tic        AS ticker,
        datadate,
        atq,       -- total assets
        ibq,       -- income before extraordinary items
        ceqq,      -- common equity
        ltq,       -- total liabilities
        cshoq,     -- shares outstanding (thousands)
        prccq,     -- quarter-end price
        saleq      -- net sales
    FROM comp.fundq
    WHERE tic IN ({ticker_str})
      AND datadate BETWEEN '2021-10-01' AND '2024-03-31'
      AND indfmt = 'INDL'
      AND datafmt = 'STD'
      AND popsrc  = 'D'
      AND consol  = 'C'
    ORDER BY tic, datadate
""")
print(f"Compustat: {len(comp)} rows for {comp['ticker'].nunique()} firms")

comp = comp.dropna(subset=["atq"])
comp = comp[comp["atq"] > 0].copy()
comp["datadate"] = pd.to_datetime(comp["datadate"])

# Raw market equity (billions); lnmve = log(mve) computed in rebuild_master_v2
comp["mve"]      = (comp["cshoq"] * comp["prccq"]).clip(lower=0.001)
comp["roa"]      = comp["ibq"] / comp["atq"]
comp["bm"]       = comp["ceqq"] / comp["mve"]
comp["leverage"] = comp["ltq"]  / comp["atq"]

comp_clean = comp[["ticker", "datadate", "roa", "mve",
                   "bm", "leverage"]].copy()

# ── Section 2: I/B/E/S earnings surprise ─────────────────────────────────────
print("\nQuerying I/B/E/S...")
ibes = db.raw_sql(f"""
    SELECT
        a.ticker,
        a.pends        AS period_end,
        a.anndats      AS announce_date,
        a.value        AS eps_actual,
        b.medest       AS eps_consensus,
        b.numest       AS n_analysts,
        b.stdev        AS forecast_dispersion
    FROM ibes.actu_epsus a
    LEFT JOIN ibes.statsum_epsus b
        ON  a.ticker   = b.ticker
        AND a.pends    = b.fpedats
        AND b.fpi      = '6'
        AND b.statpers BETWEEN a.anndats - 30 AND a.anndats
    WHERE a.ticker  IN ({ticker_str})
      AND a.pends   BETWEEN '2021-10-01' AND '2024-03-31'
      AND a.pdicity = 'QTR'
    ORDER BY a.ticker, a.pends
""")
print(f"I/B/E/S: {len(ibes)} rows")

ibes = ibes.dropna(subset=["eps_actual", "eps_consensus"]).copy()
ibes["surp"]          = ibes["eps_actual"] - ibes["eps_consensus"]
ibes["announce_date"] = pd.to_datetime(ibes["announce_date"])

# Keep one row per firm-quarter (most recent announcement date)
ibes = (ibes.sort_values("announce_date")
            .drop_duplicates(subset=["ticker", "period_end"], keep="last"))

db.close()

# ── Section 3: Match to firm-quarters ─────────────────────────────────────────
print("\nMatching controls to firm-quarters...")
results = []

for _, row in calls.iterrows():
    ticker   = row["ticker"]
    quarter  = row["quarter"]
    call_dt  = row["call_date"]

    # Compustat: most recent quarterly filing within 120 days before call
    firm_comp = comp_clean[
        (comp_clean["ticker"] == ticker) &
        (comp_clean["datadate"] <= call_dt) &
        (comp_clean["datadate"] >= call_dt - pd.Timedelta(days=120))
    ].sort_values("datadate")

    if len(firm_comp) > 0:
        c = firm_comp.iloc[-1]
        roa      = c["roa"]
        mve      = c["mve"]
        bm       = c["bm"]
        leverage = c["leverage"]
    else:
        roa = mve = bm = leverage = np.nan

    # I/B/E/S: announcement within ±7 days of call
    firm_ibes = ibes[
        (ibes["ticker"] == ticker) &
        (ibes["announce_date"] >= call_dt - pd.Timedelta(days=7)) &
        (ibes["announce_date"] <= call_dt + pd.Timedelta(days=7))
    ]

    if len(firm_ibes) > 0:
        surp       = firm_ibes.iloc[0]["surp"]
        n_analysts = firm_ibes.iloc[0]["n_analysts"]
    else:
        surp = n_analysts = np.nan

    results.append({
        "ticker":     ticker,
        "quarter":    quarter,
        "roa":        roa,
        "mve":        mve,        # raw; lnmve = log(mve) in rebuild_master_v2
        "bm":         bm,
        "leverage":   leverage,
        "surp":       surp,
        "n_analysts": n_analysts,
    })

ctrl_df = pd.DataFrame(results)

# ── Summary and save ──────────────────────────────────────────────────────────
print(f"\nMatched {ctrl_df['roa'].notna().sum()} / {len(ctrl_df)} "
      f"Compustat observations")
print(f"Matched {ctrl_df['surp'].notna().sum()} / {len(ctrl_df)} "
      f"I/B/E/S observations")
print(ctrl_df[["roa", "mve", "bm", "leverage"]].describe().round(3))

ctrl_df.to_csv(out_path, index=False)
print(f"\nSaved: financial_controls_all.csv  ({len(ctrl_df)} rows)")
