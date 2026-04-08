#!/usr/bin/env python3
"""
01_eu_adr_sample_v2.py
European ADR sample selection: 23 firms across 11 countries.

Selects from a 36-firm seed list. Validates CRSP coverage
(8 quarters, price >= $5, mktcap >= $5B) and confirms presence
in taqmsec. Excludes semi-annual reporters. Stratifies by
volatility within country.

Output: data/european_adr_sample.csv

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang
# Advisor: Daniel Rigobon
"""

import wrds
import pandas as pd
import numpy as np
import os
from pathlib import Path

RANDOM_SEED = 42
data_dir = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
data_dir.mkdir(parents=True, exist_ok=True)

# ── Seed list: 36 European ADR candidates ─────────────────────────────────────
# NYSE/NASDAQ tickers for European firms with ADR listings.
# Semi-annual reporters flagged; excluded from final sample post-validation.
SEED_LIST = [
    # France
    {"ticker": "SNY",  "company": "Sanofi SA",            "country": "France",      "semi_annual": False},
    {"ticker": "TTE",  "company": "TotalEnergies SE",     "country": "France",      "semi_annual": False},
    # Germany
    {"ticker": "SAP",  "company": "SAP SE",               "country": "Germany",     "semi_annual": False},
    # Italy
    {"ticker": "E",    "company": "Eni SpA",              "country": "Italy",       "semi_annual": False},
    {"ticker": "RACE", "company": "Ferrari NV",           "country": "Italy",       "semi_annual": False},
    {"ticker": "STM",  "company": "STMicroelectronics",   "country": "Italy",       "semi_annual": False},
    # Netherlands
    {"ticker": "ASML", "company": "ASML Holding NV",      "country": "Netherlands", "semi_annual": False},
    {"ticker": "ING",  "company": "ING Groep NV",         "country": "Netherlands", "semi_annual": False},
    {"ticker": "PHG",  "company": "Philips NV",           "country": "Netherlands", "semi_annual": False},
    # Norway
    {"ticker": "EQNR", "company": "Equinor ASA",          "country": "Norway",      "semi_annual": False},
    # Spain
    {"ticker": "BBVA", "company": "Banco Bilbao Vizcaya", "country": "Spain",       "semi_annual": False},
    {"ticker": "SAN",  "company": "Banco Santander SA",   "country": "Spain",       "semi_annual": False},
    {"ticker": "TEF",  "company": "Telefonica SA",        "country": "Spain",       "semi_annual": False},
    # Sweden
    {"ticker": "ERIC", "company": "Ericsson",             "country": "Sweden",      "semi_annual": False},
    # Switzerland
    {"ticker": "ABB",  "company": "ABB Ltd",              "country": "Switzerland", "semi_annual": False},
    {"ticker": "NVS",  "company": "Novartis AG",          "country": "Switzerland", "semi_annual": False},
    {"ticker": "CFR",  "company": "Richemont SA",         "country": "Switzerland", "semi_annual": True},  # excluded
    # UK
    {"ticker": "AZN",  "company": "AstraZeneca PLC",      "country": "UK",          "semi_annual": False},
    {"ticker": "BP",   "company": "BP plc",               "country": "UK",          "semi_annual": False},
    {"ticker": "GSK",  "company": "GSK plc",              "country": "UK",          "semi_annual": False},
    {"ticker": "HSBC", "company": "HSBC Holdings plc",    "country": "UK",          "semi_annual": False},
    {"ticker": "SHEL", "company": "Shell plc",            "country": "UK",          "semi_annual": False},
    {"ticker": "RIO",  "company": "Rio Tinto plc",        "country": "UK",          "semi_annual": True},  # excluded
    # Belgium
    {"ticker": "ARGX", "company": "argenx SE",            "country": "Belgium",     "semi_annual": False},
    # Finland
    {"ticker": "NOK",  "company": "Nokia Oyj",            "country": "Finland",     "semi_annual": False},
    # Additional candidates (may be excluded by CRSP/TAQ checks)
    {"ticker": "DGE",  "company": "Diageo plc",           "country": "UK",          "semi_annual": True},  # semi-annual
    {"ticker": "REL",  "company": "RELX plc",             "country": "UK",          "semi_annual": True},  # semi-annual
    {"ticker": "BATS", "company": "British Am. Tobacco",  "country": "UK",          "semi_annual": True},  # semi-annual
    {"ticker": "ULVR", "company": "Unilever plc",         "country": "UK",          "semi_annual": True},  # semi-annual
    {"ticker": "UL",   "company": "Unilever plc (ADR)",   "country": "UK",          "semi_annual": False},
    {"ticker": "LIN",  "company": "Linde plc",            "country": "Germany",     "semi_annual": False},
    {"ticker": "ICLR", "company": "ICON plc",             "country": "Ireland",     "semi_annual": False},
    {"ticker": "STZ",  "company": "Constellation Brands", "country": "Other",       "semi_annual": False},
    {"ticker": "DEO",  "company": "Diageo (ADR)",         "country": "UK",          "semi_annual": False},
    {"ticker": "RDS",  "company": "Shell (old ADR)",      "country": "UK",          "semi_annual": False},
]

seed_df = pd.DataFrame(SEED_LIST)
# Exclude semi-annual reporters immediately
seed_df = seed_df[~seed_df["semi_annual"]].reset_index(drop=True)
tickers  = seed_df["ticker"].tolist()
ticker_str = "', '".join(tickers)

# ── WRDS CRSP validation ───────────────────────────────────────────────────────
print("Connecting to WRDS...")
db = wrds.Connection()

crsp = db.raw_sql(f"""
    SELECT a.permno, a.ticker, a.date, a.ret, a.shrout, a.prc, a.exchcd
    FROM crsp.msf a
    INNER JOIN crsp.stocknames b ON a.permno = b.permno
    WHERE b.ticker IN ('{ticker_str}')
      AND a.date BETWEEN '2021-01-01' AND '2023-12-31'
      AND a.exchcd IN (1, 2, 3)
      AND a.ret IS NOT NULL
      AND ABS(a.prc) >= 5
""")

study_quarters = pd.period_range("2022Q1", "2023Q4", freq="Q")
crsp["period"] = pd.to_datetime(crsp["date"]).dt.to_period("Q")
crsp["mktcap_b"] = crsp["shrout"].abs() * crsp["prc"].abs() / 1e6  # billions

# Require 8 quarters of coverage
coverage = (crsp[crsp["period"].isin(study_quarters)]
            .groupby("permno")
            .agg(n_quarters=("period", "nunique"),
                 mean_price=("prc", lambda x: x.abs().mean()),
                 mean_mktcap_b=("mktcap_b", "mean"))
            .reset_index())

# mktcap >= $5B filter
coverage = coverage[
    (coverage["n_quarters"] == 8) &
    (coverage["mean_mktcap_b"] >= 5.0)
]

# Latest ticker per permno
latest_tickers = (crsp.sort_values("date")
                  .groupby("permno")[["ticker", "exchcd"]]
                  .last()
                  .reset_index())

coverage = coverage.merge(latest_tickers, on="permno")

# ── taqmsec availability check ─────────────────────────────────────────────────
# Spot-check Q1 2022 and Q1 2023 for each firm
taq_results = []
for _, row in coverage.iterrows():
    tkr = row["ticker"]
    ok_22, ok_23 = False, False
    for tbl in ["taqmsec.ctm_20220201", "taqmsec.ctm_20220202"]:
        try:
            chk = db.raw_sql(f"""
                SELECT COUNT(*) AS n FROM {tbl}
                WHERE sym_root = '{tkr}'
            """)
            if chk["n"].iloc[0] > 0:
                ok_22 = True
                break
        except Exception:
            pass
    for tbl in ["taqmsec.ctm_20230201", "taqmsec.ctm_20230202"]:
        try:
            chk = db.raw_sql(f"""
                SELECT COUNT(*) AS n FROM {tbl}
                WHERE sym_root = '{tkr}'
            """)
            if chk["n"].iloc[0] > 0:
                ok_23 = True
                break
        except Exception:
            pass
    taq_results.append({"permno": row["permno"],
                        "taq_2022Q1": ok_22,
                        "taq_2023Q1": ok_23,
                        "taq_ok": ok_22 and ok_23})
    print(f"  TAQ check {tkr}: 2022={ok_22}  2023={ok_23}")

taq_df = pd.DataFrame(taq_results)
coverage = coverage.merge(taq_df, on="permno")
final = coverage[coverage["taq_ok"]].copy()

# ── Merge metadata from seed list ─────────────────────────────────────────────
final = final.merge(seed_df[["ticker", "company", "country"]],
                    on="ticker", how="left")

# ── Volatility stratification within country ──────────────────────────────────
np.random.seed(RANDOM_SEED)
crsp_full = crsp[crsp["permno"].isin(final["permno"])].copy()
crsp_full["date"] = pd.to_datetime(crsp_full["date"])

vol_df = (crsp_full[crsp_full["date"] <= "2022-06-30"]
          .groupby("permno")["ret"]
          .std()
          .reset_index(name="hist_vol"))

final = final.merge(vol_df, on="permno", how="left")

# Quintile within country
final["vol_quintile"] = (final.groupby("country")["hist_vol"]
                         .transform(lambda x:
                             pd.qcut(x, q=min(5, x.nunique()),
                                     labels=False, duplicates="drop") + 1
                             if len(x) >= 2 else 1))

# Firm-quarter count
final["n_firm_quarters"] = 8
final["sample_type"]     = "EU_ADR_v2"

# ── Save ───────────────────────────────────────────────────────────────────────
out_cols = ["permno", "ticker", "company", "country",
            "exchcd", "hist_vol", "vol_quintile",
            "mean_price", "mean_mktcap_b",
            "taq_2022Q1", "taq_2023Q1", "taq_ok",
            "n_firm_quarters", "sample_type"]

final[out_cols].to_csv(data_dir / "european_adr_sample.csv", index=False)
print(f"\nFinal EU ADR sample: {len(final)} firms")
print(final[["ticker", "country", "mean_mktcap_b"]].to_string())
print(f"\nCountries covered: {final['country'].nunique()}")
print(final.groupby("country")["ticker"].apply(list))

db.close()
