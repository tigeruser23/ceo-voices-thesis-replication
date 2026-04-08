#!/usr/bin/env python3
"""
38_sync_eu_adr.py
Lee-Ready signed order imbalance for EU ADR firms using the
open-reaction window.

DESIGN RATIONALE (open-reaction window):
  EU earnings calls occur 07:00-09:00 ET (pre-market). NYSE TAQ
  data is illiquid pre-open; using a call-synchronized window
  would yield noisy or zero-trade windows. Instead we anchor
  to NYSE open:
    Pre-window:   09:00-09:30 ET  (30 min before open)
    Event window: 09:30-10:30 ET  (first 60 min of trading)
  The EU indicator in pooled regressions absorbs any OI-level
  difference attributable to this window choice.

COLUMN CORRECTION:
  TAQ cqm tables use 'ofrsiz' for ask size. The EU download
  script stores this as 'asksiz' for clarity. This script
  handles both column names.

Clean rewrite (v2): no iterative patches from prior runs.

Output: data/synchronized/eu_adr_synchronized.csv

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang 
# Advisor: Daniel Rigobon
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

data_dir = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
taq_dir  = data_dir / "taq" / "eu"
out_dir  = data_dir / "synchronized"
out_dir.mkdir(parents=True, exist_ok=True)

sample = pd.read_csv(data_dir / "european_adr_sample.csv")

# Open-reaction window bounds (ET, fixed)
PRE_START  = pd.Timestamp("1900-01-01 09:00:00")   # date replaced per call
PRE_END    = pd.Timestamp("1900-01-01 09:30:00")
EVENT_START = pd.Timestamp("1900-01-01 09:30:00")
EVENT_END   = pd.Timestamp("1900-01-01 10:30:00")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_taq_eu(ticker: str, quarter: str):
    """Load EU TAQ parquet; return (trades_df, quotes_df) or (None, None)."""
    path = taq_dir / f"{ticker}_{quarter}_taq.parquet"
    if not path.exists():
        return None, None

    df = pd.read_parquet(path)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="mixed", errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])

    trades = df[df["type"] == "trade"].copy()
    quotes = df[df["type"] == "quote"].copy()
    return trades, quotes


def sign_trades(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """
    Lee-Ready trade signing.
    Quote midpoint rule first; tick rule for ties.
    Handles both 'ask'/'asksiz' and 'ofr'/'ofrsiz' column names.
    """
    if trades.empty or quotes.empty:
        return trades.assign(sign=np.nan)

    # Normalise ask column name
    if "ask" in quotes.columns:
        ask_col = "ask"
    elif "ofr" in quotes.columns:
        ask_col = "ofr"
    else:
        return trades.assign(sign=np.nan)

    quotes = quotes[["timestamp", "bid", ask_col]].dropna().copy()
    quotes["midpoint"] = (quotes["bid"] + quotes[ask_col]) / 2
    quotes = quotes[["timestamp", "midpoint"]].sort_values("timestamp")

    trades = trades.sort_values("timestamp").copy()

    trades = pd.merge_asof(
        trades, quotes,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("5s")
    )

    trades["sign"] = np.where(
        trades["price"] > trades["midpoint"],  1,
        np.where(trades["price"] < trades["midpoint"], -1, np.nan)
    )

    # Tick rule for midpoint ties / missing midpoint
    tick = np.sign(trades["price"].diff())
    trades["sign"] = np.where(trades["sign"].isna(), tick, trades["sign"])

    return trades


def order_imbalance(signed_trades: pd.DataFrame):
    """Returns (OI, n_trades). OI = signed_volume / total_volume."""
    valid = signed_trades.dropna(subset=["sign"])
    if len(valid) == 0:
        return np.nan, 0
    return float(valid["sign"].sum() / len(valid)), len(valid)


# ── Main loop ─────────────────────────────────────────────────────────────────
quarters = [f"Q{q}_{y}" for y in [2022, 2023] for q in [1, 2, 3, 4]]
all_results = []

for _, firm in sample.iterrows():
    ticker = firm["ticker"]

    for quarter in quarters:
        trades, quotes = load_taq_eu(ticker, quarter)

        if trades is None:
            print(f"  MISS {ticker} {quarter}: no TAQ file")
            continue

        if trades.empty:
            print(f"  EMPTY {ticker} {quarter}: no trade records")
            all_results.append({
                "ticker": ticker, "quarter": quarter,
                "pre_oi": np.nan, "pre_n_trades": 0,
                "event_oi": np.nan, "event_n_trades": 0,
                "oi_shift": np.nan,
                "window_type": "open_reaction",
                "dropped_reason": "no_trades",
            })
            continue

        # Build date-aware window bounds from actual trade dates
        call_date = trades["timestamp"].dt.date.mode()[0]
        date_str  = str(call_date)

        pre_s  = pd.Timestamp(f"{date_str} 09:00:00")
        pre_e  = pd.Timestamp(f"{date_str} 09:30:00")
        evt_s  = pd.Timestamp(f"{date_str} 09:30:00")
        evt_e  = pd.Timestamp(f"{date_str} 10:30:00")

        signed = sign_trades(trades, quotes)

        pre_trades   = signed[(signed["timestamp"] >= pre_s) &
                               (signed["timestamp"] <  pre_e)]
        event_trades = signed[(signed["timestamp"] >= evt_s) &
                               (signed["timestamp"] <  evt_e)]

        oi_pre,   n_pre   = order_imbalance(pre_trades)
        oi_event, n_event = order_imbalance(event_trades)

        # Drop if insufficient activity
        if n_event < 10:
            print(f"  DROP {ticker} {quarter}: only {n_event} event-window trades")
            all_results.append({
                "ticker": ticker, "quarter": quarter,
                "pre_oi": oi_pre, "pre_n_trades": n_pre,
                "event_oi": oi_event, "event_n_trades": n_event,
                "oi_shift": np.nan,
                "window_type": "open_reaction",
                "dropped_reason": "insufficient_trades",
            })
            continue

        oi_shift = (float(oi_event) - float(oi_pre)
                    if not (np.isnan(oi_pre) or np.isnan(oi_event))
                    else np.nan)

        all_results.append({
            "ticker":           ticker,
            "quarter":          quarter,
            "call_date":        str(call_date),
            "pre_oi":           oi_pre,
            "pre_n_trades":     n_pre,
            "event_oi":         oi_event,
            "event_n_trades":   n_event,
            "oi_shift":         oi_shift,
            "window_type":      "open_reaction",
            "dropped_reason":   None,
        })

        print(f"  OK  {ticker} {quarter}  "
              f"pre_n={n_pre}  event_n={n_event}  "
              f"oi_shift={oi_shift:.4f}" if oi_shift is not None
              else f"  OK  {ticker} {quarter}  oi_shift=NaN")

# ── Save ───────────────────────────────────────────────────────────────────────
sync_df = pd.DataFrame(all_results)
sync_df.to_csv(out_dir / "eu_adr_synchronized.csv", index=False)

valid = sync_df["oi_shift"].notna().sum()
print(f"\nTotal firm-quarters processed: {len(sync_df)}")
print(f"Valid OI-shift observations:   {valid}")
print(f"Dropped (insufficient trades): "
      f"{(sync_df['dropped_reason'] == 'insufficient_trades').sum()}")
