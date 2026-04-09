#!/usr/bin/env python3
"""
28_run_finbert_eu.py
Analyst tone scoring for EU ADR earnings calls via ProsusAI/finbert.

Identical methodology to 28_run_finbert.py (US pipeline):
  - Extracts Q&A section analyst turns only
  - Excludes management (CEO/CFO) and operator turns
  - Weighted by word count per sentence
  - analyst_tone = sum_s w_s * (P(pos)_s - P(neg)_s)

Includes skip-existing guard: re-running the script will not
re-score already-processed calls.

Output: data/finbert/finbert_tone_eu.csv

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang 
# Advisor: Daniel Rigobon
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax

data_dir  = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
trans_dir = data_dir / "transcripts" / "europe" / "processed"
out_dir   = data_dir / "finbert"
out_dir.mkdir(parents=True, exist_ok=True)
out_path  = out_dir / "finbert_tone_eu.csv"

# Model setup 
MODEL_NAME = "ProsusAI/finbert"
tokenizer  = BertTokenizer.from_pretrained(MODEL_NAME)
model      = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# ProsusAI/finbert label order: [positive, negative, neutral]
# Confirmed from model config.json: id2label = {0: positive, 1: negative, 2: neutral}
IDX_POS, IDX_NEG = 0, 1

#  Skip-existing guard 
existing_keys = set()
if out_path.exists():
    existing = pd.read_csv(out_path)
    existing_keys = set(zip(existing["ticker"], existing["quarter"]))
    print(f"Found {len(existing_keys)} already-scored EU calls. Skipping.")

#  Analyst turn extraction (identical to US pipeline) 
ANALYST_PATTERNS = [
    r'^(?:analyst|question|q)\s*:',
    r'^\s{0,4}[A-Z][a-z]+ [A-Z][a-z]+\s*[\-\u2013]\s*(?:analyst|research)',
]

MGMT_PATTERNS = [
    r'^(?:ceo|cfo|coo|president|operator|moderator|chairman)\s*:',
    r'^(?:thank you|operator|ladies and gentlemen)',
]

def is_analyst_turn(block: str) -> bool:
    return any(re.search(p, block, re.IGNORECASE) for p in ANALYST_PATTERNS)

def is_management_turn(block: str) -> bool:
    return any(re.search(p, block, re.IGNORECASE) for p in MGMT_PATTERNS)

def extract_analyst_sentences(text: str) -> list:
    # Split on Q&A section header
    qa_parts = re.split(
        r'(?i)\b(?:questions?\s+and\s+answers?|q\s*&\s*a\s+session|'
        r'question[- ]and[- ]answer)',
        text
    )
    qa_text = qa_parts[-1] if len(qa_parts) > 1 else text

    blocks     = re.split(r'\n{2,}', qa_text)
    sentences  = []
    in_analyst = False

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if is_analyst_turn(block):
            in_analyst = True
        elif is_management_turn(block):
            in_analyst = False
        if in_analyst:
            sents = [s.strip()
                     for s in re.split(r'(?<=[.!?])\s+', block)
                     if len(s.strip()) > 10]
            sentences.extend(sents)

    return sentences

def score_sentence(sentence: str) -> np.ndarray:
    tokens = tokenizer(sentence, return_tensors="pt",
                       truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**tokens).logits
        probs  = softmax(logits, dim=1).squeeze().numpy()
    return probs  # [P(pos), P(neg), P(neutral)]

#  Main scoring loop 
results    = []
txt_files  = sorted(trans_dir.glob("*.txt"))
print(f"Found {len(txt_files)} EU transcript files.")

for path in txt_files:
    parts   = path.stem.split("_")
    ticker  = parts[0]
    quarter = "_".join(parts[1:])

    if (ticker, quarter) in existing_keys:
        continue

    text      = path.read_text(encoding="utf-8", errors="replace")
    sentences = extract_analyst_sentences(text)

    if not sentences:
        results.append({
            "ticker":        ticker,
            "quarter":       quarter,
            "analyst_tone":  np.nan,
            "n_sentences":   0,
            "p_pos_mean":    np.nan,
            "p_neg_mean":    np.nan,
        })
        print(f"  WARN {ticker} {quarter}: no analyst sentences found")
        continue

    scores  = [score_sentence(s) for s in sentences]
    weights = np.array([len(s.split()) for s in sentences], dtype=float)
    weights /= weights.sum()

    p_pos = float(sum(w * s[IDX_POS] for w, s in zip(weights, scores)))
    p_neg = float(sum(w * s[IDX_NEG] for w, s in zip(weights, scores)))

    results.append({
        "ticker":       ticker,
        "quarter":      quarter,
        "analyst_tone": p_pos - p_neg,
        "n_sentences":  len(sentences),
        "p_pos_mean":   p_pos,
        "p_neg_mean":   p_neg,
    })
    print(f"  OK  {ticker} {quarter}  "
          f"n_sents={len(sentences)}  tone={p_pos - p_neg:.4f}")

#  Save (append to existing if any) 
new_df = pd.DataFrame(results)

if out_path.exists() and len(results) > 0:
    old_df   = pd.read_csv(out_path)
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["ticker", "quarter"], keep="last")
elif out_path.exists():
    combined = pd.read_csv(out_path)
else:
    combined = new_df

combined.to_csv(out_path, index=False)

print(f"\nScored: {len(results)} new calls")
print(f"Total EU calls in file: {len(combined)}")
print(f"Non-missing tone: {combined['analyst_tone'].notna().sum()}")
