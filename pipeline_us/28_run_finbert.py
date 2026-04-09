#!/usr/bin/env python3
"""
28_run_finbert.py
Score analyst tone in earnings call Q&A sections using ProsusAI/finbert.

Extracts analyst speaking turns only (excludes management and operator).
analyst_tone = sum_s w_s * (P(pos)_s - P(neg)_s), weighted by word count.
FinBERT label order: [positive=0, negative=1, neutral=2].

Input:  data/transcripts/processed/*.txt
Output: data/finbert/finbert_tone_results_all.csv

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang, Princeton Senior Thesis 
# Advisor: Daniel Rigobon
"""

import pandas as pd, numpy as np, re, os

from pathlib import Path

from transformers import BertTokenizer, BertForSequenceClassification

import torch

from torch.nn.functional import softmax



data_dir  = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

trans_dir = data_dir / "transcripts" / "processed"

out_dir   = data_dir / "finbert"

out_dir.mkdir(exist_ok=True)



MODEL_NAME = "ProsusAI/finbert"

tokenizer  = BertTokenizer.from_pretrained(MODEL_NAME)

model      = BertForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()



ANALYST_PATTERNS = [

    r'^(?:analyst|question|q):',

    r'^\s{0,4}[A-Z][a-z]+ [A-Z][a-z]+\s*[\-\u2013]\s*(?:analyst|research)',

]



def is_analyst_turn(block):

    return any(re.search(p, block, re.IGNORECASE) for p in ANALYST_PATTERNS)



def extract_analyst_sentences(text):

    qa_parts = re.split(r'(?i)\bquestions?\s+and\s+answers?|q\s*&\s*a\s+session', text)

    qa_text  = qa_parts[-1] if len(qa_parts) > 1 else text

    blocks   = re.split(r'\n{2,}', qa_text)

    sentences, in_analyst = [], False

    for block in blocks:

        block = block.strip()

        if not block: continue

        if is_analyst_turn(block):

            in_analyst = True

        elif re.search(r'^(?:ceo|cfo|operator|moderator):', block, re.I):

            in_analyst = False

        if in_analyst:

            sentences += [s.strip()

                          for s in re.split(r'(?<=[.!?])\s+', block)

                          if len(s.strip()) > 10]

    return sentences



def score_sentence(sentence):

    tokens = tokenizer(sentence, return_tensors="pt",

                       truncation=True, max_length=512, padding=True)

    with torch.no_grad():

        probs = softmax(model(**tokens).logits, dim=1).squeeze().numpy()

    return probs



results = []

for path in sorted(trans_dir.glob("*.txt")):

    parts     = path.stem.split("_")

    ticker    = parts[0]

    quarter   = "_".join(parts[1:])

    text      = path.read_text(encoding='utf-8', errors='replace')

    sentences = extract_analyst_sentences(text)

    if not sentences:

        results.append({"ticker": ticker, "quarter": quarter,

                        "analyst_tone": np.nan, "n_sentences": 0})

        continue

    scores  = [score_sentence(s) for s in sentences]

    weights = np.array([len(s.split()) for s in sentences], dtype=float)

    weights /= weights.sum()

    p_pos = sum(w * s[0] for w, s in zip(weights, scores))

    p_neg = sum(w * s[1] for w, s in zip(weights, scores))

    results.append({"ticker": ticker, "quarter": quarter,

                    "analyst_tone": float(p_pos - p_neg),

                    "n_sentences": len(sentences)})



pd.DataFrame(results).to_csv(out_dir / "finbert_tone_results_all.csv", index=False)

