#!/usr/bin/env python3
"""
35_rename_2023.py
Standardises Refinitiv StreetEvents audio and transcript filenames to
the pipeline convention: TICKER_Qn_YYYY.{mp3,txt}

Quarter assigned by calendar convention (not fiscal):
  Q1: Jan–Mar | Q2: Apr–Jun | Q3: Jul–Sep | Q4: Oct–Dec

Corrects previous bug where both Q3 and Q4 2023 calls were
mapped into Q4_2023 slots due to a fiscal-vs-calendar offset error.
Clears any existing 2023 processed files before re-renaming.

Input:  data/audio/raw/*.mp3
        data/transcripts/raw/*.txt
        data/selected_sample_40_FINAL.csv
Output: data/audio/processed/
        data/transcripts/processed/

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang, Princeton Senior Thesis  
# Advisor: Daniel Rigobon
"""



import os, re, shutil

from datetime import datetime

import pandas as pd



base   = f"/scratch/network/{os.environ['USER']}/thesis_week1/data"

a_raw  = f"{base}/audio/raw"

a_proc = f"{base}/audio/processed"

t_raw  = f"{base}/transcripts/raw"

t_proc = f"{base}/transcripts/processed"



sample = pd.read_csv(f"{base}/selected_sample_40_FINAL.csv")

valid  = set(sample['ticker'].tolist())

MONTH  = {

    'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,

    'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12,

}



def date_to_quarter(dt):

    m = dt.month

    if   m <= 3:  return f"Q1_{dt.year}"

    elif m <= 6:  return f"Q2_{dt.year}"

    elif m <= 9:  return f"Q3_{dt.year}"

    else:         return f"Q4_{dt.year}"



def parse_audio_date(fname):

    m = re.match(r'^(\d{2})(\d{2})(\d{2})_([A-Z]+)_', fname)

    if not m: return None, None

    return datetime(int(m.group(3))+2000,

                    int(m.group(1)), int(m.group(2))), m.group(4)



def parse_transcript_date(fname):

    m = re.match(r'^(\d{4})-([A-Za-z]+)-(\d{2})-([A-Z]+)[.\-]', fname)

    if not m: return None, None

    mon = MONTH.get(m.group(2))

    if not mon: return None, None

    return datetime(int(m.group(1)), mon, int(m.group(3))), m.group(4)



os.makedirs(a_proc, exist_ok=True)

os.makedirs(t_proc, exist_ok=True)



for f in os.listdir(t_proc):

    if '2023' in f:

        os.remove(os.path.join(t_proc, f))



ok, skipped = 0, 0

for fname in sorted(os.listdir(t_raw)):

    if not fname.endswith('.txt'): continue

    dt, ticker = parse_transcript_date(fname)

    if dt is None or ticker not in valid:

        skipped += 1; continue

    shutil.copy2(os.path.join(t_raw, fname),

                 os.path.join(t_proc, f"{ticker}_{date_to_quarter(dt)}.txt"))

    ok += 1



for fname in sorted(os.listdir(a_raw)):

    if not fname.endswith('.mp3'): continue

    dt, ticker = parse_audio_date(fname)

    if dt is None or ticker not in valid:

        skipped += 1; continue

    shutil.copy2(os.path.join(a_raw, fname),

                 os.path.join(a_proc, f"{ticker}_{date_to_quarter(dt)}.mp3"))

    ok += 1



print(f"Renamed: {ok}  |  Skipped: {skipped}")

