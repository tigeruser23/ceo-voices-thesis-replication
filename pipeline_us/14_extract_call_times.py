#!/usr/bin/env python3
"""
14_extract_call_times.py
Extract earnings call dates and times from Refinitiv StreetEvents
transcript headers.

Parses the standard Refinitiv header format:
  "OCTOBER 25, 2022 / 8:30PM GMT"
  "NOVEMBER 03, 2022 / 2:00PM GMT"

Converts from GMT to US Eastern Time (handles EDT/EST automatically).
Output used by:
  - 09_download_taq_data.py  (TAQ download window)
  - 27_sync_all.py           (OI event-window alignment)

Output: data/call_times_extracted.csv

# NOTE: Portions of this script were debugged with assistance
# from Claude AI (Anthropic). Core statistical design and all
# empirical choices are my own.
# Author: Olivia Yang
# Advisor: Daniel Rigobon
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pytz   # pip install pytz

base         = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")
trans_dir    = base / "transcripts" / "processed"
output_path  = base / "call_times_extracted.csv"

print("=" * 60)
print("EXTRACTING CALL TIMES FROM TRANSCRIPTS")
print("=" * 60)

if not trans_dir.exists():
    raise FileNotFoundError(f"Transcripts directory not found: {trans_dir}")

transcript_files = sorted(trans_dir.glob("*.txt"))
print(f"Found {len(transcript_files)} transcript files")

MONTH_MAP = {
    "JANUARY":1, "FEBRUARY":2, "MARCH":3,    "APRIL":4,
    "MAY":5,     "JUNE":6,     "JULY":7,      "AUGUST":8,
    "SEPTEMBER":9,"OCTOBER":10,"NOVEMBER":11,"DECEMBER":12,
}

# Refinitiv header pattern: MONTH DD, YYYY / HH:MM[AM|PM] GMT
DATE_PATTERN = re.compile(
    r"([A-Z]+)\s+(\d{1,2}),\s+(\d{4})\s*/\s*(\d{1,2}):(\d{2})(AM|PM)\s+GMT",
    re.IGNORECASE
)

eastern = pytz.timezone("US/Eastern")
gmt     = pytz.UTC

call_times, errors = [], []

for fpath in transcript_files:
    parts   = fpath.stem.split("_")
    if len(parts) < 3:
        errors.append({"file": fpath.name, "error": "Cannot parse filename"})
        continue

    ticker  = parts[0]
    quarter = f"{parts[1]}_{parts[2]}"   # e.g. Q3_2022

    try:
        content = fpath.read_text(encoding="utf-8", errors="ignore")[:5000]
    except Exception as e:
        errors.append({"file": fpath.name, "error": str(e)})
        continue

    m = DATE_PATTERN.search(content)
    if not m:
        errors.append({"file": fpath.name, "error": "Date/time pattern not found"})
        print(f"  WARN {fpath.name}: no timestamp in header")
        continue

    month_str = m.group(1).upper()
    month     = MONTH_MAP.get(month_str)
    if not month:
        errors.append({"file": fpath.name, "error": f"Unknown month: {month_str}"})
        continue

    day    = int(m.group(2))
    year   = int(m.group(3))
    hour   = int(m.group(4))
    minute = int(m.group(5))
    am_pm  = m.group(6).upper()

    # Convert to 24-hour
    if am_pm == "PM" and hour != 12:
        hour += 12
    elif am_pm == "AM" and hour == 12:
        hour = 0

    try:
        call_gmt = gmt.localize(datetime(year, month, day, hour, minute))
    except ValueError as e:
        errors.append({"file": fpath.name, "error": f"Date error: {e}"})
        continue

    call_et   = call_gmt.astimezone(eastern)
    call_hour = call_et.hour + call_et.minute / 60.0

    # FIX: NYSE market hours are 9:30–16:00 ET (not 9:00)
    is_market_hours = int(9.5 <= call_hour < 16.0)

    call_times.append({
        "ticker":            ticker,
        "quarter":           quarter,
        "call_date":         call_et.strftime("%Y-%m-%d"),
        "call_time_gmt":     call_gmt.strftime("%H:%M:%S"),
        "call_time_et":      call_et.strftime("%H:%M:%S"),
        "call_datetime_gmt": call_gmt.strftime("%Y-%m-%d %H:%M:%S"),
        "call_datetime_et":  call_et.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week":       call_et.strftime("%A"),
        "call_hour_et":      round(call_hour, 3),
        "is_market_hours":   is_market_hours,
    })

print(f"\nExtracted: {len(call_times)} / {len(transcript_files)}")
print(f"Errors:    {len(errors)}")

if not call_times:
    print("ERROR: no call times extracted. Check transcript format.")
    raise SystemExit(1)

df = (pd.DataFrame(call_times)
        .sort_values(["call_date", "call_time_et"])
        .reset_index(drop=True))

df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")

print("\nQuarter distribution:")
print(df["quarter"].value_counts().sort_index().to_string())

print("\nCall time distribution (ET):")
df["hour_et"] = df["call_hour_et"].apply(int)
for hr, cnt in df["hour_et"].value_counts().sort_index().items():
    suffix = "AM" if hr < 12 else "PM"
    disp   = hr if hr <= 12 else hr - 12
    disp   = 12 if disp == 0 else disp
    print(f"  {disp:2d}:xx {suffix}: {'█' * cnt} ({cnt})")

print(f"\nDuring market hours (9:30-16:00 ET): "
      f"{df['is_market_hours'].sum()} / {len(df)}")

if errors:
    print("\nFiles with errors:")
    for e in errors:
        print(f"  {e['file']}: {e['error']}")
