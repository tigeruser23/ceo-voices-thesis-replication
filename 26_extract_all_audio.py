
#!/usr/bin/env python3

"""

26_extract_all_audio.py / 36_extract_audio_2023.py

Extract 88-dimensional eGeMAPS acoustic features from earnings call MP3s.

"""

import opensmile

import pandas as pd

import os

from pathlib import Path



data_dir  = Path(f"/scratch/network/{os.environ['USER']}/thesis_week1/data")

audio_dir = data_dir / "audio" / "processed"

out_dir   = data_dir / "audio_features"

out_dir.mkdir(exist_ok=True)



smile = opensmile.Smile(

    feature_set=opensmile.FeatureSet.eGeMAPSv02,

    feature_level=opensmile.FeatureLevel.Functionals,

)



results, errors = [], []

for mp3_path in sorted(audio_dir.glob("*.mp3")):

    stem    = mp3_path.stem

    parts   = stem.split("_")

    if len(parts) < 3: continue

    ticker  = parts[0]

    quarter = "_".join(parts[1:])

    try:

        feats = smile.process_file(str(mp3_path))

        row   = {"ticker": ticker, "quarter": quarter}

        row.update(feats.iloc[0].to_dict())

        results.append(row)

    except Exception as e:

        errors.append({"file": stem, "error": str(e)})

        print(f"  ERR {stem}: {e}")



df = pd.DataFrame(results)

df.to_csv(out_dir / "audio_features_all308.csv", index=False)

print(f"Extracted: {len(results)} calls | {df.shape[1]-2} eGeMAPS features")

