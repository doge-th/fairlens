#!/usr/bin/env python3
"""Fetch the Statlog German Credit dataset (UCI id=144) into data/german_credit.csv.

Run once before `scripts/run_analysis.py` (which reads the local CSV):
    python3 scripts/fetch_data.py
"""
import os

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

ds = fetch_ucirepo(id=144)  # Statlog (German Credit Data)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
out = os.path.join(ROOT, "data", "german_credit.csv")
df.to_csv(out, index=False)
print(f"saved {out}  shape={df.shape}  class counts={df['class'].value_counts().to_dict()}")
