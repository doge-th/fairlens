#!/usr/bin/env python3
"""Fetch the LendingClub accepted-loans dataset (loan_two variant, 396,030 rows,
2007–2018Q4, loan_status ∈ {Fully Paid, Charged Off}) into
data/lending_club_loan_two.csv.

Source: Hugging Face dataset ghty123/lending_club_loan_dataset (public mirror of the
classic Kaggle cleaned version). Run once:
    python3 scripts/fetch_lendingclub.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "lending_club_loan_two.csv")
URL = "https://hf-mirror.com/datasets/ghty123/lending_club_loan_dataset/resolve/main/lending_club_loan_two.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
if os.path.exists(OUT) and os.path.getsize(OUT) > 90_000_000:
    print("already present:", OUT)
    sys.exit(0)

print("downloading (~100 MB)...")
subprocess.run(["curl", "-sL", "--max-time", "900", "-o", OUT, URL], check=True)
size = os.path.getsize(OUT)
assert size > 90_000_000, f"download too small: {size} bytes"
print(f"saved {OUT}  ({size/1e6:.1f} MB)")
