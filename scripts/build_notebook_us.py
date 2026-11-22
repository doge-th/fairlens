#!/usr/bin/env python3
"""Build notebooks/fairlens-lendingclub.ipynb from scripts/run_analysis_us.py and
execute it so outputs are real. Mirrors build_notebook.py for the 2020 study."""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "scripts", "run_analysis_us.py")
OUT = os.path.join(ROOT, "notebooks", "fairlens-lendingclub.ipynb")

src = open(SRC).read()
src = src.replace(
    'ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',
    'ROOT = os.path.abspath("..")  # notebook lives in notebooks/')
src = re.sub(r'^#![^\n]*\n\s*""".*?"""\s*\n', '', src, count=1, flags=re.S)

MARKS = [
    ("# ---------- 1. load + feature engineering ----------", "data"),
    ("# ---------- 2. split 60/20/20 ----------", "split"),
    ("# ---------- 3. baselines ----------", "baselines"),
    ("# ---------- 4. models ----------", "models"),
    ("# ---------- 5. empirical cost pricing", "pricing"),
    ("# ---------- 6. fairness audit", "audit"),
    ("# ---------- 7. proxy experiment", "proxy"),
    ("# ---------- 8. policies", "policies"),
    ("# ---------- 9. charts ----------", "charts"),
    ("# ---------- 10. results_us.json ----------", "results"),
]
pos = [src.find(m[0]) for m in MARKS]
assert all(x > 0 for x in pos), pos
sections = {}
for i, (mark, name) in enumerate(MARKS):
    end = pos[i + 1] if i + 1 < len(pos) else len(src)
    sections[name] = src[src.index("\n", pos[i]) + 1:end].strip("\n")
prelude = src[:pos[0]].strip("\n")

charts = sections["charts"]
fig_marks = ["# fig_us1:", "# fig_us2:", "# fig_compare:"]
fpos = [charts.find(m) for m in fig_marks]
assert all(x >= 0 for x in fpos), fpos
fig_cells = []
for i, m in enumerate(fig_marks):
    end = fpos[i + 1] if i + 1 < len(fig_marks) else len(charts)
    body = charts[fpos[i]:end].strip("\n")
    body = re.sub(r"fig\.savefig\((f\"[^\"]+\"|\"[^\"]+\")\); plt\.close\(fig\)",
                  r"fig.savefig(\1)\nplt.show()", body)
    fig_cells.append(body)

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines) + "\n"}

def code(body):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": body + "\n"}

cells = [
    md("# FairLens follow-up — the same audit on 2020s US marketplace lending",
       "",
       "**Track:** AI + Finance (follow-up to the German study) · **Team:** Hao Tan · "
       "**Dataset:** LendingClub accepted loans 2007–2018Q4 (396,030 rows, loan_two variant)",
       "",
       "The German study (`fairlens-analysis.ipynb`) audited a 1994 credit model and found a +27 pp "
       "approval gap against under-25 applicants, proxy reconstruction of removed age, and threshold "
       "policies that were statistically fragile at n=200. This notebook re-runs the **same protocol** "
       "on LendingClub — where the application contains **no protected attributes at all** — and asks "
       "what survives: the audit dimension shifts to employment length (early-career ≤2y/unreported "
       "vs established ≥3y), and the cost ratio is estimated from the data instead of a document.",
       "",
       "> This project is a prototype and is not financial advice."),
    md("## 0 · Setup"),
    code("%matplotlib inline\n" + prelude),
    md("## 1 · Data: 396,030 issued loans (2007–2018Q4)",
       "",
       "Application-stage features only: LendingClub's own outputs (grade, sub_grade, int_rate, "
       "installment, list status) are post-decision and excluded. Target: Charged Off = bad risk."),
    code(sections["data"]),
    md("## 2 · Split: 60/20/20, policies selected on validation"),
    code(sections["split"]),
    md("## 3 · Baseline"),
    code(sections["baselines"]),
    md("## 4 · Model + honest difficulty check",
       "",
       "LendingClub default risk is much harder to predict than 1994 German credit risk "
       "(AUC ≈ 0.68 vs 0.79). At the default 0.5 threshold the model barely filters anything — "
       "which is itself an audit finding, as the approval chart shows."),
    code(sections["models"]),
    md("## 5 · Pricing the cost from the data",
       "",
       "The German brief documented a 1:5 cost ratio. LendingClub documents none, so we estimate it: "
       "charge-off loss (≈50% of principal) vs forgone investor interest. The data implies ≈2.6:1."),
    code(sections["pricing"]),
    md("## 6 · The audit: early-career vs established borrowers",
       "",
       "With ~22k early-career rows in test, the bootstrap CI is tight — whatever gap we measure, "
       "we can be confident about it (unlike the German study's wide CI)."),
    code(sections["audit"]),
    md("## 7 · Does hiding employment length help? (proxy experiment)"),
    code(sections["proxy"]),
    md("## 8 · The fairness–cost map, validation-selected",
       "",
       "Same protocol as the German study: non-degeneracy floor (both groups ≥40% approval), "
       "landmarks selected on validation, reported once on test."),
    code(sections["policies"]),
    md("### The map"),
    code(fig_cells[1]),
    md("### Approval chart & the cross-market comparison"),
    code(fig_cells[0]),
    code(fig_cells[2]),
    md("## 9 · Save every number"),
    code(sections["results"]),
    md("## 10 · What the two markets teach together",
       "",
       "**Germany 1994 (n=1,000):** +27 pp gap, wide CI, parity policies fragile — they did not "
       "transfer from validation to test.",
       "",
       "**US 2020 (n=396,030):** the thin-file gap is tiny (+0.9 pp, CI [+0.7, +1.1]) and the "
       "validation-selected parity policy **does transfer** (test gap +0.1 pp) while cutting cost "
       "by ~10%. The German fragility was a small-sample phenomenon, not a market property.",
       "",
       "**And one invariant across both markets:** the fair policy is *cheaper* than the default "
       "(cost ratio 0.68 in Germany, 0.90 in the US) — because in both datasets the default 0.5 "
       "threshold approves far too many bad risks, and tightening (while keeping parity) saves money. "
       "Choosing the metric — accuracy vs cost vs recall — changes every conclusion, in both eras."),
    md("## Sources",
       "",
       "- Data: [LendingClub loan data (loan_two cleaned variant)](https://huggingface.co/datasets/ghty123/lending_club_loan_dataset), mirror of the classic Kaggle release (2007–2018Q4, public)",
       "- Fairness background: Kamiran & Calders (2009); Hardt, Price & Srebro (2016)",
       "- Tooling: scikit-learn, pandas, matplotlib (pinned in `requirements.txt`)",
       "- AI tools: ZCode (AI coding assistant) wrote and debugged the analysis under the team's direction; all results reproduce from this notebook and `scripts/run_analysis_us.py`."),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.9.6"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"us-{i:02d}"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)

import nbformat
from nbclient import NotebookClient
nb_node = nbformat.read(OUT, as_version=4)
client = NotebookClient(nb_node, timeout=1200, kernel_name="python3",
                        resources={"metadata": {"path": os.path.join(ROOT, "notebooks")}})
client.execute()
nbformat.write(nb_node, OUT)
print("notebook executed & saved:", OUT)
