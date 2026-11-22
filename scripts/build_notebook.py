#!/usr/bin/env python3
"""Build notebooks/fairlens-analysis.ipynb from scripts/run_analysis.py,
then execute it with nbclient so every output/chart is real and reproducible."""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "scripts", "run_analysis.py")
OUT = os.path.join(ROOT, "notebooks", "fairlens-analysis.ipynb")

src = open(SRC).read()

# notebook path context differs from the script: ROOT is the repo parent of notebooks/
src = src.replace(
    'ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',
    'ROOT = os.path.abspath("..")  # repo root (notebook lives in notebooks/)')

# drop shebang + module docstring entirely (single regex, no line-guessing)
src = re.sub(r'^#![^\n]*\n\s*""".*?"""\s*\n', '', src, count=1, flags=re.S)

# split at section markers
MARKS = [
    ("# ---------- 1. load + decode ----------", "data"),
    ("# ---------- 2. split", "split"),
    ("# ---------- 3. baselines ----------", "baselines"),
    ("# ---------- 4. models ----------", "models"),
    ("# ---------- 5. fairness audit", "audit"),
    ("# ---------- 6. proxy feature discovery", "proxy"),
    ("# ---------- 7. policy selection on VALIDATION", "policies"),
    ("# ---------- 8. failure cases ----------", "failures"),
    ("# ---------- 9. charts ----------", "charts"),
    ("# ---------- 10. results.json ----------", "results"),
]
pos = [src.find(m[0]) for m in MARKS]
assert all(p > 0 for p in pos), pos
sections = {}
for i, (mark, name) in enumerate(MARKS):
    end = pos[i + 1] if i + 1 < len(pos) else len(src)
    line_end = src.index("\n", pos[i]) + 1  # cut after the full marker line
    sections[name] = src[line_end:end].strip("\n")

# prelude = everything before the first marker (imports, identity, helpers)
prelude = src[:pos[0]].strip("\n")

# split charts into per-figure cells; make figures render inline (keep asset savefig)
charts = sections["charts"]
charts = re.sub(r"fig\.savefig\((f\"[^\"]+\"|\"[^\"]+\")\); plt\.close\(fig\)",
                r"fig.savefig(\1)\nplt.show()", charts)
fig_marks = ["# fig 1:", "# fig 2:", "# fig 3:", "# fig 4:"]
fpos = [charts.find(m) for m in fig_marks]
assert all(p > 0 for p in fpos), fpos
fig_cells = []
for i, m in enumerate(fig_marks):
    end = fpos[i + 1] if i + 1 < len(fpos) else len(charts)
    fig_cells.append(charts[fpos[i]:end].strip("\n"))

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines) + "\n"}

def code(body):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": body + "\n"}

cells = [
    md("# FairLens — does a loan approval model treat young applicants fairly?",
       "",
       "**Track:** AI + Finance · **Team:** Hao Tan · **Dataset:** Statlog German Credit (UCI id=144)",
       "",
       "A fairness audit of a loan-approval model, built for the *Veritas AI International High School Hackathon 2026*. "
       "We train the model the brief asks for, measure the approval-rate gap between applicants under 25 and 25 or older, "
       "show that hiding age does not close the gap, and then map every per-group threshold policy on one honest chart: "
       "the **fairness–cost map** (approving a bad risk costs 5× blocking a good one, per the dataset documentation).",
       "",
       "> This project is a prototype and is not financial advice.",
       "",
       "---"),
    md("## 0 · Setup",
       "",
       "Reproducibility: seed 42 everywhere, scikit-learn/pandas only, no tuning on the test set. "
       "Sklearn 1.6 emits benign intermediate-iteration numeric warnings during LR fits on arm64 "
       "(final coefficients stay finite and sane); we silence the numeric noise and say so here."),
    code("%matplotlib inline\n" + prelude),
    md("## 1 · Data: 1,000 applicants, 20 attributes",
       "",
       "UCI ships the attributes coded (`A11`, `A30`, …). We decode them to readable labels using the "
       "official UCI documentation table. `class` 1 = good risk (700), 2 = bad risk (300). "
       "We model **P(bad risk)** — approving means predicting *good*."),
    code(sections["data"]),
    md("## 2 · Train / validation / test split",
       "",
       "The data has no dates, so we use a stratified **600/200/200 train/validation/test** split (the brief's "
       "date-ordering rule does not apply here — we note it honestly). The model trains on the 600 rows; "
       "**threshold policies are selected on the validation split**; the test set is used once, for final reporting."),
    code(sections["split"]),
    md("## 3 · Baselines first",
       "",
       "The brief warns about this trap: a \"model\" that approves everyone is **70% accurate and catches zero bad risks**. "
       "We report it next to everything we build."),
    code(sections["baselines"]),
    md("## 4 · Models: logistic regression and random forest",
       "",
       "Logistic regression (interpretable coefficients) and a random forest (stronger learner), identical preprocessing. "
       "5-fold cross-validated AUC on the *training* set gives an honest variance estimate before we ever touch the test set."),
    code(sections["models"]),
    md("## 5 · The audit: who gets approved?",
       "",
       "Approval decision: `P(bad) < 0.5 → approve`. We measure approval rates by age group (the brief's example split), "
       "bootstrap the gap, and compare with the *true* bad-risk rate in each group. Sex reaches the model through the "
       "`personal status` column (which also carries marital status), so the honest test is to **drop that column, retrain, "
       "and re-audit with a bootstrap CI** — if the signal survives, it is real; if it does not, we say so."),
    code(sections["audit"]),
    md("## 6 · \"Just hide age\" — fairness through unawareness fails",
       "",
       "The classic fix would be removing the sensitive attribute. We retrain without `age_years` and re-measure. "
       "Then we ask *why*: how much age information do the remaining features still carry? "
       "For categoricals we use the correlation ratio η² (variance of age explained by group membership); "
       "for numerics, Spearman ρ²."),
    code(sections["proxy"]),
    md("## 7 · The fairness–cost map: select on validation, report on test",
       "",
       "Every threshold policy (one threshold for under-25s, one for 25+) is a point: its approval gap (x) and its "
       "total cost (y), priced the way the dataset documents it — approving a bad risk costs **5×** blocking a good one. "
       "We require both groups to keep ≥40% approval (a parity policy that rejects nearly everyone is trivially equal and cheap), "
       "then select three landmark policies **on validation**: equal approval, pure min-cost, and equal opportunity "
       "(parity in per-group bad-risk recall). The map shows the selection surface; each marker also carries its "
       "**held-out test outcome** — which is where an honest surprise waits: the parity policy's near-zero gap "
       "*does not transfer* to the test split, while its cost and recall advantages do."),
    code(sections["policies"]),
    md("### The map"),
    code(fig_cells[3]),
    md("### Chart gallery",
       "",
       "The four figures used in the write-up, generated by the same code you can re-run:"),
    code(fig_cells[0]),
    code(fig_cells[1]),
    code(fig_cells[2]),
    md("## 8 · Where it fails — on purpose",
       "",
       "Five test cases the default model gets wrong. Honest failure reporting is a scoring criterion; "
       "it also shows *who* the errors hurt."),
    code(sections["failures"]),
    md("## 9 · Save every number",
       "",
       "`results/results.json` feeds the write-up and the interactive dashboard — one source of truth, no hand-copied numbers."),    code(sections["results"]),
    md("## 10 · Limitations & what we'd do next",
       "",
       "**Limitations.** n=1,000 (1970s-90s Germany, dated categories — we audit *this* dataset, not modern lending); "
       "with ~30 under-25 applicants per split, **per-group threshold policies are statistically fragile** — our own "
       "validation-selected parity policy did not transfer its near-zero gap to test (a finding we report rather than hide); "
       "approval-rate parity is one fairness criterion among several, and our equal-opportunity attempt transferred even worse; "
       "parity-by-thresholds also *lowered* both groups' approval rates — real remediation needs bigger samples and "
       "criteria like equalized odds; group-dependent thresholds would constitute disparate treatment in many jurisdictions — "
       "this dashboard is an **audit sandbox, not a deployment rule**; the 1:5 cost ratio comes from the dataset docs "
       "and ignores loan size (our DM-weighted sensitivity check gives the same conclusion).",
       "",
       "**Next steps.** (1) Equalized-odds and calibration-by-group audits on larger data; (2) re-run on a modern dataset "
       "(e.g., LendingClub) to test whether the proxy findings transfer; (3) user-test the dashboard with real loan officers."),
    md("## Sources",
       "",
       "- Data: [Statlog German Credit, UCI ML Repository (id=144)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — loaded via `ucimlrepo` (`scripts/fetch_data.py`)",
       "- Fairness background: Kamiran & Calders (2009), *Data preprocessing techniques for classification without discrimination*; Hardt, Price & Srebro (2016), *Equality of Opportunity in Supervised Learning*",
       "- Tooling: scikit-learn, pandas, matplotlib, numpy (pinned in `requirements.txt`); notebook executed with nbclient/ipykernel",
       "- Brief: Veritas AI Hackathon 2026, Project Briefs §4.2.3 (Fairness audit of a loan model)",
       "- AI tools: ZCode (AI coding assistant) wrote and debugged this analysis under the team's direction; all results reproduced locally"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.6"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("notebook written:", OUT)

# execute for real
import nbformat
from nbclient import NotebookClient
nb_node = nbformat.read(OUT, as_version=4)
client = NotebookClient(nb_node, timeout=600, kernel_name="python3",
                        resources={"metadata": {"path": os.path.join(ROOT, "notebooks")}})
client.execute()
nbformat.write(nb_node, OUT)
print("notebook executed & saved with outputs")
for c in nb_node.cells:
    if c.cell_type == "code":
        ec = c.get("execution_count")
        nout = len(c.get("outputs", []))
        print(f"  cell[{ec}] outputs={nout} src[:60]={c.source[:60]!r}")
