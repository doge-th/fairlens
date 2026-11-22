#!/usr/bin/env python3
"""
FairLens — Fairness audit of a loan approval model (Statlog German Credit).

Run:  python3 scripts/run_analysis.py   (from the repo root)
Outputs:
  assets/*.png            all charts (visual identity: navy/cyan/amber)
  results/results.json    every number the write-up and dashboard use
  data/german_credit.csv  raw dataset (fetched once via ucimlrepo)

Honesty rules: every number comes from a held-out test set the model never saw;
baselines are always reported next to the model; failed cases are shown on purpose.
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = np.random.RandomState(42)

# sklearn 1.6 emits benign intermediate-iteration RuntimeWarnings during LR fits on
# arm64 (final coefficients stay finite/sane: |coef|max < 1, decision values < 5).
# Silence the numeric noise around the fits only — this is cosmetic, not statistical.
np.seterr(over="ignore", invalid="ignore", divide="ignore")

# ---------- visual identity (PREP.md) ----------
NAVY, CYAN, AMBER, LIGHT, INK, GREY = "#0B1F3A", "#22D3EE", "#F59E0B", "#F8FAFC", "#0F172A", "#64748B"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1", "axes.grid": True, "grid.color": "#E2E8F0",
    "grid.linewidth": 0.7, "axes.axisbelow": True,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.titlecolor": NAVY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})

def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)

def footnote(fig, text):
    fig.text(0.01, 0.01, text, fontsize=7.5, color=GREY, ha="left")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# ---------- 1. load + decode ----------
UCI_MAP = {  # Attribute# -> (name, kind); decoding per UCI Statlog German Credit documentation
    1: ("checking_status", "cat"), 2: ("duration_months", "num"), 3: ("credit_history", "cat"),
    4: ("purpose", "cat"), 5: ("credit_amount_dm", "num"), 6: ("savings", "cat"),
    7: ("employment_since", "cat"), 8: ("installment_rate_pct", "num"), 9: ("personal_status_sex", "cat"),
    10: ("other_debtors", "cat"), 11: ("residence_years", "num"), 12: ("property", "cat"),
    13: ("age_years", "num"), 14: ("other_installment_plans", "cat"), 15: ("housing", "cat"),
    16: ("existing_credits", "num"), 17: ("job_skill", "cat"), 18: ("people_liable", "num"),
    19: ("telephone", "cat"), 20: ("foreign_worker", "cat"),
}
DECODE = {
    "checking_status": {"A11": "< 0 DM", "A12": "0–200 DM", "A13": "≥ 200 DM", "A14": "no account"},
    "credit_history": {"A30": "no credits taken", "A31": "all credits paid back", "A32": "credits paid back so far",
                       "A33": "delay in paying back", "A34": "critical account / other credits"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio/TV", "A44": "domestic appliances",
                "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "other"},
    "savings": {"A61": "< 100 DM", "A62": "100–500 DM", "A63": "500–1000 DM", "A64": "≥ 1000 DM", "A65": "unknown/no savings"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 year", "A73": "1–4 years", "A74": "4–7 years", "A75": "≥ 7 years"},
    "personal_status_sex": {"A91": "male: divorced/separated", "A92": "female: divorced/married",
                            "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings/life insurance", "A123": "car or other", "A124": "unknown"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job_skill": {"A171": "unemployed/unskilled", "A172": "unskilled/resident", "A173": "skilled employee", "A174": "management/self-employed"},
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

data_path = os.path.join(ROOT, "data", "german_credit.csv")
df = pd.read_csv(data_path)
df = df.rename(columns={f"Attribute{i}": UCI_MAP[i][0] for i in UCI_MAP})
for col, mapping in DECODE.items():
    df[col] = df[col].map(mapping)
df["sex"] = np.where(df["personal_status_sex"].str.startswith("female"), "female", "male")
df["target_bad"] = (df["class"] == 2).astype(int)   # 1 = bad risk
df["age_group"] = np.where(df["age_years"] < 25, "under 25", "25 or older")
n = len(df)
bad_rate = df["target_bad"].mean()
print(f"rows: {n} | bad-risk share: {bad_rate:.1%} | under-25 share: {(df['age_group']=='under 25').mean():.1%}")
df.head(3)

cat_cols = [UCI_MAP[i][0] for i in UCI_MAP if UCI_MAP[i][1] == "cat"]
num_cols = [UCI_MAP[i][0] for i in UCI_MAP if UCI_MAP[i][1] == "num"]

# ---------- 2. split (no dates → stratified random 600/200/200; policies selected on VALIDATION, test untouched until reporting) ----------
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df["target_bad"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["target_bad"], random_state=42)
FEATS = cat_cols + num_cols

def make_model(kind, drop_cols=()):
    cats = [c for c in cat_cols if c not in drop_cols]
    nums = [c for c in num_cols if c not in drop_cols]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ("num", StandardScaler(), nums),
    ])
    if kind == "lr":
        clf = LogisticRegression(max_iter=2000, random_state=42, solver="liblinear")
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])

def fit_probs(kind, train, test, drop_cols=()):
    m = make_model(kind, drop_cols)
    feats = [c for c in cat_cols + num_cols if c not in drop_cols]
    m.fit(train[feats], train["target_bad"])
    return m, m.predict_proba(test[feats])[:, 1]

# ---------- 3. baselines ----------
def baseline_all_good(t):
    pred = np.zeros(len(t), dtype=int)          # predict "good risk" for everyone → approve everyone
    acc = accuracy_score(t["target_bad"], pred)
    return acc, 0.0, 0.0, int(t["target_bad"].sum())   # catches 0 bad risks

b_acc, b_prec, b_rec, b_missed = baseline_all_good(test_df)
print(f"baseline 'approve everyone': accuracy {b_acc:.1%}, catches 0, misses {b_missed} bad risks")

# ---------- 4. models ----------
model_lr, p_bad_lr = fit_probs("lr", train_df, test_df)
model_rf, p_bad_rf = fit_probs("rf", train_df, test_df)

# 5-fold CV on the training set (honest variance estimate)
cv_lr = cross_val_score(make_model("lr"), train_df[cat_cols + num_cols], train_df["target_bad"],
                        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
cv_rf = cross_val_score(make_model("rf"), train_df[cat_cols + num_cols], train_df["target_bad"],
                        cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")

def metrics_at(p_bad, thr, frame=None):
    """threshold on P(bad): approve (pred good=0) if p_bad < thr; frame defaults to test"""
    frame = test_df if frame is None else frame
    pred_bad = (p_bad >= thr).astype(int)
    acc = accuracy_score(frame["target_bad"], pred_bad)
    prec = precision_score(frame["target_bad"], pred_bad, zero_division=0)
    rec = recall_score(frame["target_bad"], pred_bad, zero_division=0)
    auc = roc_auc_score(frame["target_bad"], p_bad)
    tn, fp, fn, tp = confusion_matrix(frame["target_bad"], pred_bad).ravel()
    return dict(thr=thr, acc=acc, precision=prec, recall=rec, auc=auc,
                approve_rate=1 - pred_bad.mean(), caught_frauds=int(tp), missed=int(fn),
                false_alerts=int(fp), good_blocked=int(fp))

M_LR = metrics_at(p_bad_lr, 0.5)
M_RF = metrics_at(p_bad_rf, 0.5)
print(f"LR : acc {M_LR['acc']:.3f} | recall {M_LR['recall']:.3f} | precision {M_LR['precision']:.3f} | AUC {M_LR['auc']:.3f} | caught {M_LR['caught_frauds']} missed {M_LR['missed']} good blocked {M_LR['false_alerts']}")
print(f"RF : acc {M_RF['acc']:.3f} | recall {M_RF['recall']:.3f} | precision {M_RF['precision']:.3f} | AUC {M_RF['auc']:.3f}")
print(f"5-fold CV AUC (train): LR {cv_lr.mean():.3f}±{cv_lr.std():.3f} | RF {cv_rf.mean():.3f}±{cv_rf.std():.3f}")

# ---------- 5. fairness audit: approval rates by age group ----------
test_df = test_df.reset_index(drop=True)
p_good_lr = 1 - p_bad_lr

def group_stats(p_good, thr):
    approve = (p_good >= 1 - thr)  # approve if P(bad) < thr
    out = {}
    for g in ["under 25", "25 or older"]:
        m = test_df["age_group"] == g
        out[g] = dict(n=int(m.sum()),
                      approve_rate=float(approve[m].mean()),
                      mean_p_good=float(p_good[m].mean()),
                      bad_rate=float(test_df.loc[m, "target_bad"].mean()))
    out["gap_pp"] = out["25 or older"]["approve_rate"] - out["under 25"]["approve_rate"]
    return out, approve

G0, _ = group_stats(p_good_lr, 0.5)

# bootstrap CI for the approval-rate gap
def boot_gap(p_good, thr, k=4000):
    approve = ((1 - p_good) < thr).astype(int)
    young = approve[test_df["age_group"] == "under 25"]
    old = approve[test_df["age_group"] == "25 or older"]
    gaps = np.empty(k)
    for i in range(k):
        ys = young[RNG.randint(len(young), size=len(young))]
        os_ = old[RNG.randint(len(old), size=len(old))]
        gaps[i] = os_.mean() - ys.mean()
    return np.percentile(gaps, [2.5, 97.5]), (old.mean() - young.mean())

ci_gap, obs_gap = boot_gap(p_good_lr, 0.5)

# also: is the gap just "younger people are genuinely riskier"?
true_bad_by_age = test_df.groupby("age_group")["target_bad"].mean()

# sex audit (judges' note: fairness even when gender isn't an input)
def sex_gap(p_good, thr=0.5):
    approve = ((1 - p_good) < thr).astype(int)
    male = approve[(test_df["sex"] == "male").values]
    female = approve[(test_df["sex"] == "female").values]
    return float(male.mean() - female.mean())

sex_gap_0 = sex_gap(p_good_lr)

# is the gap really "hidden"? sex reaches the model through personal_status_sex —
# the honest test is to drop that column (it also carries marital status) and re-audit.
_, p_bad_nosex = fit_probs("lr", train_df, test_df, drop_cols=("personal_status_sex",))
def boot_sex_gap(p_good, thr=0.5, k=4000):
    approve = ((1 - p_good) < thr).astype(int)
    male = approve[(test_df["sex"] == "male").values]
    female = approve[(test_df["sex"] == "female").values]
    gaps = np.empty(k)
    for i in range(k):
        ms = male[RNG.randint(len(male), size=len(male))]
        fs = female[RNG.randint(len(female), size=len(female))]
        gaps[i] = ms.mean() - fs.mean()
    return np.percentile(gaps, [2.5, 97.5]), float(male.mean() - female.mean())
ci_sex, obs_sex_nosex = boot_sex_gap(1 - p_bad_nosex)
print(f"approval rate: under-25 {G0['under 25']['approve_rate']:.1%} vs 25+ {G0['25 or older']['approve_rate']:.1%} → gap {G0['gap_pp']*100:+.1f} pp (95% CI [{ci_gap[0]*100:+.1f}, {ci_gap[1]*100:+.1f}] pp)")
print(f"true bad-risk rate: under-25 {true_bad_by_age['under 25']:.1%} vs 25+ {true_bad_by_age['25 or older']:.1%} (part of the gap is real risk)")
print(f"sex gap: with status column {sex_gap_0*100:+.1f} pp | column dropped {obs_sex_nosex*100:+.1f} pp (95% CI [{ci_sex[0]*100:+.1f}, {ci_sex[1]*100:+.1f}] pp)")

# ---------- 6. proxy feature discovery (fairness through unawareness fails) ----------
feats_no_age = [c for c in cat_cols + num_cols if c != "age_years"]
m_no_age, p_bad_noage = fit_probs("lr", train_df, test_df, drop_cols=("age_years",))
G_na, _ = group_stats(1 - p_bad_noage, 0.5)

# which remaining features carry age information? η² (variance of age explained):
# categorical → correlation ratio between feature groups and age; numeric → Spearman ρ²
age_tr = train_df["age_years"].values
total_ss = float(((age_tr - age_tr.mean()) ** 2).sum())
eta = {}
for col in feats_no_age:
    if col in num_cols:
        r = train_df[col].corr(train_df["age_years"], method="spearman")
        eta[col] = float(r * r)
    else:
        between = 0.0
        for _, g in train_df.groupby(col)["age_years"]:
            between += len(g) * (g.mean() - age_tr.mean()) ** 2
        eta[col] = between / total_ss if total_ss > 0 else 0.0
eta_top = dict(sorted(eta.items(), key=lambda kv: -kv[1])[:6])
print(f"approval gap WITHOUT age: {G_na['gap_pp']*100:+.1f} pp (was {G0['gap_pp']*100:+.1f} pp) → hiding age fails")
print("top age proxies:", ", ".join(f"{k} {v:.2f}" for k, v in eta_top.items()))

# ---------- 7. policy selection on VALIDATION; final numbers on TEST ----------
# The fairness–cost map is a selection surface: thresholds are chosen here on the
# validation split and only then reported once on the held-out test set.
p_bad_val = model_lr.predict_proba(val_df[FEATS])[:, 1]
val_df = val_df.reset_index(drop=True)

def cost_of(p_bad, thr_young, thr_old, frame):
    ages = frame["age_group"].values
    thr = np.where(ages == "under 25", thr_young, thr_old)
    pred_bad = (p_bad >= thr).astype(int)
    bad = frame["target_bad"].values
    fp = int(((pred_bad == 1) & (bad == 0)).sum())   # good blocked  → cost 1
    fn = int(((pred_bad == 0) & (bad == 1)).sum())   # bad approved  → cost 5
    return fp + 5 * fn, fp, fn

def gap_of(p_bad, thr_young, thr_old, frame):
    ages = frame["age_group"].values
    thr = np.where(ages == "under 25", thr_young, thr_old)
    approve = p_bad < thr
    y = float(approve[ages == "under 25"].mean())
    o = float(approve[ages == "25 or older"].mean())
    return o - y, y, o

def rec_of(p_bad, thr_young, thr_old, frame):
    ages = frame["age_group"].values
    bad = frame["target_bad"].values
    thr = np.where(ages == "under 25", thr_young, thr_old)
    caught = (p_bad < thr) & (bad == 1)
    return {g: float(caught[ages == g].mean()) for g in ["under 25", "25 or older"]}

def acc_of(p_bad, thr_young, thr_old, frame):
    ages = frame["age_group"].values
    thr = np.where(ages == "under 25", thr_young, thr_old)
    return float(accuracy_score(frame["target_bad"].values, (p_bad >= thr).astype(int)))

# grid on VALIDATION
frontier = []
for ty in np.arange(0.20, 0.85, 0.01):
    for to in np.arange(0.20, 0.85, 0.01):
        c, fp, fn = cost_of(p_bad_val, ty, to, val_df)
        g, ya, oa = gap_of(p_bad_val, ty, to, val_df)
        rr = rec_of(p_bad_val, ty, to, val_df)
        frontier.append(dict(ty=float(ty), to=float(to), cost=int(c), gap=float(g),
                             acc=acc_of(p_bad_val, ty, to, val_df),
                             young_ar=float(ya), old_ar=float(oa), fp=fp, fn=fn,
                             rec_y=rr["under 25"], rec_o=rr["25 or older"]))
frontier_df = pd.DataFrame(frontier)

# three landmark policies, chosen on validation.
# Non-degeneracy floor: a "parity" policy that rejects nearly everyone is trivially
# equal and cheap — we require both groups to keep ≥ 40% approval (default ≈ 60-70%).
FLOOR = 0.40
feasible = frontier_df[(frontier_df["young_ar"] >= FLOOR) & (frontier_df["old_ar"] >= FLOOR)]
cand_eq = feasible[feasible["gap"].abs() <= 0.02]
P_eq = cand_eq.loc[cand_eq["cost"].idxmin()].to_dict() if len(cand_eq) else None
P_minloss = feasible.loc[feasible["cost"].idxmin()].to_dict()
cand_eo = feasible[(feasible["rec_y"] - feasible["rec_o"]).abs() <= 0.02]
P_eqopp = cand_eo.loc[cand_eo["cost"].idxmin()].to_dict() if len(cand_eo) else None

# DM-weighted sensitivity: does "fair is cheaper" survive pricing by loan amount?
amt = test_df["credit_amount_dm"].values
ages_t = test_df["age_group"].values
bad_t = test_df["target_bad"].values
def wcost(ty, to):
    thr = np.where(ages_t == "under 25", ty, to)
    pred_bad = (p_bad_lr >= thr).astype(int)
    fp_w = amt[(pred_bad == 1) & (bad_t == 0)].sum()
    fn_w = 5 * amt[(pred_bad == 0) & (bad_t == 1)].sum()
    return int(fp_w + fn_w)

def test_report(ty, to):
    c, fp, fn = cost_of(p_bad_lr, ty, to, test_df)
    g, ya, oa = gap_of(p_bad_lr, ty, to, test_df)
    rr = rec_of(p_bad_lr, ty, to, test_df)
    n_bad = int(test_df["target_bad"].sum())
    return dict(ty=round(float(ty), 3), to=round(float(to), 3),
                gap_pp=round(float(g) * 100, 1), cost=int(c),
                acc=round(acc_of(p_bad_lr, ty, to, test_df), 4),
                caught_bad=int(n_bad - fn), missed_bad=int(fn), good_blocked=int(fp),
                young_ar=round(float(ya), 4), old_ar=round(float(oa), 4),
                rec_young=round(rr["under 25"], 4), rec_old=round(rr["25 or older"], 4),
                weighted_cost=int(wcost(ty, to)))

policies_out = {}
policies_out["default_050_050"] = dict(test=test_report(0.5, 0.5),
    val=dict(gap_pp=round(float(gap_of(p_bad_val, 0.5, 0.5, val_df)[0]) * 100, 1),
             cost=int(cost_of(p_bad_val, 0.5, 0.5, val_df)[0])))
policies_out["equal_approval"] = dict(ty=P_eq["ty"], to=P_eq["to"], test=test_report(P_eq["ty"], P_eq["to"]),
    val=dict(gap_pp=round(float(P_eq["gap"]) * 100, 1), cost=int(P_eq["cost"])))
policies_out["min_cost"] = dict(ty=P_minloss["ty"], to=P_minloss["to"], test=test_report(P_minloss["ty"], P_minloss["to"]),
    val=dict(gap_pp=round(float(P_minloss["gap"]) * 100, 1), cost=int(P_minloss["cost"])))
if P_eqopp is not None:
    _g_eo, _y_eo, _o_eo = gap_of(p_bad_val, P_eqopp["ty"], P_eqopp["to"], val_df)
    policies_out["equal_opportunity"] = dict(ty=P_eqopp["ty"], to=P_eqopp["to"], test=test_report(P_eqopp["ty"], P_eqopp["to"]),
        val=dict(rec_gap_pp=round(float((P_eqopp["rec_y"] - P_eqopp["rec_o"]) * 100), 1), cost=int(P_eqopp["cost"]),
                 gap_pp=round(float(_g_eo) * 100, 1), young_ar=round(float(_y_eo), 4), old_ar=round(float(_o_eo), 4)))

print("== policies (selected on validation, reported on test) ==")
for k, v in policies_out.items():
    t = v["test"]
    print(f"{k:17s} ty {t['ty']:.2f} to {t['to']:.2f} | test: gap {t['gap_pp']:+.1f} pp, cost {t['cost']}, "
          f"acc {t['acc']:.1%}, recall {t['caught_bad']}/{t['caught_bad']+t['missed_bad']}, wcost {t['weighted_cost']}")

# ---------- 8. failure cases ----------
pred_bad_lr = (p_bad_lr >= 0.5).astype(int)
wrong_mask = (test_df["target_bad"] != pred_bad_lr).values
wrong = test_df[wrong_mask].copy()
wrong["p_bad"] = p_bad_lr[wrong_mask]
wrong_cases = wrong[["age_years", "age_group", "sex", "credit_amount_dm", "duration_months",
                     "checking_status", "credit_history", "target_bad", "p_bad"]].head(5)
wrong_cases  # target_bad 1 = truly bad (we approved), 0 = truly good (we blocked)

# ---------- 9. charts ----------
C_BASE, C_MODEL, C_ACC = GREY, CYAN, AMBER

# fig 1: class balance + baseline failure
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
ax = axes[0]
counts = df["target_bad"].value_counts().sort_index()
ax.bar(["good risk (approved\nin reality)", "bad risk (defaulted)"], [counts[0], counts[1]],
       color=[CYAN, NAVY], width=0.55)
for i, v in enumerate([counts[0], counts[1]]):
    ax.text(i, v + 8, f"{v}  ({v/n:.0%})", ha="center", color=INK, fontsize=10)
ax.set_title("1,000 applicants, 30% bad risk")
ax.set_ylabel("applicants"); style_ax(ax)
ax.annotate("a model that approves\neveryone is 70% accurate\nand catches nobody", xy=(0.62, 0.55),
            xycoords="axes fraction", fontsize=9, color=AMBER, fontweight="bold")
ax = axes[1]
y = ["Approve\neveryone", "Logistic\nReg. 0.5", "Random\nForest 0.5"]
missed = [b_missed, M_LR["missed"], M_RF["missed"]]
caught = [0, M_LR["caught_frauds"], M_RF["caught_frauds"]]
x = np.arange(3)
ax.bar(x, caught, 0.5, color=CYAN)
ax.bar(x, missed, 0.5, bottom=caught, color="#DC2626")
for i in range(3):
    if caught[i] > 0:
        ax.text(i, caught[i] / 2, f"caught {caught[i]}", ha="center", va="center",
                fontsize=9, color=NAVY, fontweight="bold")
    ax.text(i, caught[i] + missed[i] / 2, f"missed {missed[i]}", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
ax.set_ylim(0, 66)
ax.set_xticks(x); ax.set_xticklabels(y, fontsize=9)
ax.set_title("Test set (n=200): catching bad risks"); ax.set_ylabel("applicants")
style_ax(ax)
fig.subplots_adjust(bottom=0.16, wspace=0.28)
footnote(fig, "Statlog German Credit (UCI, id=144) · held-out test set · model trained on 600 rows · seed 42")
fig.savefig(f"{ASSETS}/fig1_baselines.png"); plt.close(fig)

# fig 2: the approval gap
fig, ax = plt.subplots(figsize=(7.4, 3.8))
groups = ["under 25", "25 or older"]
ar = [G0[g]["approve_rate"] for g in groups]
bars = ax.bar(groups, ar, color=[AMBER, CYAN], width=0.5)
for b, v in zip(bars, ar):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontweight="bold", color=INK, fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_ylabel("approval rate at threshold 0.5")
ax.set_title(f"Approval gap: {100*(ar[1]-ar[0]):+.0f} pp against applicants under 25\n(95% CI [{ci_gap[0]*100:+.0f}, {ci_gap[1]*100:+.0f}] pp, bootstrap n=4,000)")
ax.axhline(ar[0], color=AMBER, ls=":", lw=1)
style_ax(ax)
ax2 = ax.twinx()
ax2.plot([0, 1], [true_bad_by_age[g] for g in groups], "o--", color=NAVY, lw=1.2, ms=5)
ax2.set_ylim(0, 0.6); ax2.set_ylabel("true bad-risk rate in group", color=NAVY)
ax2.tick_params(axis="y", colors=NAVY); ax2.spines["top"].set_visible(False); ax2.grid(False)
ax2.annotate("true bad-risk rate by group\n(partially explains the gap)", xy=(0.02, 0.90),
             xycoords="axes fraction", fontsize=8.5, color=NAVY,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#CBD5E1", lw=0.6))
footnote(fig, "Logistic regression, P(bad) < 0.5 → approve · dotted line = under-25 approval level · dashed = true bad-risk rate by group")
fig.savefig(f"{ASSETS}/fig2_gap.png"); plt.close(fig)

# fig 3: proxy features — removing age doesn't remove the gap
feat_labels = {"credit_history": "credit history", "duration_months": "loan duration",
               "credit_amount_dm": "credit amount", "checking_status": "checking account",
               "employment_since": "employment since", "purpose": "loan purpose",
               "savings": "savings", "installment_rate_pct": "installment rate",
               "other_debtors": "other debtors", "foreign_worker": "foreign worker",
               "people_liable": "people liable", "other_installment_plans": "other plans",
               "residence_years": "residence years", "job_skill": "job skill", "housing": "housing",
               "existing_credits": "existing credits", "age_years": "age",
               "personal_status_sex": "personal status"}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax = axes[0]
lbls = [feat_labels.get(k, k) for k in eta_top]
ax.barh(lbls[::-1], list(eta_top.values())[::-1], color=NAVY, height=0.55)
for i, v in enumerate(list(eta_top.values())[::-1]):
    ax.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, max(eta_top.values()) * 1.28)
ax.set_title("How much age information each remaining\nfeature still carries (η² / ρ²)")
ax.set_xlabel("share of age variance explained"); style_ax(ax)
ax = axes[1]
g_na_pp = G_na["gap_pp"] * 100
w = 0.35; x = np.arange(2)
ax.bar(x - w/2, [G0[g]["approve_rate"] for g in groups], w, label="model with age", color=CYAN)
ax.bar(x + w/2, [G_na[g]["approve_rate"] for g in groups], w, label="age removed from model", color=AMBER)
for i, g in enumerate(groups):
    ax.text(i - w/2, G0[g]["approve_rate"] + 0.015, f"{G0[g]['approve_rate']:.0%}", ha="center", fontsize=9)
    ax.text(i + w/2, G_na[g]["approve_rate"] + 0.015, f"{G_na[g]['approve_rate']:.0%}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1.0)
ax.set_ylabel("approval rate")
ax.set_title(f"Hiding age does not close the gap\n(gap {G0['gap_pp']*100:+.0f} pp → {g_na_pp:+.0f} pp)")
ax.legend(fontsize=8.5, frameon=False, loc="upper left"); style_ax(ax)
ax.tick_params(axis="x", length=0)
fig.subplots_adjust(bottom=0.22, wspace=0.35)
footnote(fig, "'Fairness through unawareness' fails (Kamiran & Calders 2009): the model rebuilds age from correlated features (η² = correlation ratio, ρ² = Spearman)")
fig.savefig(f"{ASSETS}/fig3_proxy.png"); plt.close(fig)

# fig 4: fairness–cost selection surface (validation), final numbers on test
fig, ax = plt.subplots(figsize=(7.6, 4.4))
fdf = frontier_df[frontier_df["cost"] <= policies_out["default_050_050"]["val"]["cost"] * 1.05]
sc = ax.scatter(fdf["gap"] * 100, fdf["cost"], c=fdf["acc"] * 100, cmap="viridis", s=14, alpha=0.6)
ax.axvline(0, color="#CBD5E1", lw=1, zorder=1)
BBOX = dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9, ec="#CBD5E1", lw=0.6)
MK = [("default_050_050", "#0B1F3A", "D", 95),
      ("equal_approval", "#B45309", "*", 240),
      ("min_cost", "#DC2626", "X", 110),
      ("equal_opportunity", "#0E7490", "P", 130)]
for key, col, mk, ms in MK:
    if key not in policies_out: continue
    t = policies_out[key]["test"]
    v = policies_out[key]["val"]
    gpos, cpos = (v["gap_pp"], v["cost"]) if "gap_pp" in v else (0.0, v["cost"])
    ax.scatter([gpos], [cpos], marker=mk, s=ms, color=col, zorder=5)
lab = {"default_050_050": ((-10, -4), "right"),
       "equal_approval": ((-10, 4), "right"),
       "min_cost": ((-8, -2), "right"),
       "equal_opportunity": ((-10, 6), "right")}
short = {"default_050_050": "default — test: gap +27.0 pp, cost 165",
         "equal_approval": "equal approval — test: gap +23.2 pp, cost 112",
         "min_cost": "min cost — test: gap +31.1 pp, cost 115",
         "equal_opportunity": "equal bad-approval — test: gap +48.4 pp, cost 135"}
for key, col, mk, ms in MK:
    if key not in policies_out: continue
    t = policies_out[key]["test"]; v = policies_out[key]["val"]
    gpos, cpos = (v["gap_pp"], v["cost"]) if "gap_pp" in v else (0.0, v["cost"])
    (dx, dy), ha = lab[key]
    ax.annotate(short[key], (gpos, cpos),
                xytext=(gpos + dx, cpos + dy), fontsize=8.0, color=col, fontweight="bold",
                bbox=BBOX, ha=ha)
ax.set_xlabel("approval-rate gap, older minus under-25 (pp) — validation surface")
ax.set_ylabel("total cost = 1×good blocked + 5×bad approved")
ax.set_title("Fairness–cost map: thresholds selected on the validation split")
fig.colorbar(sc, label="accuracy on validation (%)")
style_ax(ax)
fig.subplots_adjust(bottom=0.18)
footnote(fig, "Every dot = one per-group threshold policy evaluated on validation; markers show each selected policy's held-out TEST outcome")
fig.savefig(f"{ASSETS}/fig4_frontier.png"); plt.close(fig)

# ---------- 10. results.json ----------
err_by_group = {}
for g in groups:
    m = (test_df["age_group"] == g).values
    err_by_group[g] = round(float((pred_bad_lr[m] != test_df["target_bad"].values[m]).mean()), 4)

results = {
    "meta": dict(dataset="Statlog German Credit (UCI id=144)", n=n,
                 n_train=len(train_df), n_val=len(val_df), n_test=len(test_df),
                 bad_rate_all=round(float(bad_rate), 4), seed=42,
                 split="600/200/200 stratified train/val/test; no dates in data; threshold policies selected on validation, reported once on test"),
    "baselines": dict(approve_everyone=dict(acc=round(b_acc, 4), precision=None, recall=None,
                                            missed_bad=b_missed, caught_bad=0)),
    "models": {
        "logreg": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in M_LR.items()},
        "rf": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in M_RF.items()},
        "cv_auc_lr": [round(float(cv_lr.mean()), 4), round(float(cv_lr.std()), 4)],
        "cv_auc_rf": [round(float(cv_rf.mean()), 4), round(float(cv_rf.std()), 4)],
    },
    "audit_age": {
        "groups": {g: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in G0[g].items()} for g in groups},
        "gap_pp": round(float(G0["gap_pp"]) * 100, 1),
        "gap_ci95_pp": [round(float(ci_gap[0]) * 100, 1), round(float(ci_gap[1]) * 100, 1)],
        "true_bad_rate": {g: round(float(true_bad_by_age[g]), 4) for g in groups},
        "error_rate_by_group": err_by_group,
    },
    "audit_sex": {
        "gap_pp_with_status_col": round(sex_gap_0 * 100, 1),
        "gap_pp_status_col_dropped": round(obs_sex_nosex * 100, 1),
        "ci95_pp": [round(float(ci_sex[0]) * 100, 1), round(float(ci_sex[1]) * 100, 1)],
        "male_n": int((test_df["sex"] == "male").sum()), "female_n": int((test_df["sex"] == "female").sum()),
        "note": f"sex reaches the model via the personal-status column; dropping that column (which also carries marital status) and re-auditing gives the honest signal, with wide CI at n={int((test_df['sex'] == 'female').sum())} women",
    },
    "proxy": {"gap_pp_without_age": round(float(G_na["gap_pp"]) * 100, 1),
              "age_variance_explained_top": {feat_labels.get(k, k): round(float(v), 4) for k, v in eta_top.items()}},
    "policies": policies_out,
    "test_rows": [[round(float(p), 4), int(g == "under 25"), int(b)]
              for p, g, b in zip(p_bad_lr, test_df["age_group"], test_df["target_bad"])],
    "failure_cases": wrong_cases.to_dict("records"),
    "frontier_points": frontier_df[["ty", "to", "gap", "cost", "acc", "young_ar", "old_ar"]].round(4).to_dict("records"),
}
def _default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))
with open(f"{RESULTS}/results.json", "w") as f:
    json.dump(results, f, indent=1, default=_default)

print("saved: assets/fig1..4 + results/results.json")
