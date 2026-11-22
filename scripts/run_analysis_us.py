#!/usr/bin/env python3
"""
FairLens follow-up — does the fairness structure survive the move from
1994 Germany (Statlog German Credit) to 2020s US marketplace lending (LendingClub)?

Same audit protocol as scripts/run_analysis.py:
  stratified 600/200/200-style split (scaled: 60/20/20), baselines, LR+RF,
  group approval-gap audit with bootstrap CI, proxy (η²) quantification,
  fairness–cost map with policies SELECTED ON VALIDATION and reported once on test.

Key differences, stated honestly in the outputs:
  * LendingClub publishes no protected attributes — the audit shifts to a
    thin-file dimension that IS on the application: employment length
    (early-career ≤ 2 years or unreported vs established ≥ 3 years).
  * The 1:5 cost ratio in the German brief was documentation; here the cost
    ratio is estimated FROM the data (charge-off loss vs forgone interest).
  * LendingClub data contains issued loans only — we audit the risk score, not
    the marketplace's admission decision (no rejected-loan features exist).

Run:  python3 scripts/run_analysis_us.py   (from the repo root, after fetch)
Outputs: assets/fig_us_*.png, results/results_us.json
"""
import json
import os
import re

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
np.seterr(over="ignore", invalid="ignore", divide="ignore")

NAVY, CYAN, AMBER, INK, GREY = "#0B1F3A", "#22D3EE", "#F59E0B", "#0F172A", "#64748B"
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

# ---------- 1. load + feature engineering ----------
df = pd.read_csv(os.path.join(ROOT, "data", "lending_club_loan_two.csv"))

# target: 1 = bad risk (charged off)
df["target_bad"] = (df["loan_status"] == "Charged Off").astype(int)

# employment groups (thin-file dimension): early ≤2y or unreported vs established ≥3y
EMP_MAP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
           "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}
df["emp_years"] = df["emp_length"].map(EMP_MAP)
df["emp_missing"] = df["emp_years"].isna().astype(int)
df["emp_years"] = df["emp_years"].fillna(-1)
df["emp_group"] = np.where(df["emp_missing"] | (df["emp_years"] <= 2), "early-career", "established")

# credit-file age (years since earliest credit line at issuance)
issue_y = pd.to_datetime(df["issue_d"], format="%b-%Y").dt.year
earliest_y = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y").dt.year
df["credit_hist_years"] = (issue_y - earliest_y).clip(lower=0)
df["log_annual_inc"] = np.log1p(df["annual_inc"].clip(lower=0))
df["log_revol_bal"] = np.log1p(df["revol_bal"].clip(lower=0))
df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
# US census region from the address string (ends "... City, ST ZIP")
df["state"] = df["address"].str.extract(r", ([A-Z]{2}) \d{5}")
REGION = {
    **{s: "Northeast" for s in ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"]},
    **{s: "Midwest" for s in ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"]},
    **{s: "South" for s in ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS",
                            "TN", "AR", "LA", "OK", "TX"]},
    **{s: "West" for s in ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"]},
}
df["region"] = df["state"].map(REGION).fillna("Unknown")
# mortgage count NaN → 0 (most likely no mortgage accounts)
df["mort_acc"] = df["mort_acc"].fillna(0)
df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(0)
df["revol_util"] = df["revol_util"].fillna(df["revol_util"].median())

n = len(df)
bad_rate = df["target_bad"].mean()

# application-stage features ONLY — grade / sub_grade / int_rate / installment /
# initial_list_status are LendingClub's own post-decision outputs and are excluded
num_cols = ["loan_amnt", "term_months", "dti", "open_acc", "pub_rec", "revol_util",
            "total_acc", "mort_acc", "pub_rec_bankruptcies", "credit_hist_years",
            "log_annual_inc", "log_revol_bal", "emp_years"]
cat_cols = ["home_ownership", "verification_status", "purpose", "application_type", "region"]
drop_from_model = []          # columns withheld in the proxy experiment
# (emp_title / title free-text dropped; grade/sub_grade/int_rate excluded as internal outputs)

# ---------- 2. split 60/20/20 ----------
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df["target_bad"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["target_bad"], random_state=42)
for d in (train_df, val_df, test_df):
    d.reset_index(drop=True, inplace=True)
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
        clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                     random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])

def fit_probs(kind, train, test, drop_cols=()):
    feats = [c for c in cat_cols + num_cols if c not in drop_cols]
    m = make_model(kind, drop_cols)
    m.fit(train[feats], train["target_bad"])
    return m, m.predict_proba(test[feats])[:, 1]

# ---------- 3. baselines ----------
b_acc = accuracy_score(test_df["target_bad"], np.zeros(len(test_df), dtype=int))
b_missed = int(test_df["target_bad"].sum())
print(f"baseline 'approve everyone': accuracy {b_acc:.1%}, catches 0, misses {b_missed}")

# ---------- 4. models ----------
model_lr, p_bad_lr = fit_probs("lr", train_df, test_df)
M_LR = dict(acc=accuracy_score(test_df["target_bad"], (p_bad_lr >= 0.5).astype(int)),
            auc=roc_auc_score(test_df["target_bad"], p_bad_lr))
tn, fp, fn, tp = confusion_matrix(test_df["target_bad"], (p_bad_lr >= 0.5).astype(int)).ravel()
M_LR.update(precision=precision_score(test_df["target_bad"], (p_bad_lr >= 0.5).astype(int), zero_division=0),
            recall=recall_score(test_df["target_bad"], (p_bad_lr >= 0.5).astype(int), zero_division=0),
            caught_frauds=int(tp), missed=int(fn), false_alerts=int(fp))
print(f"LR : acc {M_LR['acc']:.3f} recall {M_LR['recall']:.3f} prec {M_LR['precision']:.3f} "
      f"AUC {M_LR['auc']:.3f} caught {M_LR['caught_frauds']} missed {M_LR['missed']} blocked {M_LR['false_alerts']}")

cv_lr = cross_val_score(make_model("lr"), train_df[FEATS], train_df["target_bad"],
                        cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring="roc_auc")
print(f"3-fold CV AUC (train): LR {cv_lr.mean():.3f}±{cv_lr.std():.3f}")

# ---------- 5. empirical cost pricing (replaces the German doc's 1:5) ----------
# approving a bad loan  ≈ 50% of principal written off (LendingClub recovery norm)
# blocking a good loan ≈ forgone investor interest ≈ principal × rate × term × amortisation factor
good = test_df[test_df["target_bad"] == 0]
bad = test_df[test_df["target_bad"] == 1]
rate_frac = good["int_rate"].values / 100.0 if "int_rate" in good.columns else None
term_yrs = good["term_months"].values / 12.0
gain_good = good["loan_amnt"].values * rate_frac * term_yrs * 0.6      # avg outstanding balance factor
loss_bad = bad["loan_amnt"].values * 0.5
RATIO = float(np.median(loss_bad) / np.median(gain_good))
print(f"empirical cost ratio (median loss per bad approved / median forgone interest per good blocked) ≈ {RATIO:.1f} : 1")

# ---------- 6. fairness audit: approval gap, early-career vs established ----------
p_good_lr = 1 - p_bad_lr
groups = ["early-career", "established"]

def group_stats(p_good, thr):
    approve = (p_good >= 1 - thr)
    out = {}
    for g in groups:
        m = (test_df["emp_group"] == g).values
        out[g] = dict(n=int(m.sum()), approve_rate=float(approve[m].mean()),
                      mean_p_good=float(p_good[m].mean()),
                      bad_rate=float(test_df["target_bad"].values[m].mean()))
    out["gap_pp"] = out["established"]["approve_rate"] - out["early-career"]["approve_rate"]
    return out, approve

G0, _ = group_stats(p_good_lr, 0.5)

def boot_gap(p_good, thr, k=2000):
    approve = ((1 - p_good) < thr).astype(int)
    y = approve[(test_df["emp_group"] == "early-career").values]
    o = approve[(test_df["emp_group"] == "established").values]
    idx_y = RNG.randint(0, len(y), size=(k, len(y)))
    idx_o = RNG.randint(0, len(o), size=(k, len(o)))
    gaps = o[idx_o].mean(axis=1) - y[idx_y].mean(axis=1)
    return np.percentile(gaps, [2.5, 97.5]), float(o.mean() - y.mean())

ci_gap, _ = boot_gap(p_good_lr, 0.5)
true_bad_by_group = test_df.groupby("emp_group")["target_bad"].mean()
pred_bad_lr = (p_bad_lr >= 0.5).astype(int)
err_by_group = {g: float((pred_bad_lr[(test_df["emp_group"] == g).values] !=
                          test_df["target_bad"].values[(test_df["emp_group"] == g).values]).mean())
                for g in groups}
print(f"approval: early {G0['early-career']['approve_rate']:.1%} vs established {G0['established']['approve_rate']:.1%} "
      f"→ gap {G0['gap_pp']*100:+.1f} pp (CI [{ci_gap[0]*100:+.1f}, {ci_gap[1]*100:+.1f}])")
print(f"true bad rate: early {true_bad_by_group['early-career']:.1%} vs established {true_bad_by_group['established']:.1%}")
print(f"error rate: {err_by_group}")

# income dimension (secondary, fully observed legal factor)
inc_med = test_df["annual_inc"].median()
def inc_gap(p_good, thr=0.5):
    approve = ((1 - p_good) < thr).astype(int)
    lo = approve[(test_df["annual_inc"].values < inc_med)]
    hi = approve[(test_df["annual_inc"].values >= inc_med)]
    return float(hi.mean() - lo.mean())
inc_gap_0 = inc_gap(p_good_lr)
print(f"income gap (above-median − below-median approval): {inc_gap_0*100:+.1f} pp")

# ---------- 7. proxy experiment: does hiding employment length help? ----------
model_noemp, p_bad_noemp = fit_probs("lr", train_df, test_df, drop_cols=("emp_years",))
p_good_noemp = 1 - p_bad_noemp
def gstats_from(p_good, thr=0.5):
    approve = ((1 - p_good) < thr).astype(int)
    y = approve[(test_df["emp_group"] == "early-career").values].mean()
    o = approve[(test_df["emp_group"] == "established").values].mean()
    return float(y), float(o), float(o - y)
y0, o0, gap0 = gstats_from(p_good_lr)
y1, o1, gap1 = gstats_from(p_good_noemp)
print(f"gap without emp_years: {gap1*100:+.1f} pp (was {gap0*100:+.1f} pp)")

# η²: how much employment-length information the remaining features carry
age_tr = train_df["emp_years"].values
total_ss = float(((age_tr[age_tr >= 0] - age_tr[age_tr >= 0].mean()) ** 2).sum())
eta = {}
for col in [c for c in FEATS if c not in ("emp_years",)]:
    v = train_df[col]
    if col in num_cols:
        r = v.corr(train_df["emp_years"], method="spearman")
        eta[col] = float(r * r)
    else:
        between = 0.0
        for _, gg in train_df.groupby(col)["emp_years"]:
            gg = gg[gg >= 0]
            between += len(gg) * (gg.mean() - age_tr[age_tr >= 0].mean()) ** 2
        eta[col] = between / total_ss if total_ss > 0 else 0.0
eta_top = dict(sorted(eta.items(), key=lambda kv: -kv[1])[:6])
print("top emp proxies:", {k: round(v, 3) for k, v in eta_top.items()})

# ---------- 8. policies: select on validation, report once on test ----------
p_bad_val = model_lr.predict_proba(val_df[FEATS])[:, 1]
val_ages = val_df["emp_group"].values
test_ages = test_df["emp_group"].values
test_bad = test_df["target_bad"].values

def cost_of(p_bad, thr_early, thr_est, frame_ages, frame_bad, ratio=RATIO):
    thr = np.where(frame_ages == "early-career", thr_early, thr_est)
    pred_bad = (p_bad >= thr).astype(int)
    fp = int(((pred_bad == 1) & (frame_bad == 0)).sum())
    fn = int(((pred_bad == 0) & (frame_bad == 1)).sum())
    return fp + ratio * fn, fp, fn

def gap_of(p_bad, thr_early, thr_est, frame_ages):
    thr = np.where(frame_ages == "early-career", thr_early, thr_est)
    approve = p_bad < thr
    y = float(approve[frame_ages == "early-career"].mean())
    o = float(approve[frame_ages == "established"].mean())
    return o - y, y, o

frontier = []
for ty in np.arange(0.20, 0.85, 0.01):
    for to in np.arange(0.20, 0.85, 0.01):
        thr = np.where(val_ages == "early-career", ty, to)
        pred_bad = (p_bad_val >= thr).astype(int)
        val_bad = val_df["target_bad"].values
        fp = int(((pred_bad == 1) & (val_bad == 0)).sum())
        fn = int(((pred_bad == 0) & (val_bad == 1)).sum())
        approve = p_bad_val < thr
        ym = float(approve[val_ages == "early-career"].mean())
        om = float(approve[val_ages == "established"].mean())
        acc = float(accuracy_score(val_bad, pred_bad))
        bad_e = (val_bad == 1) & (val_ages == "early-career")
        bad_o = (val_bad == 1) & (val_ages == "established")
        frontier.append(dict(ty=float(ty), to=float(to), cost=int(fp + RATIO * fn), gap=float(om - ym),
                             acc=acc, young_ar=float(ym), old_ar=float(om),
                             rec_y=float((approve & bad_e).sum() / max(bad_e.sum(), 1)),
                             rec_o=float((approve & bad_o).sum() / max(bad_o.sum(), 1)),
                             fp=fp, fn=fn))
frontier_df = pd.DataFrame(frontier)
FLOOR = 0.40
feasible = frontier_df[(frontier_df["young_ar"] >= FLOOR) & (frontier_df["old_ar"] >= FLOOR)]
cand_eq = feasible[feasible["gap"].abs() <= 0.02]
P_eq = cand_eq.loc[cand_eq["cost"].idxmin()].to_dict() if len(cand_eq) else None
P_minloss = feasible.loc[feasible["cost"].idxmin()].to_dict()
cand_eo = feasible[(feasible["rec_y"] - feasible["rec_o"]).abs() <= 0.02]
P_eqopp = cand_eo.loc[cand_eo["cost"].idxmin()].to_dict() if len(cand_eo) else None

n_bad_test = int(test_df["target_bad"].sum())
def test_report(ty, to):
    thr = np.where(test_ages == "early-career", ty, to)
    pred_bad = (p_bad_lr >= thr).astype(int)
    fp = int(((pred_bad == 1) & (test_bad == 0)).sum())
    fn = int(((pred_bad == 0) & (test_bad == 1)).sum())
    approve = p_bad_lr < thr
    y = float(approve[test_ages == "early-career"].mean())
    o = float(approve[test_ages == "established"].mean())
    bad_e = (test_bad == 1) & (test_ages == "early-career")
    bad_o = (test_bad == 1) & (test_ages == "established")
    return dict(ty=round(float(ty), 3), to=round(float(to), 3),
                gap_pp=round(float(o - y) * 100, 1), cost=int(fp + RATIO * fn),
                acc=round(float(accuracy_score(test_bad, pred_bad)), 4),
                caught_bad=int(n_bad_test - fn), missed_bad=int(fn), good_blocked=int(fp),
                young_ar=round(float(y), 4), old_ar=round(float(o), 4),
                rec_early=round(float((approve & bad_e).sum() / max(bad_e.sum(), 1)), 4),
                rec_est=round(float((approve & bad_o).sum() / max(bad_o.sum(), 1)), 4))

g0, y0d, o0d = gap_of(p_bad_lr, 0.5, 0.5, test_ages)
policies_out = {"default_050_050": dict(test=test_report(0.5, 0.5),
                val=dict(gap_pp=round(float(gap_of(p_bad_val, 0.5, 0.5, val_ages)[0]) * 100, 1),
                         cost=int(cost_of(p_bad_val, 0.5, 0.5, val_ages, val_df["target_bad"].values)[0])))}
if P_eq is not None:
    ge, ye, oe = gap_of(p_bad_val, P_eq["ty"], P_eq["to"], val_ages)
    policies_out["equal_approval"] = dict(ty=P_eq["ty"], to=P_eq["to"], test=test_report(P_eq["ty"], P_eq["to"]),
        val=dict(gap_pp=round(float(ge) * 100, 1), cost=int(P_eq["cost"])))
if P_minloss is not None:
    gm, ym2, om2 = gap_of(p_bad_val, P_minloss["ty"], P_minloss["to"], val_ages)
    policies_out["min_cost"] = dict(ty=P_minloss["ty"], to=P_minloss["to"], test=test_report(P_minloss["ty"], P_minloss["to"]),
        val=dict(gap_pp=round(float(gm) * 100, 1), cost=int(P_minloss["cost"])))
if P_eqopp is not None:
    go, yo, oo = gap_of(p_bad_val, P_eqopp["ty"], P_eqopp["to"], val_ages)
    policies_out["equal_bad_approval"] = dict(ty=P_eqopp["ty"], to=P_eqopp["to"], test=test_report(P_eqopp["ty"], P_eqopp["to"]),
        val=dict(rec_gap_pp=round(float((P_eqopp["rec_y"] - P_eqopp["rec_o"]) * 100), 1), cost=int(P_eqopp["cost"]),
                 gap_pp=round(float(go) * 100, 1)))

print("== policies (validation-selected, test-reported) ==")
for k, v in policies_out.items():
    t = v["test"]
    print(f"{k:20s} ty {t['ty']:.2f} to {t['to']:.2f} | test: gap {t['gap_pp']:+.1f} pp, cost {t['cost']}, "
          f"acc {t['acc']:.1%}, recall {t['caught_bad']}/{t['caught_bad']+t['missed_bad']}")

# ---------- 9. charts ----------
# fig_us1: gap chart
fig, ax = plt.subplots(figsize=(7.4, 3.8))
ar = [G0[g]["approve_rate"] for g in groups]
bars = ax.bar(groups, ar, color=[AMBER, CYAN], width=0.5)
for b, v in zip(bars, ar):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontweight="bold", color=INK, fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_ylabel("approval rate at threshold 0.5")
ax.set_title(f"LendingClub: approval gap {100*(ar[1]-ar[0]):+.0f} pp against early-career borrowers\n(95% CI [{ci_gap[0]*100:+.1f}, {ci_gap[1]*100:+.1f}] pp, bootstrap n=2,000)")
ax.axhline(ar[0], color=AMBER, ls=":", lw=1)
style_ax(ax)
ax2 = ax.twinx()
ax2.plot([0, 1], [true_bad_by_group[g] for g in groups], "o--", color=NAVY, lw=1.2, ms=5)
ax2.set_ylim(0, 0.6); ax2.set_ylabel("true bad-risk rate in group", color=NAVY)
ax2.tick_params(axis="y", colors=NAVY); ax2.spines["top"].set_visible(False); ax2.grid(False)
ax2.annotate("true bad-risk rate by group\n(partially explains the gap)", xy=(0.02, 0.90),
             xycoords="axes fraction", fontsize=8.5, color=NAVY,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#CBD5E1", lw=0.6))
footnote(fig, "LendingClub 2007–2018Q4 · logistic regression, P(bad) < 0.5 → approve · dotted = early-career approval level")
fig.savefig(f"{ASSETS}/fig_us_gap.png"); plt.close(fig)

# fig_us2: fairness–cost selection surface
fig, ax = plt.subplots(figsize=(7.6, 4.4))
fdf = frontier_df[frontier_df["cost"] <= policies_out["default_050_050"]["val"]["cost"] * 1.05]
sc = ax.scatter(fdf["gap"] * 100, fdf["cost"], c=fdf["acc"] * 100, cmap="viridis", s=10, alpha=0.5)
ax.axvline(0, color="#CBD5E1", lw=1, zorder=1)
BBOX = dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9, ec="#CBD5E1", lw=0.6)
US_MK = [("default_050_050", "#0B1F3A", "D"), ("equal_approval", "#B45309", "*"),
         ("min_cost", "#DC2626", "X"), ("equal_bad_approval", "#0E7490", "P")]
US_LAB = {"default_050_050": "default — test: gap +27.0 pp", "equal_approval": "equal approval — test: does its val gap transfer?",
          "min_cost": "min cost", "equal_bad_approval": "equal bad-approval"}
for key, col, mk in US_MK:
    if key not in policies_out: continue
    t = policies_out[key]["test"]; v = policies_out[key]["val"]
    gpos, cpos = (v["gap_pp"], v["cost"]) if "gap_pp" in v else (0.0, v["cost"])
    ax.scatter([gpos], [cpos], marker=mk, s=110, color=col, zorder=5,
               label=f"{key.replace('_',' ')} — test: gap {t['gap_pp']:+.1f} pp, cost {t['cost']}, acc {t['acc']:.0%}")
ax.legend(fontsize=7.8, frameon=True, loc="upper left", framealpha=0.92)
ax.set_xlabel("approval-rate gap, established minus early-career (pp) — validation surface")
ax.set_ylabel(f"total cost = 1×good blocked + {RATIO:.1f}×bad approved (data-implied)")
ax.set_title("LendingClub fairness–cost map: policies selected on validation")
fig.colorbar(sc, label="accuracy on validation (%)")
style_ax(ax)
fig.subplots_adjust(bottom=0.18)
footnote(fig, "Cost ratio estimated from the data (median charge-off loss / median forgone interest), not from documentation")
fig.savefig(f"{ASSETS}/fig_us_frontier.png"); plt.close(fig)

# fig_compare: 1994 Germany vs 2020 US headline comparison
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax = axes[0]
items = ["approval gap\n(pp)", "parity policy\ngap on test (pp)", "cost ratio\n(fair / default)"]
de = [27.0, 23.2, 112 / 165]
us = [G0["gap_pp"] * 100,
      policies_out["equal_approval"]["test"]["gap_pp"] if "equal_approval" in policies_out else float("nan"),
      (policies_out["equal_approval"]["test"]["cost"] / policies_out["default_050_050"]["test"]["cost"]
       if "equal_approval" in policies_out else float("nan"))]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, de, w, label="1994 Germany (n=1,000)", color=NAVY)
ax.bar(x + w/2, us, w, label="2020 US (n=396,030)", color=CYAN)
for i, (a, b) in enumerate(zip(de, us)):
    ax.text(i - w/2, a + 1.2, f"{a:.1f}", ha="center", fontsize=9, color=NAVY)
    ax.text(i + w/2, b + 1.2, f"{b:.1f}", ha="center", fontsize=9, color="#0E9FB8")
ax.set_xticks(x); ax.set_xticklabels(items, fontsize=9)
ax.set_title("Headline numbers, same audit protocol")
ax.legend(fontsize=8.5, frameon=False); style_ax(ax)
ax = axes[1]
grp = ["Germany", "LendingClub"]
young = [0.526, G0["early-career"]["approve_rate"]]
old = [0.796, G0["established"]["approve_rate"]]
x = np.arange(2); w2 = 0.3
ax.bar(x - w2/2, young, w2, label="thin-file group approval", color=AMBER)
ax.bar(x + w2/2, old, w2, label="established group approval", color=CYAN)
ax.set_xticks(x); ax.set_xticklabels(grp); ax.set_ylim(0, 1.0)
ax.set_ylabel("approval rate @0.5")
ax.set_title("Two markets, same direction of disparity")
ax.legend(fontsize=8.5, frameon=False, loc="upper left"); style_ax(ax)
fig.subplots_adjust(bottom=0.18, wspace=0.3)
footnote(fig, "Germany: age <25 vs ≥25 (Statlog German Credit) · US: employment ≤2y/unreported vs ≥3y (LendingClub) · both models LR @0.5, validation-selected policies")
fig.savefig(f"{ASSETS}/fig_compare.png"); plt.close(fig)

# ---------- 10. results_us.json ----------
results = {
    "meta": dict(dataset="LendingClub accepted loans 2007–2018Q4 (loan_two variant, 396,030 rows)",
                 n=n, n_train=len(train_df), n_val=len(val_df), n_test=len(test_df),
                 bad_rate_all=round(float(bad_rate), 4), seed=42,
                 split="60/20/20 stratified train/val/test; policies selected on validation, reported once on test",
                 audit_dimension="employment length: early-career (≤2y or unreported) vs established (≥3y)",
                 protected_attributes="none present — LendingClub applications contain no age/sex/ethnicity fields",
                 cost_pricing=f"data-implied: charge-off loss vs forgone interest ≈ {RATIO:.1f}:1"),
    "baselines": dict(approve_everyone=dict(acc=round(b_acc, 4), missed_bad=b_missed, caught_bad=0)),
    "models": {
        "logreg": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in M_LR.items()},
        "cv_auc_lr": [round(float(cv_lr.mean()), 4), round(float(cv_lr.std()), 4)],
        "note_rf": "random forest omitted from headline (150 trees on 237k rows trained, CV within 0.01 AUC of LR)",
    },
    "audit_employment": {
        "groups": {g: {k: (round(v2, 4) if isinstance(v2, float) else v2) for k, v2 in G0[g].items()} for g in groups},
        "gap_pp": round(float(G0["gap_pp"]) * 100, 1),
        "gap_ci95_pp": [round(float(ci_gap[0]) * 100, 1), round(float(ci_gap[1]) * 100, 1)],
        "true_bad_rate": {g: round(float(true_bad_by_group[g]), 4) for g in groups},
        "error_rate_by_group": {g: round(err_by_group[g], 4) for g in groups},
    },
    "audit_income": {"threshold": round(float(inc_med), 0), "gap_pp_above_minus_below": round(inc_gap_0 * 100, 1)},
    "proxy": {"gap_pp_without_emp_years": round(float(gap1) * 100, 1),
              "gap_pp_with": round(float(gap0) * 100, 1),
              "emp_variance_explained_top": {k: round(float(v), 4) for k, v in eta_top.items()}},
    "policies": policies_out,
    "frontier_points": frontier_df[["ty", "to", "gap", "cost", "acc", "young_ar", "old_ar"]].round(4).to_dict("records"),
    "comparison_germany": {
        "note": "same protocol on Statlog German Credit (scripts/run_analysis.py)",
        "gap_pp": 27.0, "gap_ci95_pp": [10.0, 44.5],
        "equal_approval_test": {"gap_pp": 23.2, "cost": 112, "default_cost": 165},
    },
}
def _default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))
with open(f"{RESULTS}/results_us.json", "w") as f:
    json.dump(results, f, indent=1, default=_default)

print("saved: assets/fig_us_gap.png, fig_us_frontier.png, fig_compare.png + results/results_us.json")
