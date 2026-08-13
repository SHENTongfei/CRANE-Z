# -*- coding: utf-8 -*-
"""Fig10: cross-platform AA stratification robustness, CRANE-Z vs all baselines.

H54 layout: 2 rows x 3 panels (A-F) + full-width panel G at bottom.
A: |Cohen's d| by cohort (internal vs cohort2 chip), CRANE-Z highlighted.
B: cross-platform decay % (internal -> chip) by model, CRANE-Z near zero.
C: delta AA (years) by cohort, CRANE-Z biologically plausible.
D: Spearman(age, AA) on cohort2 by model, CRANE-Z strongest.
E: RLL AA<0 fraction by cohort (proportion biologically younger).
F: rank-calibrated predicted vs chronological age on cohort2, CRANE-Z vs Ridge.
G (full-width): Cohen's d transfer curve, internal vs cohort2 for all models.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CSV = os.path.join("output", "aa_stratification_vs_baselines.csv")
OUT_PNG = os.path.join("figures_ms", "Fig10_aa_robustness.png")
OUT_PDF = os.path.join("figures_ms", "Fig10_aa_robustness.pdf")

df = pd.read_csv(CSV)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7,
    "figure.dpi": 300,
})

C_OURS = "#C0392B"   # red accent for CRANE-Z
C_BL = "#5B6770"     # baseline grey
C_LN = "#1F4E79"     # line blue

MODELS = ["CRANE-Z", "Ridge", "Lasso", "CLC-S", "KNN", "DT"]
colors = {m: (C_OURS if m == "CRANE-Z" else C_BL) for m in MODELS}

int_d = {m: df[(df.model == m) & (df.cohort == "internal")].iloc[0]["cohen_d"] for m in MODELS}
c2_d = {m: df[(df.model == m) & (df.cohort == "cohort2")].iloc[0]["cohen_d"] for m in MODELS}
int_delta = {m: df[(df.model == m) & (df.cohort == "internal")].iloc[0]["delta_aa"] for m in MODELS}
c2_delta = {m: df[(df.model == m) & (df.cohort == "cohort2")].iloc[0]["delta_aa"] for m in MODELS}
c2_spear = {m: df[(df.model == m) & (df.cohort == "cohort2")].iloc[0]["spearman_age_aa"] for m in MODELS}
c2_neg = {m: df[(df.model == m) & (df.cohort == "cohort2")].iloc[0]["rll_aa_neg_frac"] for m in MODELS}
int_neg = {m: df[(df.model == m) & (df.cohort == "internal")].iloc[0]["rll_aa_neg_frac"] for m in MODELS}
decay = {m: (1 - abs(c2_d[m]) / abs(int_d[m])) * 100 for m in MODELS}

fig = plt.figure(figsize=(7.2, 7.6))
gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.35,
                      left=0.09, right=0.97, top=0.94, bottom=0.30)


def panel(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", fontfamily="Arial")


# A: |d| by cohort
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(MODELS))
ax.bar(x - 0.18, [abs(int_d[m]) for m in MODELS], 0.36, color=[colors[m] for m in MODELS], label="internal")
ax.bar(x + 0.18, [abs(c2_d[m]) for m in MODELS], 0.36, color=[colors[m] for m in MODELS], alpha=0.45, label="array cohort")
ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30)
ax.set_ylabel("|Cohen's d|")
ax.set_title("AA stratification effect size")
ax.legend(frameon=False, fontsize=5.5)
panel(ax, "A")

# B: decay %
ax = fig.add_subplot(gs[0, 1])
for m in MODELS:
    ax.bar(m, max(decay[m], 0), color=colors[m])
ax.axhline(decay["CRANE-Z"], color=C_LN, ls="--", lw=0.8)
ax.set_xticklabels(MODELS, rotation=30)
ax.set_ylabel("effect-size decay (%)")
ax.set_title("Cross-platform decay, internal to array")
panel(ax, "B")

# C: delta AA years
ax = fig.add_subplot(gs[0, 2])
x = np.arange(len(MODELS))
ax.bar(x - 0.18, [int_delta[m] for m in MODELS], 0.36, color=[colors[m] for m in MODELS], label="internal")
ax.bar(x + 0.18, [c2_delta[m] for m in MODELS], 0.36, color=[colors[m] for m in MODELS], alpha=0.45, label="array")
ax.axhline(-14.69, color=C_LN, ls=":", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30)
ax.set_ylabel("delta AA (years)")
ax.set_title("RLL minus RYC biological age gap")
ax.legend(frameon=False, fontsize=5.5)
panel(ax, "C")

# D: Spearman(age, AA) cohort2
ax = fig.add_subplot(gs[1, 0])
for m in MODELS:
    ax.bar(m, c2_spear[m], color=colors[m])
ax.axhline(0, color="#888", lw=0.6)
ax.set_xticklabels(MODELS, rotation=30)
ax.set_ylabel("Spearman (age, AA)")
ax.set_title("Age-rank association on array cohort")
panel(ax, "D")

# E: RLL AA<0 fraction
ax = fig.add_subplot(gs[1, 1])
x = np.arange(len(MODELS))
ax.bar(x - 0.18, [int_neg[m] for m in MODELS], 0.36, color=[colors[m] for m in MODELS], label="internal")
ax.bar(x + 0.18, [c2_neg[m] for m in MODELS], 0.36, color=[colors[m] for m in MODELS], alpha=0.45, label="array")
ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30)
ax.set_ylabel("fraction of RLL with AA<0")
ax.set_title("Long-lived biologically younger")
ax.legend(frameon=False, fontsize=5.5)
panel(ax, "E")

# F: predicted vs chronological on cohort2 (CRANE-Z vs Ridge)
ax = fig.add_subplot(gs[1, 2])
c2raw = pd.read_csv("C:/Users/TS/Desktop/crane/external_test_data/cohort2/cohort2_validation_blood.csv", index_col=0)
meta = [c for c in ["age_mid", "group", "sex"] if c in c2raw.columns]
expr = c2raw.drop(columns=meta)
chrono = c2raw["age_mid"].to_numpy(float)
grp = c2raw["group"].to_numpy(str)
from scipy.stats import rankdata
# CRANE-Z AA from output
crane_aa = pd.read_csv(os.path.join("output", "aa_cohort2_validation.csv"))
cal = crane_aa.groupby("sample")["pred_calib"].first() if "pred_calib" in crane_aa.columns else crane_aa.groupby("sample")["pred"].first()
samp = cal.index
mask = [s in samp for s in c2raw.index]
ax.scatter(chrono[mask], cal.reindex(c2raw.index[mask]).to_numpy(), c=[C_OURS if g == "RLL" else C_BL for g in grp[mask]], s=10, alpha=0.7)
ax.plot([20, 100], [20, 100], ls="--", color="#888", lw=0.7)
ax.set_xlabel("chronological age (yr)"); ax.set_ylabel("CRANE-Z calibrated bio age")
ax.set_title("CRANE-Z bio age on array cohort")
panel(ax, "F")

# G (full width): transfer curve - discrete model x-axis to avoid value overlap
ax = fig.add_subplot(gs[2, :])
xpos = np.arange(len(MODELS))
for i, m in enumerate(MODELS):
    ax.plot([xpos[i] - 0.15, xpos[i] + 0.15], [abs(int_d[m]), abs(c2_d[m])],
            marker="o", ms=4.5, lw=1.6, color=colors[m], label=m)
    ax.text(xpos[i], min(abs(int_d[m]), abs(c2_d[m])) - 0.18,
            f"{abs(int_d[m]):.2f}", ha="center", va="top", fontsize=5.5, color=colors[m])
    ax.text(xpos[i], max(abs(int_d[m]), abs(c2_d[m])) + 0.10,
            f"{abs(c2_d[m]):.2f}", ha="center", va="bottom", fontsize=5.5, color=colors[m])
ax.set_xticks(xpos); ax.set_xticklabels(MODELS, fontsize=6.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["internal", "array cohort"])
ax.set_ylim(0, 3.6)
ax.set_ylabel("|Cohen's d| transfer")
ax.set_title("Transfer of stratification effect size from RNA-seq to array platform")
ax.legend(frameon=False, ncol=6, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.12))
panel(ax, "G")

os.makedirs("figures_ms", exist_ok=True)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print("saved Fig10_aa_robustness.png/pdf")
