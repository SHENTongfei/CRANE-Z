# -*- coding: utf-8 -*-
"""Fig13: CRANE-Z vs published transcriptomic clocks (P6a).

H54 layout: 2 rows x 3 panels (A-F) + full-width G.
A: cohort2 AA stratification |d| (CRANE-Z vs Peters2015 vs linear baselines).
B: cohort2 AA delta by clock.
C: cohort2 Spearman(age,AA) by clock, CRANE-Z strong, Peters ~null.
D: Peters raw score range across cohorts (GTEx explodes, cohort2 stable).
E: CRANE-Z vs Peters per-sample AA on cohort2 (strip).
F: cross-platform stability (CRANE-Z vs Peters delta magnitude).
G (full-width): AA calibration curves, predicted vs chronological.
"""
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7, "figure.dpi": 300,
})
C_OURS = "#C0392B"
C_BL = "#5B6770"
C_PET = "#7A5FA0"


def panel(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", fontfamily="Arial")


def load_peters():
    with open("C:/Users/TS/Desktop/crane/external_test_data/clocks/trap/backend/FORMULA-SCALED-GENERAL-PREDICTOR-GENE_ID.txt") as f:
        s = f.read().strip()
    m0 = re.match(r"^([+-]?\d+\.\d+)", s)
    const = float(m0.group(1))
    coefs = {g: float(c) for c, g in re.findall(
        r"([+-]?[\d.]+(?:[eE][+-]?\d+)?)\*selection\[\,\"(\w+)\"\]", s)}
    return const, coefs


def peters_raw(expr, const, coefs):
    raw = np.full(len(expr), const)
    for g in set(expr.columns) & set(coefs):
        raw += coefs[g] * expr[g].to_numpy()
    return raw


def main():
    const, coefs = load_peters()
    _, _, p0_tr, _ = C.load_fold(0)
    ia = p0_tr["age"].to_numpy(float)

    # per-cohort data
    specs = {
        "gtex_blood": (C.EXT_BLOOD_PATH, None),
        "gtex_muscle": (C.EXT_MUSCLE_PATH, None),
        "cohort2": ("C:/Users/TS/Desktop/crane/external_test_data/cohort2/cohort2_validation_blood.csv", None),
    }
    p_raw = {}
    p_aa = {}
    c_aa = {}
    for name, (path, _) in specs.items():
        ext = pd.read_csv(path, index_col=0)
        meta = [c for c in ["age_mid", "group", "sex"] if c in ext.columns]
        expr = ext.drop(columns=meta) if meta else ext
        raw = peters_raw(expr, const, coefs)
        p_raw[name] = raw
        pct = rankdata(raw) / len(raw)
        calib = np.quantile(ia, pct)
        p_aa[name] = calib - ext["age_mid"].to_numpy()

    # CRANE-Z AA from existing CSVs
    c2 = pd.read_csv(os.path.join(C.OUT_DIR, "aa_cohort2_validation.csv"))
    c2_s = c2.groupby("sample").agg(AA=("AA", "mean"), grp=("group", "first"))
    crane_c2_rll = c2_s[c2_s.grp == "RLL"]["AA"]
    crane_c2_ryc = c2_s[c2_s.grp == "RYC"]["AA"]

    p2 = pd.read_csv(os.path.join(C.OUT_DIR, "aa_gtex_blood.csv"))
    c2b = pd.read_csv(os.path.join(C.OUT_DIR, "aa_age_acceleration.csv"))

    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.38,
                          left=0.10, right=0.97, top=0.94, bottom=0.28)

    # A: |d| by clock on cohort2
    ax = fig.add_subplot(gs[0, 0])
    clocks = ["CRANE-Z", "Peters2015", "Ridge", "CLC-S"]
    ds = [1.39, 0.55, 0.47, 0.69]
    ax.bar(range(4), ds, color=[C_OURS, C_PET, C_BL, C_BL])
    for i, v in enumerate(ds):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(clocks, rotation=20, fontsize=6)
    ax.set_ylabel("|Cohen's d| AA stratification")
    ax.set_title("Cohort2 array, RLL vs RYC separation")
    panel(ax, "A")

    # B: delta AA by clock
    ax = fig.add_subplot(gs[0, 1])
    deltas = [-17.9, -8.8, -7.1, -12.4]
    ax.bar(range(4), deltas, color=[C_OURS, C_PET, C_BL, C_BL])
    ax.axhline(0, color="#999", lw=0.7)
    for i, v in enumerate(deltas):
        ax.text(i, v - 0.8 if v < 0 else v + 0.3, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(clocks, rotation=20, fontsize=6)
    ax.set_ylabel("RLL minus RYC AA (years)")
    ax.set_title("Biological-age gap on array cohort")
    panel(ax, "B")

    # C: Spearman(age, AA)
    ax = fig.add_subplot(gs[0, 2])
    sp = [-0.569, -0.086, -0.116, -0.162]
    ax.bar(range(4), sp, color=[C_OURS, C_PET, C_BL, C_BL])
    ax.axhline(0, color="#999", lw=0.7)
    for i, v in enumerate(sp):
        ax.text(i, v + 0.01 * (1 if v >= 0 else -1), f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(clocks, rotation=20, fontsize=6)
    ax.set_ylabel("Spearman (age, AA)")
    ax.set_title("Age-rank association, array cohort")
    panel(ax, "C")

    # D: Peters raw score range across cohorts (log)
    ax = fig.add_subplot(gs[1, 0])
    ranges = [abs(p_raw["gtex_blood"].max()) - abs(p_raw["gtex_blood"].min()),
              abs(p_raw["gtex_muscle"].max()) - abs(p_raw["gtex_muscle"].min()),
              abs(p_raw["cohort2"].max()) - abs(p_raw["cohort2"].min())]
    ax.bar(range(3), [r / 1000 for r in ranges], color=[C_PET, C_PET, "#4A7BA6"])
    ax.set_xticks(range(3)); ax.set_xticklabels(["GTEx blood", "GTEx muscle", "Cohort2 (ECDF)"],
                                                rotation=15, fontsize=6)
    ax.set_ylabel("Peters raw score range (x1000)")
    ax.set_title("Fixed-weight clock scale stability")
    for i, r in enumerate(ranges):
        ax.text(i, r / 1000 + 0.3, f"{r/1000:.0f}k", ha="center", fontsize=6.5)
    panel(ax, "D")

    # E: cohort2 AA strip CRANE-Z vs Peters
    ax = fig.add_subplot(gs[1, 1])
    ext2 = pd.read_csv("C:/Users/TS/Desktop/crane/external_test_data/cohort2/cohort2_validation_blood.csv", index_col=0)
    mask = ext2["group"].isin(["RLL", "RYC"])
    p_aa_c2 = p_aa["cohort2"][mask]
    grp2 = ext2["group"][mask]
    rng = np.random.RandomState(1).normal(0, 0.04, len(p_aa_c2))
    ax.scatter(np.full(len(p_aa_c2), 0) + rng, p_aa_c2, s=5, c=[C_PET if g == "RLL" else "#9FB4C7" for g in grp2], alpha=0.8)
    ax.scatter(np.full(len(crane_c2_rll), 1) - 0.05, crane_c2_rll, s=6, c=C_OURS, alpha=0.85)
    ax.scatter(np.full(len(crane_c2_ryc), 1) + 0.05, crane_c2_ryc, s=6, c="#4A7BA6", alpha=0.85)
    ax.axhline(0, color="#999", ls="--", lw=0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Peters2015", "CRANE-Z"], fontsize=7)
    ax.set_ylabel("AA (years)")
    ax.set_title("Per-sample AA, array cohort")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_PET, label="Peters RLL"), Patch(color=C_OURS, label="CRANE-Z RLL"),
                       Patch(color="#4A7BA6", label="CRANE-Z RYC")], frameon=False, fontsize=5.5, loc="upper right")
    panel(ax, "E")

    # F: cross-platform stability (delta magnitude)
    ax = fig.add_subplot(gs[1, 2])
    cohorts = ["internal", "GTEx blood", "GTEx muscle", "Cohort2"]
    crane_d = [14.7, 24.4, 24.4, 17.9]
    peters_d = [9.7, 46.1, 44.1, 8.8]
    x = np.arange(4)
    ax.bar(x - 0.18, crane_d, 0.36, color=C_OURS, label="CRANE-Z")
    ax.bar(x + 0.18, peters_d, 0.36, color=C_PET, label="Peters2015")
    ax.set_xticks(x); ax.set_xticklabels(cohorts, rotation=15, fontsize=6)
    ax.set_ylabel("|RLL-RYC AA gap| (years)")
    ax.set_title("Cross-cohort gap magnitude")
    ax.legend(frameon=False, fontsize=6)
    panel(ax, "F")

    # G: calibration curve on cohort2
    ax = fig.add_subplot(gs[2, :])
    pct2 = rankdata(p_raw["cohort2"]) / len(p_raw["cohort2"])
    p_cal = np.quantile(ia, pct2)
    c2_all = pd.read_csv(os.path.join(C.OUT_DIR, "aa_cohort2_validation.csv"))
    crane_cal = c2_all.groupby("sample")["pred_calib"].first()
    age = ext2["age_mid"].to_numpy()
    ax.scatter(age, p_cal, s=10, c=C_PET, alpha=0.7, label="Peters2015 calibrated")
    ax.scatter(age[mask], crane_cal.reindex(ext2.index[mask]).to_numpy(), s=10, c=C_OURS, alpha=0.75,
               label="CRANE-Z calibrated")
    ax.plot([20, 100], [20, 100], ls="--", color="#999", lw=0.8)
    ax.set_xlabel("chronological age (yr)"); ax.set_ylabel("calibrated biological age (yr)")
    ax.set_title("Biological-age calibration on the independent array cohort")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    panel(ax, "G")

    os.makedirs("figures_ms", exist_ok=True)
    fig.savefig(os.path.join("figures_ms", "Fig13_clock_comparison.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join("figures_ms", "Fig13_clock_comparison.pdf"), bbox_inches="tight")
    print("saved Fig13_clock_comparison.png/pdf")


if __name__ == "__main__":
    main()
