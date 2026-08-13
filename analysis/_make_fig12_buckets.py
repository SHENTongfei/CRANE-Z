# -*- coding: utf-8 -*-
"""Fig12: age-bucket error structure (P5b).

H54 layout: 2x2 + full-width E.
A: MAE by age bucket (bar).
B: bias by age bucket (signed).
C: AA by chronological age scatter (internal, RLL red), deceleration deepens.
D: negative-AA fraction by bucket.
E (full-width): predicted vs chronological age with bucket shading.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7, "figure.dpi": 300,
})
C_RLL = "#C0392B"
C_RYC = "#4A7BA6"
C_ACC = "#2E7D32"


def panel(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", fontfamily="Arial")


def main():
    df = pd.read_csv(os.path.join(C.OUT_DIR, "aa_internal_validation.csv"))
    df = df[df.group.isin(["RLL", "RYC"])].copy()
    df["err"] = df["pred"] - df["chrono"]
    df["abs_err"] = df["err"].abs()

    buckets = [(0, 50, "<50"), (50, 70, "50-70"), (70, 200, ">=70")]
    bnames = [b[2] for b in buckets]
    maes = []
    biases = []
    neg_frac = []
    ns = []
    for lo, hi, _ in buckets:
        b = df[(df.chrono >= lo) & (df.chrono < hi)]
        maes.append(b.abs_err.mean())
        biases.append(b.err.mean())
        ns.append(len(b))
        rll_b = b[b.group == "RLL"]
        neg_frac.append((rll_b.AA < 0).mean() if len(rll_b) else np.nan)

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.35,
                          left=0.10, right=0.97, top=0.94, bottom=0.30)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(range(3), maes, color=[C_ACC, C_ACC, C_ACC])
    ax.set_xticks(range(3)); ax.set_xticklabels(bnames, fontsize=6.5)
    ax.set_ylabel("MAE (years)")
    ax.set_title("Prediction error by age bucket")
    for i, (m, n) in enumerate(zip(maes, ns)):
        ax.text(i, m + 0.15, f"{m:.1f} (n={n})", ha="center", fontsize=6)
    ax.set_ylim(0, max(maes) * 1.25)
    panel(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(3), biases, color=["#D32F2F" if b < 0 else "#1976D2" for b in biases])
    ax.axhline(0, color="#999", lw=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(bnames, fontsize=6.5)
    ax.set_ylabel("mean signed error (years)")
    ax.set_title("Prediction bias by age bucket")
    for i, b in enumerate(biases):
        ax.text(i, b + 0.25 * (1 if b >= 0 else -1), f"{b:+.1f}", ha="center", fontsize=6.5)
    panel(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    rll = df[df.group == "RLL"]
    ax.scatter(rll.chrono, rll.AA, s=9, c=C_RLL, alpha=0.75, label="RLL")
    ryc = df[df.group == "RYC"]
    ax.scatter(ryc.chrono, ryc.AA, s=9, c=C_RYC, alpha=0.6, label="RYC")
    ax.axhline(0, color="#999", ls="--", lw=0.7)
    rho, p = stats.spearmanr(rll.chrono, rll.AA)
    ax.set_xlabel("chronological age (yr)"); ax.set_ylabel("AA (years)")
    ax.set_title(f"AA deepens with age within RLL (Spearman {rho:+.2f}, p={p:.2e})")
    ax.legend(frameon=False, fontsize=6)
    panel(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(3), neg_frac, color=C_RLL)
    ax.axhline(0.5, color="#999", ls="--", lw=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(bnames, fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("RLL negative-AA fraction")
    ax.set_title("Individual-level deceleration by bucket")
    for i, f in enumerate(neg_frac):
        if not np.isnan(f):
            ax.text(i, f + 0.02, f"{f:.2f}", ha="center", fontsize=6.5)
    panel(ax, "D")

    ax = fig.add_subplot(gs[2, :])
    ax.scatter(df.chrono, df.pred, s=9, c=[C_RLL if g == "RLL" else C_RYC for g in df.group],
               alpha=0.75)
    ax.plot([0, 115], [0, 115], ls="--", color="#999", lw=0.8)
    ax.set_xlabel("chronological age (yr)"); ax.set_ylabel("predicted biological age (yr)")
    ax.set_title("Predicted versus chronological age across the full spectrum")
    rho2, p2 = stats.spearmanr(df.chrono, df.pred)
    ax.text(0.02, 0.95, f"Spearman {rho2:.3f}, p={p2:.1e}", transform=ax.transAxes, fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_RLL, label="RLL"), Patch(color=C_RYC, label="RYC")],
              frameon=False, fontsize=6.5, loc="lower right")
    panel(ax, "E")

    os.makedirs("figures_ms", exist_ok=True)
    fig.savefig(os.path.join("figures_ms", "Fig12_age_buckets.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join("figures_ms", "Fig12_age_buckets.pdf"), bbox_inches="tight")
    print("saved Fig12_age_buckets.png/pdf")


if __name__ == "__main__":
    main()
