# -*- coding: utf-8 -*-
"""Fig11: ssGSEA immune-axis validation (P5a).

H54 layout: 2 rows x 3 panels (A-F) + full-width G.
A: composite immune score violin by group (4 cohorts), CRANE-Z red for RLL.
B: per-pathway Cohen's d by cohort heatmap (rows=pathways, cols=cohorts).
C: composite score ~ chronological age scatter (internal, Spearman annotation).
D: composite score ~ age (cohort2 array, Spearman annotation).
E: RLL in top-half fraction by cohort (bar).
F: pathway score age-correlation by cohort (bar, sign matters).
G (full-width): per-sample composite score strip across cohorts, RLL vs RYC.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from gseapy import ssgsea
from scipy import stats

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7,
    "figure.dpi": 300,
})
C_OURS = "#C0392B"
C_BL = "#5B6770"
C_RLL = "#C0392B"
C_RYC = "#4A7BA6"

GSETS = {
    "AntigenProcessing": ["KIR3DL2","KIR2DL3","KIR2DS2","KIR2DL2","KLRC3","CALR","IFI30","IFNG","KLRD1","HLA-B","PDIA3","TAP1","HLA-DRA","KIR3DL1","HLA-C","HLA-DMA","HSPA6","KIR2DS4","HLA-DQB1","HSPA1L","KIR2DS5","RFXANK","PSME1","CD8A","PSME2","HSPA1B","CTSB"],
    "NK_cytotoxicity": ["GZMB","ULBP2","SLAMF7","TUBB4B","VAMP2","KIR3DL1","ULBP1","CLEC2A","KLRF2","VAMP7","TUBB","ULBP3","PLEKHM2","GZMH","KIR2DL3","KIR2DL2","KIR3DL2","KLRD1","KLRK1","NKG7","PRF1","GNLY"],
    "NK_chemotaxis": ["CCL3","CCL5","XCL1","CCL4"],
    "Neg_IL1b": ["GHSR","CARD16","CARD17","NLRP7","CX3CR1","GSTP1","GIT1","ZC3H12A","IL1B","CASP1","PYCARD","TNF","IL6"],
}


def scores_matrix(expr):
    et = np.log2(expr.T + 1)
    ss = ssgsea(et, gene_sets=GSETS, outdir=None, threads=2, min_size=2, seed=42)
    res = ss.res2d
    out = {}
    for term in GSETS:
        sub = res[res["Term"] == term][["Name", "ES"]]
        out[term] = sub.set_index("Name")["ES"]
    m = pd.DataFrame(out)
    return m.reindex(list(et.columns))


def cohend(a, b):
    s = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) /
                (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s else 0


def panel(ax, letter):
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", fontfamily="Arial")


def main():
    X0, _, p0, _ = C.load_fold(0)
    blood = pd.read_csv(C.EXT_BLOOD_PATH, index_col=0)
    muscle = pd.read_csv(C.EXT_MUSCLE_PATH, index_col=0)
    c2 = pd.read_csv("C:/Users/TS/Desktop/crane/external_test_data/cohort2/cohort2_validation_blood.csv",
                     index_col=0)

    cohorts = [
        ("internal", X0, p0["group"], p0["age"]),
        ("gtex_blood", blood.drop(columns=["age_mid", "group", "sex"]), blood["group"], blood["age_mid"]),
        ("gtex_muscle", muscle.drop(columns=["age_mid", "group", "sex"]), muscle["group"], muscle["age_mid"]),
        ("cohort2", c2.drop(columns=["age_mid", "group", "sex"]), c2["group"], c2["age_mid"]),
    ]
    comp_all = {}
    scores_all = {}
    ages_all = {}
    grps_all = {}
    d_group = {}
    rho_age = {}
    top_rll = {}

    for name, expr, grp, age in cohorts:
        sc = scores_matrix(expr)
        scores_all[name] = sc
        z = (sc - sc.mean()) / sc.std()
        comp = z.mean(axis=1)
        comp_all[name] = comp
        ages_all[name] = age
        grps_all[name] = grp
        rll = comp[grp == "RLL"].to_numpy()
        ryc = comp[grp == "RYC"].to_numpy()
        d_group[name] = cohend(rll, ryc)
        rho_age[name] = stats.spearmanr(comp.to_numpy(), age.to_numpy()).correlation
        top_rll[name] = (comp[grp == "RLL"] > np.median(comp)).mean()
        print(f"[{name}] d={d_group[name]:+.2f} rho_age={rho_age[name]:+.3f}")

    # per-pathway d by cohort
    pws = list(GSETS.keys())
    dmat = np.zeros((len(pws), 4))
    for j, pw in enumerate(pws):
        for i, (name, _, grp, _) in enumerate(cohorts):
            sc = scores_all[name]
            s = sc[pw].astype(float)
            rll = s[grp == "RLL"].to_numpy()
            ryc = s[grp == "RYC"].to_numpy()
            dmat[j, i] = cohend(rll, ryc)

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.38,
                          left=0.10, right=0.97, top=0.94, bottom=0.28)
    cnames = ["internal", "gtex_blood", "gtex_muscle", "cohort2"]

    # A: composite violin by group
    ax = fig.add_subplot(gs[0, 0])
    positions = []
    labels = []
    for i, name in enumerate(cnames):
        comp = comp_all[name]
        grp = grps_all[name]
        for g, off, col in [("RLL", -0.18, C_RLL), ("RYC", 0.18, C_RYC)]:
            vals = comp[grp == g].to_numpy()
            parts = ax.violinplot([vals], positions=[i + off], widths=0.32,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(col); pc.set_alpha(0.65)
            parts["cmedians"].set_color("#222"); parts["cmedians"].set_linewidth(1.2)
        positions.append(i)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Internal", "GTEx blood", "GTEx muscle", "Array cohort"],
                                                rotation=15, fontsize=5.8)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_ylabel("composite immune score (z)")
    ax.set_title("Immune-effector score by longevity group")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_RLL, label="RLL"), Patch(color=C_RYC, label="RYC")],
              frameon=False, fontsize=6, loc="upper left")
    panel(ax, "A")

    # B: pathway x cohort d heatmap
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(dmat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["Int", "GTExB", "GTExM", "C2"], fontsize=6)
    ax.set_yticks(range(len(pws))); ax.set_yticklabels(pws, fontsize=6)
    for j in range(len(pws)):
        for i in range(4):
            ax.text(i, j, f"{dmat[j, i]:+.2f}", ha="center", va="center", fontsize=5.5,
                    color="#111")
    ax.set_title("Pathway x cohort Cohen's d")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    panel(ax, "B")

    # C: internal score ~ age
    ax = fig.add_subplot(gs[0, 2])
    comp = comp_all["internal"]; age = ages_all["internal"]; grp = grps_all["internal"]
    ax.scatter(age[grp == "RLL"], comp[grp == "RLL"], s=8, c=C_RLL, alpha=0.7, label="RLL")
    ax.scatter(age[grp == "RYC"], comp[grp == "RYC"], s=8, c=C_RYC, alpha=0.7, label="RYC")
    rho, p = stats.spearmanr(comp.to_numpy(), age.to_numpy())
    ax.set_xlabel("chronological age (yr)"); ax.set_ylabel("composite immune score")
    ax.set_title(f"Internal, Spearman {rho:+.2f} (p={p:.1e})")
    ax.legend(frameon=False, fontsize=6)
    panel(ax, "C")

    # D: cohort2 score ~ age
    ax = fig.add_subplot(gs[1, 0])
    comp = comp_all["cohort2"]; age = ages_all["cohort2"]; grp = grps_all["cohort2"]
    ax.scatter(age[grp == "RLL"], comp[grp == "RLL"], s=10, c=C_RLL, alpha=0.8, label="RLL")
    ax.scatter(age[grp == "RYC"], comp[grp == "RYC"], s=10, c=C_RYC, alpha=0.8, label="RYC")
    rho, p = stats.spearmanr(comp.to_numpy(), age.to_numpy())
    ax.set_xlabel("chronological age (yr)"); ax.set_ylabel("composite immune score")
    ax.set_title(f"Array cohort, Spearman {rho:+.2f} (p={p:.2e})")
    ax.legend(frameon=False, fontsize=6)
    panel(ax, "D")

    # E: RLL top-half fraction
    ax = fig.add_subplot(gs[1, 1])
    fracs = [top_rll[n] for n in cnames]
    ax.bar(range(4), fracs, color=[C_OURS if n == "internal" else C_BL for n in cnames])
    ax.axhline(0.5, color="#999", ls="--", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Int", "GTExB", "GTExM", "C2"], fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("RLL above median score")
    ax.set_title("Long-lived enrichment in high immune score")
    for i, f in enumerate(fracs):
        ax.text(i, f + 0.02, f"{f:.2f}", ha="center", fontsize=6)
    panel(ax, "E")

    # F: age correlation by cohort
    ax = fig.add_subplot(gs[1, 2])
    rhos = [rho_age[n] for n in cnames]
    ax.bar(range(4), rhos, color=[C_OURS if n == "internal" else C_BL for n in cnames])
    ax.axhline(0, color="#999", lw=0.7)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Int", "GTExB", "GTExM", "C2"], fontsize=6)
    ax.set_ylabel("Spearman (score, age)")
    ax.set_title("Immune score rises with age")
    for i, r in enumerate(rhos):
        ax.text(i, r + 0.01 * (1 if r >= 0 else -1), f"{r:+.2f}", ha="center", fontsize=6)
    panel(ax, "F")

    # G: per-sample strip across cohorts
    ax = fig.add_subplot(gs[2, :])
    for i, name in enumerate(cnames):
        comp = comp_all[name]; grp = grps_all[name]
        rll = comp[grp == "RLL"].to_numpy()
        ryc = comp[grp == "RYC"].to_numpy()
        jit = np.random.RandomState(42 + i).normal(0, 0.05, len(rll))
        ax.scatter(np.full(len(rll), i - 0.15) + jit, rll, s=6, c=C_RLL, alpha=0.75)
        jit2 = np.random.RandomState(7 + i).normal(0, 0.05, len(ryc))
        ax.scatter(np.full(len(ryc), i + 0.15) + jit2, ryc, s=6, c=C_RYC, alpha=0.75)
        ax.text(i, comp.max() + 0.25, f"d={d_group[name]:+.2f}", ha="center", fontsize=6.5,
                color="#222", fontweight="bold")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Internal", "GTEx blood", "GTEx muscle", "Array cohort"],
                                                fontsize=7)
    ax.set_ylabel("composite immune score (z)")
    ax.set_title("Sample-level immune-effector score across four cohorts, long-lived in red")
    from matplotlib.patches import Patch as P2
    ax.legend(handles=[P2(color=C_RLL, label="RLL"), P2(color=C_RYC, label="RYC")],
              frameon=False, fontsize=6.5, loc="upper left")
    panel(ax, "G")

    os.makedirs("figures_ms", exist_ok=True)
    fig.savefig(os.path.join("figures_ms", "Fig11_immune_axis.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join("figures_ms", "Fig11_immune_axis.pdf"), bbox_inches="tight")
    print("saved Fig11_immune_axis.png/pdf")


if __name__ == "__main__":
    main()
