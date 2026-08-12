# -*- coding: utf-8 -*-
"""Fig9 GSEA mechanism figure (H54 layout: 2x3 grid + bottom full-width panel)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = r"C:/Users/TS/Desktop/crane"
OUT = os.path.join(BASE, "Code_v2", "output")
FIG = os.path.join(BASE, "Code_v2", "figures_ms")
os.makedirs(FIG, exist_ok=True)

NPG = {"vermilion": "#E64B35", "blue": "#4DBBD5", "green": "#00A087",
       "navy": "#3C5488", "salmon": "#F39B7F", "greyblue": "#8491B4",
       "teal": "#91D1C2", "coffee": "#7E6148", "grey": "#B0B0B0", "black": "#000000"}
C_OURS = NPG["vermilion"]; C_BL = NPG["greyblue"]; C_G = NPG["green"]; C_NV = NPG["navy"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.8, "figure.dpi": 600, "savefig.dpi": 600,
})

def panel(ax, letter, x=0.0, y=1.08, fs=11):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fs, fontweight="bold", va="top", ha="left")

def save(fig, name, w_mm=183):
    w_in = w_mm / 25.4
    h0 = fig.get_size_inches()[1]
    h_in = h0 * (w_in / fig.get_size_inches()[0])
    fig.set_size_inches(w_in, h_in)
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=600)
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# ---------------- data (1000-perm final) ----------------
res = pd.read_csv(os.path.join(OUT, "gsea_results_all_1000.csv"))
imm = pd.read_csv(os.path.join(OUT, "gsea_immune_pathways_1000.csv"))
rho = pd.read_csv(os.path.join(OUT, "gsea_gene_age_rho.csv"))

# immune-pathway hits (FDR<0.25), top by NES
hits = imm[imm["FDR q-val"] < 0.25].sort_values("NES", ascending=False)
top = hits.head(12).iloc[::-1]  # horizontal bar: lowest NES at bottom -> top at top
short = [t.split(" (GO:")[0][:52] for t in top["Term"]]

# all strong hits by source
strong = res[res["FDR q-val"] < 0.05].copy()
src_color = {"KEGG": C_OURS, "GOBP": C_G, "Reactome": C_NV, "WikiPath": C_BL}

# gene-age trends for panel C (bottom full-width): GZMH/KIR3DL2 age-up, CR2/LRRN3 age-down
X_tr = pd.read_csv(os.path.join(BASE, "training_data", "balanced_folds", "fold_0", "train_expr.csv"), index_col=0)
p_tr = pd.read_csv(os.path.join(BASE, "training_data", "balanced_folds", "fold_0", "train_pheno.csv"), index_col=0)
age = p_tr["age"].to_numpy(np.float64)
genes_c = ["GZMH", "KIR3DL2", "CR2", "LRRN3"]
bins = np.array([10, 30, 50, 70, 90, 115])
binm = (bins[:-1] + bins[1:]) / 2
n_genes = len(genes_c)
y_means = np.zeros((n_genes, len(binm)))
y_sems = np.zeros_like(y_means)
for j, g in enumerate(genes_c):
    if g not in X_tr.columns:
        print(f"[warn] {g} not in expr"); continue
    v = X_tr[g].to_numpy(np.float64)
    for b in range(len(binm)):
        m = (age >= bins[b]) & (age < bins[b + 1])
        if m.sum() >= 3:
            y_means[j, b] = v[m].mean()
            y_sems[j, b] = v[m].std() / np.sqrt(m.sum())

# ---------------- figure (H54: 2 rows x 3 + bottom full-width panel G) ----------------
fig = plt.figure(figsize=(11, 8.6))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85], wspace=0.85, hspace=0.95,
                      left=0.20, right=0.98, top=0.94, bottom=0.07)

# A: immune pathway NES horizontal bars
ax = fig.add_subplot(gs[0, 0])
colors = [C_G if v >= 0 else C_BL for v in top["NES"]]
ax.barh(range(len(top)), top["NES"], color=colors, alpha=0.85)
ax.set_yticks(range(len(top))); ax.set_yticklabels(short, fontsize=6.5)
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("Normalised enrichment score (NES)")
ax.set_title("Immune pathways enriched in the\ntranscriptomic age axis", fontsize=8)
panel(ax, "A")

# B: all strong hits by source (NES vs FDR)
ax = fig.add_subplot(gs[0, 1])
for src, sub in strong.groupby("source"):
    ax.scatter(sub["NES"], -np.log10(sub["FDR q-val"] + 1e-12),
               s=22, color=src_color[src], alpha=0.75, label=src, edgecolors="none")
ax.axhline(-np.log10(0.05), color="grey", ls="--", lw=0.8)
ax.text(ax.get_xlim()[1] * 0.99, -np.log10(0.05) + 0.08, "FDR=0.05", ha="right", fontsize=6.5, color="grey")
ax.set_xlabel("NES"); ax.set_ylabel("$-$log$_{10}$(FDR)")
ax.set_title("All pathways enriched at FDR<0.05\n(n=%d)" % len(strong), fontsize=8)
ax.legend(frameon=False, fontsize=6)
panel(ax, "B")

# C: representative driving transcripts (age-up + age-down, complements panel G)
ax = fig.add_subplot(gs[0, 2])
markers_c = ["GZMH", "KIR3DL2", "CR2", "LRRN3"]
labels_c = ["GZMH (NK effector)", "KIR3DL2 (KIR)", "CR2 (naive-B)", "LRRN3 (immuno-senescence)"]
cols_c = [C_G if rho.set_index("gene").loc[g, "rho"] > 0 else C_BL for g in markers_c]
rhos_c = [rho.set_index("gene").loc[g, "rho"] for g in markers_c]
ax.barh(range(4), rhos_c, color=cols_c, alpha=0.85)
ax.set_yticks(range(4)); ax.set_yticklabels(labels_c, fontsize=6.5)
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("Spearman $\\rho$ with age")
ax.set_title("Representative driving transcripts", fontsize=8)
panel(ax, "C")

# D: per-source NES boxen (strong hits)
ax = fig.add_subplot(gs[1, 0])
for i, src in enumerate(["GOBP", "KEGG", "Reactome", "WikiPath"]):
    v = strong[strong["source"] == src]["NES"]
    ax.scatter(np.full(len(v), i) + np.random.uniform(-0.15, 0.15, len(v)), v,
               s=18, color=src_color[src], alpha=0.8, edgecolors="none")
    ax.axhline(0, color="grey", lw=0.5, ls=":")
ax.set_xticks(range(4)); ax.set_xticklabels(["GO-BP", "KEGG", "React.", "WikiP."], fontsize=7)
ax.set_ylabel("NES"); ax.set_title("Strong-hit NES by library", fontsize=8)
panel(ax, "D")

# E: enrichment direction consistency (age-up vs age-down hits)
ax = fig.add_subplot(gs[1, 1])
n_pos = (strong["NES"] > 0).sum(); n_neg = (strong["NES"] < 0).sum()
ax.bar(["age-up", "age-down"], [n_pos, n_neg],
       color=[C_G, C_BL], alpha=0.85)
for i, v in enumerate([n_pos, n_neg]):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=8)
ax.set_ylabel("Gene sets (FDR<0.05)")
ax.set_title("Enrichment polarity", fontsize=8)
panel(ax, "E")

# F: FDR distribution of strong hits
ax = fig.add_subplot(gs[1, 2])
ax.hist(strong["FDR q-val"], bins=12, color=C_NV, alpha=0.85, edgecolor="white")
ax.axvline(0.01, color="grey", ls="--", lw=0.8)
ax.set_xlabel("FDR"); ax.set_ylabel("Count")
ax.set_title("FDR of strong hits", fontsize=8)
panel(ax, "F")

# G: bottom full-width — gene-age trends for 4 driving transcripts
ax = fig.add_subplot(gs[2, :])
markers = ["o", "s", "^", "D"]
for j, g in enumerate(genes_c):
    ax.errorbar(binm, y_means[j], yerr=y_sems[j], marker=markers[j], ms=4,
                lw=1.2, capsize=2, label=g, color=[C_G, C_OURS, C_NV, C_BL][j])
ax.set_xlabel("Chronological age (yr)")
ax.set_ylabel("Expression (log scale)")
ax.set_title("Driving transcripts across the age axis: NK effectors rise, naive-B/immunosenescence markers fall", fontsize=8)
ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper right")
panel(ax, "G")

save(fig, "Fig9_gsea", 183)
print("Fig9_gsea.png/pdf done")
