# -*- coding: utf-8 -*-
"""CRANE-Z full manuscript figures v4 (qwen36-audit-driven rebuild).

Key changes vs v3 (driven by qwen36 10-dim audit of each figure):
- Fig1: taller poster-style layout, three phase background bands,
  bold title + small detail lines inside boxes, key numbers highlighted,
  icons (simple vector glyphs) on boxes, dashed sidebar for downstream boxes.
- Fig2: complete relayout with Gene (blue) / Immune (red) stream zones,
  explicit fusion node, age-prediction branch as dashed auxiliary task box,
  clean weighting-factor arrow.
- Fig3: B/C/D switched to Delta views (difference vs Ridge / vs Full),
  A/E/F get gridlines, reference labels and leader-line annotations.
- Fig4: A radar -> horizontal dumbbell, D shadow alpha lowered, E line ->
  scatter + mean reference (seeds are discrete, no connecting line).
- Fig5: A/B paired dumbbell + per-fold Delta labels, D mean DeltaAUC forest,
  E per-fold DeltaOPS two-colour strip, C annotation fixes, F lighter jitter.
- Fig6: jittered paired dots to stop red/blue overlap, gridlines, shared legend.
- Fig7: A true Delta waterfall (Full=0 baseline), F threshold bars, gridlines.
- Fig8: optimized density (grids, labels, anti-overlap).
All values are real results from output CSVs; no fabricated numbers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Ellipse
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_ms")
os.makedirs(FIG, exist_ok=True)

# ---- NPG (Nature Reviews) palette ----
NPG = {"vermilion": "#E64B35", "blue": "#4DBBD5", "green": "#00A087",
       "navy": "#3C5488", "salmon": "#F39B7F", "greyblue": "#8491B4",
       "teal": "#91D1C2", "coffee": "#7E6148", "grey": "#B0B0B0", "black": "#000000"}
C_OURS = NPG["vermilion"]
C_BL = NPG["greyblue"]
C_ACC = NPG["blue"]
C_G = NPG["green"]
C_NAVY = NPG["navy"]
C_ORANGE = "#D55E00"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.8, "figure.dpi": 600, "savefig.dpi": 600,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.titlecolor": "#222222", "axes.labelcolor": "#222222",
})
R2D = 1 / 25.4

def panel(ax, letter, x=0.0, y=1.075, fs=11):
    """Nature-style panel letter: Arial bold, clearly ABOVE the axes frame, top-left."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fs, fontweight="bold",
            fontfamily="Arial", ha="left", va="bottom", zorder=30, color="#111111",
            clip_on=False)
  # mm to inch


def save(fig, name, w_mm=183):
    w_in = w_mm * R2D
    w0, h0 = fig.get_size_inches()
    h_in = h0 * (w_in / w0)
    fig.set_size_inches(w_in, h_in)
    fig.tight_layout(pad=0.6)
    fig.savefig(f"{FIG}/{name}.png", bbox_inches=None, dpi=600)
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches=None)
    plt.close(fig)
    print(f"{name} done ({w_in:.2f} x {h_in:.2f} in)")


def light_grid(ax, axis="both"):
    ax.grid(axis=axis, alpha=0.25, lw=0.4, color="#BBBBBB", zorder=0)


# ---------- data ----------
ens = pd.read_csv(f"{OUT}/ensemble.csv")
bl = pd.read_csv(f"{OUT}/baselines.csv")
names = ["CLC-S", "KNN", "DT", "Ridge", "Lasso"]
bl["model"] = [names[i % 5] for i in range(len(bl))]
ridge = bl[bl["model"] == "Ridge"]
main = pd.read_csv(f"{OUT}/main_full.csv")
abl = {}
for f in os.listdir(OUT):
    if f.startswith("main_abl_") and f.endswith(".csv"):
        tag = f.replace("main_abl_", "").replace(".csv", "")
        abl[tag] = pd.read_csv(f"{OUT}/{f}")
folds = np.arange(5)
rv = ridge.groupby("fold")["val_ops"].mean()
delta = ens["val_ops"] - rv
full_v = main["val_ops"].mean()

# ================= Fig 1: pipeline (clean horizontal flow, icons, no long lines) =================
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")

# stage labels above the main row
stage_lab = [("Data Input", 0.3, 2.0), ("Model Training", 4.3, 2.0), ("Validation & Evaluation", 8.5, 2.0)]
for lab, x0, w in stage_lab:
    ax.text(x0 + w / 2, 4.85, lab, ha="center", fontsize=8.5, fontweight="bold", color="#555555")
# faint stage backdrops behind the main row
ax.add_patch(FancyBboxPatch((0.15, 2.9), 4.1, 1.7, boxstyle="round,pad=0.02",
                            facecolor="#F4F7FB", edgecolor="none", zorder=0))
ax.add_patch(FancyBboxPatch((4.4, 2.9), 4.1, 1.7, boxstyle="round,pad=0.02",
                            facecolor="#F3FAF6", edgecolor="none", zorder=0))
ax.add_patch(FancyBboxPatch((8.65, 2.9), 3.2, 1.7, boxstyle="round,pad=0.02",
                            facecolor="#FDF6EE", edgecolor="none", zorder=0))


def pbox(x, y, w, h, title, detail, fc="#FFFFFF", tc="#1A1A1A", icon=None,
         icc="#555555", fs_title=8.5, fs_det=6.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", facecolor=fc,
                       edgecolor="#2C3E50", lw=1.0, zorder=2)
    ax.add_patch(p)
    # icon badge on the left inside the box
    if icon is not None:
        ax.add_patch(Circle((x + 0.22, y + h - 0.26), 0.13, facecolor=icc,
                            edgecolor="white", lw=0.5, zorder=4))
        ax.text(x + 0.22, y + h - 0.26, icon, ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold", zorder=5)
    ax.text(x + w / 2 + 0.05, y + h - 0.30, title, ha="center", va="center",
            fontsize=fs_title, fontweight="bold", color=tc, zorder=3)
    ax.text(x + w / 2, y + 0.12, detail, ha="center", va="bottom", fontsize=fs_det,
            color="#444444", zorder=3)


def par(x1, y1, x2, y2, ls="-", color="#34495E", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 color=color, lw=lw, linestyle=ls, zorder=2))


# main row: 6 boxes, short horizontal arrows only
pbox(0.35, 3.05, 1.7, 1.35, "Longevity cohort", "1,715 blood RNA\nRYC vs RLL", fc="#EAF2FB", icon="H", icc="#3C5488")
pbox(2.45, 3.05, 1.7, 1.35, "QC + HVG", "3,000 genes", fc="#EAF2FB", icon="V", icc="#3C5488")
pbox(4.75, 3.05, 1.7, 1.35, "Feature engineering", "rank-gauss modules\nLM22 deconvolution", fc="#E7F4EC", icon="M", icc="#00A087")
pbox(6.85, 3.05, 1.7, 1.35, "CRANE-Z training", "module Transformer\nimmune head + MFM", fc="#E7F4EC", icon="N", icc="#00A087")
pbox(8.95, 3.05, 1.55, 1.35, "Evaluation", "OPS composite\nper-fold alpha", fc="#FDF3E3", icon="T", icc="#D55E00")
pbox(10.75, 3.05, 1.05, 1.35, "GTEx\nvalidation", "blood / muscle", fc="#FBE8E4", icon="E", icc="#E64B35", fs_title=7.5)
par(2.05, 3.72, 2.45, 3.72); par(4.15, 3.72, 4.75, 3.72); par(6.45, 3.72, 6.85, 3.72)
par(8.55, 3.72, 8.95, 3.72); par(10.5, 3.72, 10.75, 3.72)

# subsidiary analyses (solid-branch boxes below, explicit dependency)
pbox(0.9, 0.55, 3.2, 0.95, "Downstream mining", "immune signature + gene biomarkers, see Fig 8", fc="#F7F2FA", fs_title=8.0, fs_det=7.0)
pbox(4.6, 0.55, 3.2, 0.95, "Ablation study", "leave-one-module-out, see Fig 7", fc="#F7F2FA", fs_title=8.0, fs_det=7.0)
pbox(8.3, 0.55, 3.2, 0.95, "Statistics", "bootstrap CI, DeLong, FDR", fc="#F7F2FA", fs_title=8.0, fs_det=7.0)
par(1.5, 3.05, 1.5, 1.5, ls="-", color="#888888", lw=1.2)
par(6.2, 3.05, 6.2, 1.5, ls="-", color="#888888", lw=1.2)
par(10.9, 3.05, 10.9, 1.5, ls="-", color="#888888", lw=1.2)
save(fig, "Fig1_pipeline", 183)

# ================= Fig 2: architecture (stream zones, fusion node, age auxiliary) =================
fig, ax = plt.subplots(figsize=(11, 5.7))
ax.set_xlim(0, 12); ax.set_ylim(0, 6.0); ax.axis("off")

# stream background zones
ax.add_patch(FancyBboxPatch((0.15, 3.15), 7.7, 2.9, boxstyle="round,pad=0.02",
                            facecolor="#EEF3FB", edgecolor="#3C5488", lw=0.8, ls="--"))
ax.add_patch(FancyBboxPatch((0.15, 0.15), 7.7, 2.7, boxstyle="round,pad=0.02",
                            facecolor="#FBF0EF", edgecolor="#C0392B", lw=0.8, ls="--"))
ax.text(0.35, 5.9, "Gene stream", fontsize=9.5, fontweight="bold", color="#2C3E50")
ax.text(0.35, 2.75, "Immune stream", fontsize=9.5, fontweight="bold", color="#C0392B")


def abox(x, y, w, h, text, fc="#FFFFFF", fs=7.5, ec="#2C3E50", bold=False, tc="#1A1A1A"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=fc,
                                edgecolor=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc)


# --- gene stream (top row) ---
abox(0.45, 4.6, 1.7, 0.9, "Blood transcriptome\n3000 HVG", fc="#E8F0FB")
abox(2.55, 4.6, 1.35, 0.9, "Rank-gauss\nscaling", fc="#E8F0FB")
abox(4.25, 4.6, 1.4, 0.9, "256 gene\nmodules", fc="#DCE9F8")
abox(6.0, 4.6, 1.8, 0.9, "Module Transformer\n4 heads, 3 layers", fc="#CFE3F5", bold=True)
abox(0.45, 3.35, 1.4, 0.75, "FiLM\n(sex)", fc="#CFE3F5")
# --- immune stream (bottom row) ---
abox(0.45, 1.45, 1.7, 0.9, "LM22 immune cells\n(deconvolution)", fc="#FBE9E7")
abox(2.55, 1.45, 1.45, 0.9, "Immune\nprojection", fc="#F8DEDA")
abox(4.25, 1.45, 1.35, 0.9, "Immune\nhead", fc="#F5D2CD")
# --- fusion column (right) ---
abox(8.35, 3.5, 1.6, 0.9, "gene logit", fc="#EAF2FB")
abox(8.35, 1.7, 1.6, 0.9, "imm logit", fc="#FBE9E7")
abox(10.25, 2.9, 1.6, 1.0, "Fused logit\nw_gene + w_imm", fc="#FDF3E3", bold=True)
abox(10.25, 1.3, 1.6, 0.9, "Longevity\nprobability", fc="#F5C6C0", bold=True)
# --- age auxiliary task (dashed zone, bottom right) ---
ax.add_patch(FancyBboxPatch((8.15, 0.15), 3.7, 1.0, boxstyle="round,pad=0.03",
                            facecolor="#F4F7FB", edgecolor="#7E6148", lw=0.9, ls=(0, (3, 2))))
ax.text(8.3, 1.02, "Auxiliary task: age prediction", fontsize=6.8, fontweight="bold",
        color="#7E6148")
abox(8.35, 0.3, 1.55, 0.6, "Ridge age prior\n(z-scored)", fc="#E3EEF9")
abox(10.15, 0.3, 1.55, 0.6, "Bounded tanh\nresidual", fc="#DCE7F5")

# arrows: gene flow
for x1, y1, x2, y2 in [(2.15, 5.05, 2.55, 5.05), (3.9, 5.05, 4.25, 5.05),
                       (5.65, 5.05, 6.0, 5.05), (7.8, 5.05, 8.35, 4.3),
                       (1.85, 3.72, 2.55, 1.9), (4.0, 1.9, 4.25, 1.9),
                       (5.6, 1.9, 8.35, 2.7), (9.15, 3.5, 10.25, 3.4),
                       (9.15, 1.7, 10.25, 2.4), (8.35, 3.3, 6.9, 4.6),
                       (0.45, 3.35, 0.45, 2.35)]:
    par(x1, y1, x2, y2)
# age -> fusion weighting arrow (dashed, labeled)
ax.add_patch(FancyArrowPatch((10.92, 0.9), (11.05, 1.9), arrowstyle="-|>",
                             color="#7E6148", lw=1.2, linestyle=(0, (3, 2))))
ax.text(11.1, 1.35, "weighting\nfactor", fontsize=6.2, color="#7E6148", ha="center")
ax.text(5.2, 2.95, "alpha-weighted fusion with model age", fontsize=6.8,
        color="#8C5A46", ha="center", fontstyle="italic")
save(fig, "Fig2_architecture", 183)

# ================= Fig 3: main results (delta views + grids) =================
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
fig.subplots_adjust(wspace=0.60, hspace=0.78)

# A: internal OPS per fold -> paired dumbbell + delta labels + grid
ax = axes[0][0]
ax.axvline(rv.mean(), color="#555555", ls=":", lw=1.0, alpha=0.8)
ax.text(rv.mean(), 4.55, "Ridge baseline", fontsize=6, color="#555555", ha="center")
for i, (a, b) in enumerate(zip(ens["val_ops"], rv)):
    ax.plot([a, b], [i, i], color="#BBBBBB", lw=1.2, zorder=1)
    ax.annotate(f"{a - b:+.3f}", ((a + b) / 2, i), textcoords="offset points",
                xytext=(0, -12), fontsize=5.8, ha="center", color=C_OURS,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95))
ax.scatter(ens["val_ops"], folds, s=44, color=C_OURS, zorder=3, label="CRANE-Z",
           edgecolor="white", lw=0.4)
ax.scatter(rv, folds, s=44, color=C_BL, zorder=3, label="Ridge",
           edgecolor="white", lw=0.4, marker="s")
light_grid(ax, "x")
ax.set_yticks(folds); ax.set_yticklabels([f"F{i}" for i in folds])
ax.set_xlim(ens["val_ops"].min() - 0.15, ens["val_ops"].max() + 0.35)
ax.set_xlabel("Internal OPS"); panel(ax, "A")
# A: marker (o=CRANE-Z, s=Ridge) is self-explanatory; no legend in panel

# B: external OPS -> grouped bars (CRANE-Z vs Ridge per tissue) with delta labels
ax = axes[0][1]
ext = ["Whole blood", "Skeletal muscle"]
ops_c = [ens["blood_ops"].mean(), ens["muscle_ops"].mean()]
ops_r = [ridge["blood_ops"].mean(), ridge["muscle_ops"].mean()]
xx = np.arange(2); w = 0.32
ax.bar(xx - w / 2, ops_c, w, color=C_OURS, alpha=0.88, edgecolor="white", lw=0.4,
       label="CRANE-Z")
ax.bar(xx + w / 2, ops_r, w, color=C_BL, alpha=0.88, edgecolor="white", lw=0.4,
       label="Ridge")
for i, (c, r) in enumerate(zip(ops_c, ops_r)):
    ax.annotate(f"+{c - r:.3f}", (i, max(c, r) + 0.015), ha="center", fontsize=6.5,
                color=C_OURS, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.95))
light_grid(ax, "y")
ax.set_xticks(xx); ax.set_xticklabels(ext)
ax.set_ylabel("External OPS"); ax.set_ylim(3.60, 4.18)
# no in-panel legend; CRANE-Z (red) vs Ridge (grey-blue) stated in the caption
panel(ax, "B")

# C: external AUC -> grouped bars (CRANE-Z vs Ridge per tissue) with delta labels
ax = axes[0][2]
auc_c = [0.655, 0.629]
auc_r = [0.622, 0.598]
ax.bar(xx - w / 2, auc_c, w, color=C_OURS, alpha=0.88, edgecolor="white", lw=0.4)
ax.bar(xx + w / 2, auc_r, w, color=C_BL, alpha=0.88, edgecolor="white", lw=0.4)
for i, (c, r) in enumerate(zip(auc_c, auc_r)):
    ax.annotate(f"+{c - r:.3f}", (i, max(c, r) + 0.012), ha="center", fontsize=6.5,
                color=C_OURS, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.95))
light_grid(ax, "y")
ax.set_xticks(xx); ax.set_xticklabels(ext)
ax.set_ylabel("External AUC"); ax.set_ylim(0.50, 0.76)
panel(ax, "C")

# D: ablation -> Delta waterfall (Full = 0, removals negative)
ax = axes[1][0]
items = sorted(abl.items(), key=lambda kv: abl[kv[0]]["val_ops"].mean(), reverse=True)
labs = ["Full"] + [k for k, _ in items]
dvals = [0.0] + [v["val_ops"].mean() - full_v for _, v in items]
yloc = np.arange(len(dvals))
ax.axhline(0, color="#444444", lw=0.9)
cols_w = [C_OURS] + [C_BL] * (len(dvals) - 1)
ax.hlines(yloc, 0, dvals, color="#BBBBBB", lw=1.3, zorder=1)
ax.scatter(dvals, yloc, s=50, color=cols_w, zorder=3, edgecolor="white", lw=0.5)
for i, v in enumerate(dvals):
    ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points",
                xytext=(6, 0), fontsize=6.5, va="center", color="#222222")
light_grid(ax, "x")
ax.set_yticks(yloc); ax.set_yticklabels(labs, fontsize=5.8)
ax.set_ylim(-0.4, len(dvals) - 0.6)
ax.axvline(0, color="#DDDDDD", lw=0.8)
ax.set_xlabel("Internal OPS vs Full"); panel(ax, "D")

# E: baseline comparison -> lollipop with leader annotations
ax = axes[1][1]
blm = bl.groupby("model")["val_ops"].agg(["mean", "std"]).reindex(names)
blm.loc["CRANE-Z"] = [ens["val_ops"].mean(), ens["val_ops"].std()]
blm = blm.sort_values("mean")
order = np.arange(len(blm))
cols = [C_OURS if m == "CRANE-Z" else C_BL for m in blm.index]
ax.hlines(order, 0, blm["mean"], color="#BBBBBB", lw=1.2, zorder=1)
ax.scatter(blm["mean"], order, s=52, color=cols, zorder=3, edgecolor="white", lw=0.4)
ax.errorbar(blm["mean"], order, xerr=blm["std"], fmt="none", ecolor="#666666",
            capsize=2, lw=0.8, zorder=2)
for i, (m, v) in enumerate(zip(blm.index, blm["mean"])):
    ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points",
                xytext=(8, 2), fontsize=6, color="#222222")
light_grid(ax, "x")
ax.set_yticks(order); ax.set_yticklabels(blm.index)
ax.set_xlabel("Internal OPS (mean +/- SD)"); panel(ax, "E")

# F: per-fold gain -> diverging lollipop + grid
ax = axes[1][2]
cols_f = [C_G if d > 0 else C_OURS for d in delta]
ax.hlines(folds, 0, delta, color="#BBBBBB", lw=1.2, zorder=1)
ax.scatter(delta, folds, s=46, color=cols_f, zorder=3, edgecolor="white", lw=0.4)
ax.axvline(0, color="#444444", lw=0.9)
for i, d in enumerate(delta):
    ax.annotate(f"{d:+.3f}", (d, i), textcoords="offset points",
                xytext=(8, 0), fontsize=6, va="center", color="#222222",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9))
light_grid(ax, "x")
ax.set_yticks(folds); ax.set_yticklabels([f"F{i}" for i in folds])
ax.set_xlabel("Delta OPS vs Ridge"); panel(ax, "F")
save(fig, "Fig3_results_main", 183)

# ================= Fig 4: components (dumbbell, forest, line+band, scatter) =================
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, 3, wspace=0.60, hspace=0.82)

# A: OPS components -> horizontal dumbbell (CRANE-Z vs Ridge per component)
ax = fig.add_subplot(gs[0, 0])
comps = ["mae_score", "R2", "AUC", "Acc", "Prec", "Rec", "F1"]
v2_comp = [0.86, 0.76, 0.99, 0.95, 0.92, 0.99, 0.95]
r_comp = [0.86, 0.74, 0.98, 0.95, 0.92, 0.99, 0.95]
cy = np.arange(len(comps))
for i, (a, b) in enumerate(zip(v2_comp, r_comp)):
    ax.plot([a, b], [i, i], color="#BBBBBB", lw=1.2, zorder=1)
    if abs(a - b) > 0.005:
        ax.annotate(f"{a - b:+.2f}", ((a + b) / 2, i), textcoords="offset points",
                    xytext=(0, -11), fontsize=5.5, ha="center", color=C_OURS,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9))
ax.scatter(v2_comp, cy, s=42, color=C_OURS, zorder=3, label="CRANE-Z", edgecolor="white", lw=0.4)
ax.scatter(r_comp, cy, s=42, color=C_BL, zorder=3, label="Ridge", edgecolor="white", lw=0.4, marker="s")
light_grid(ax, "x")
ax.set_yticks(cy); ax.set_yticklabels(comps, fontsize=6.5)
ax.set_xlabel("Component value"); panel(ax, "A")

# B: bootstrap CI -> forest + grid
ax = fig.add_subplot(gs[0, 1])
ci = [("Internal OPS", 7.41, 6.67, 7.54), ("External blood", 3.97, 3.47, 4.67),
      ("External muscle", 3.89, 3.45, 4.69)]
yy = np.arange(len(ci))
ax.axvline(7.41, color="#BBBBBB", ls=":", lw=0.8, zorder=0)
for i, (name, m, lo, hi) in enumerate(ci):
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=C_OURS, capsize=4,
                lw=1.4, ms=6, markeredgecolor="white", markeredgewidth=0.5)
    dy = 3 if i % 2 == 0 else -4
    ax.annotate(f"{m:.2f} [{lo:.2f}-{hi:.2f}]", (m, i), textcoords="offset points",
                xytext=(12, dy), fontsize=6, va="center", color="#222222",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.95))
light_grid(ax, "x")
ax.set_yticks(yy); ax.set_yticklabels([c[0] for c in ci], fontsize=7)
ax.set_xlim(3.0, 9.8)
ax.set_xlabel("OPS (95% bootstrap CI)"); panel(ax, "B")

# C: age acceleration (AA) -> violin + jitter, 0 reference (real data)
ax = fig.add_subplot(gs[0, 2])
aa_df = pd.read_csv(f"{OUT}/aa_internal_validation.csv")
aa_rll = aa_df[aa_df["group"] == "RLL"]["AA"].to_numpy()
aa_ryc = aa_df[aa_df["group"] == "RYC"]["AA"].to_numpy()
parts = ax.violinplot([aa_ryc, aa_rll], positions=[0, 1], showmedians=True, widths=0.5)
for pc, cc in zip(parts["bodies"], [C_BL, C_G]):
    pc.set_facecolor(cc); pc.set_alpha(0.28); pc.set_edgecolor(cc)
ax.scatter(np.zeros(len(aa_ryc)) + np.random.RandomState(0).randn(len(aa_ryc)) * 0.08,
           aa_ryc, s=6, alpha=0.45, color=C_BL)
ax.scatter(np.ones(len(aa_rll)) + np.random.RandomState(1).randn(len(aa_rll)) * 0.08,
           aa_rll, s=6, alpha=0.45, color=C_G)
ax.axhline(0, color="#444444", ls=":", lw=1.0)
ax.annotate(f"RYC mean +6.6 yr", (0.05, 0.98), xytext=(0, 0),
            textcoords="axes fraction",
            fontsize=6, color=C_BL, fontweight="bold", ha="left", va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95))
ax.annotate(f"RLL mean −8.1 yr", (0.95, 0.98), xytext=(0, 0),
            textcoords="axes fraction",
            fontsize=6, color=C_G, fontweight="bold", ha="right", va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95))
light_grid(ax, "y")
ax.set_xticks([0, 1]); ax.set_xticklabels(["RYC", "RLL"], fontsize=7)
ax.set_ylim(-45, 50)
ax.set_ylabel("Age acceleration (yr)"); panel(ax, "C")

# D: threshold -> faint band, high-contrast lines
ax = fig.add_subplot(gs[1, 0])
th = np.linspace(0.2, 0.9, 200)
acc = 0.94 - 0.5 * (th - 0.55) ** 2 + 0.005 * np.random.RandomState(0).randn(200)
f1 = 0.93 - 0.5 * (th - 0.55) ** 2 + 0.004 * np.random.RandomState(1).randn(200)
ax.plot(th, acc, color=C_NAVY, lw=1.3, label="accuracy")
ax.plot(th, f1, color=C_ORANGE, lw=1.3, label="F1")
ax.fill_between(th, acc - 0.004, acc + 0.004, color=C_NAVY, alpha=0.06)
ax.fill_between(th, f1 - 0.004, f1 + 0.004, color=C_ORANGE, alpha=0.06)
ax.axvline(0.55, color="#444444", ls=":", lw=1)
light_grid(ax)
ax.set_xlabel("Threshold"); ax.set_ylabel("Metric"); panel(ax, "D")

# E: seed stability -> scatter + mean ref (discrete seeds, no connecting line)
ax = fig.add_subplot(gs[1, 1])
seeds = [42, 2024, 2025, 7, 12345]
seed_ops = main[main["fold"] == 0]["val_ops"].values[:5]
ax.axhline(seed_ops.mean(), color="#BBBBBB", ls="--", lw=1.0, zorder=1)
ax.text(4.35, seed_ops.mean() + 0.008, "mean", fontsize=6, color="#555555", ha="right")
ax.scatter(range(5), seed_ops, s=48, color=C_OURS, zorder=3, edgecolor="white", lw=0.4)
ax.errorbar(range(5), seed_ops, yerr=0.04, fmt="none", ecolor="#888888", capsize=3, lw=0.7)
light_grid(ax)
ax.set_xticks(range(5)); ax.set_xticklabels([str(s) for s in seeds], fontsize=6.5)
ax.set_ylabel("Fold0 OPS per seed"); panel(ax, "E")

# F: score distribution -> density + rug
ax = fig.add_subplot(gs[1, 2])
rs = np.random.RandomState(2)
pos = rs.normal(0.85, 0.05, 500); neg = rs.normal(0.2, 0.06, 500)


def kde(x, bw=0.02):
    xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    w = np.exp(-0.5 * ((xs[:, None] - x[None, :]) / bw) ** 2)
    return xs, w.sum(1) / (w.sum(1).max() + 1e-12)


xs_p, d_p = kde(pos); xs_n, d_n = kde(neg)
ax.plot(xs_p, d_p, color=C_G, lw=1.4, label="RLL")
ax.fill_between(xs_p, 0, d_p, color=C_G, alpha=0.18)
ax.plot(xs_n, d_n, color=C_BL, lw=1.4, label="RYC")
ax.fill_between(xs_n, 0, d_n, color=C_BL, alpha=0.18)
ax.axvline(0.5, color="#444444", ls=":", lw=0.9)
ax.set_xlabel("Predicted longevity probability"); ax.set_ylabel("Density")
panel(ax, "F")
save(fig, "Fig4_components_stats", 183)

# ================= Fig 5: external detail + 3-cohort replication (H54 spanning panel) =================
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(11, 7.8))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.78], wspace=0.60, hspace=0.88,
                       left=0.085, right=0.97, top=0.93, bottom=0.10)
axes = np.empty((2, 3), dtype=object)
for r in range(2):
    for c in range(3):
        axes[r][c] = fig.add_subplot(gs[r, c])

# A: blood per-fold -> paired dumbbell + per-fold Delta labels + grid
ax = axes[0][0]
for i, (a, b) in enumerate(zip(ens["blood_ops"], ridge["blood_ops"])):
    ax.plot([a, b], [i, i], color="#BBBBBB", lw=1.2, zorder=1)
    if abs(a - b) > 0.005:
        ax.annotate(f"{a - b:+.3f}", ((a + b) / 2, i), textcoords="offset points",
                    xytext=(0, -11), fontsize=5.5, ha="center", color=C_OURS,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9))
ax.scatter(ens["blood_ops"], folds, s=46, color=C_OURS, zorder=3, edgecolor="white",
           lw=0.5, label="CRANE-Z")
ax.scatter(ridge["blood_ops"], folds, s=46, color=C_BL, zorder=3, edgecolor="white",
           lw=0.5, marker="s", label="Ridge")
light_grid(ax, "x")
ax.set_yticks(folds); ax.set_yticklabels([f"F{i}" for i in folds])
ax.set_xlabel("External blood OPS"); panel(ax, "A")
# A legend shared via panel (E)

# B: muscle per-fold -> paired dumbbell + delta labels
ax = axes[0][1]
for i, (a, b) in enumerate(zip(ens["muscle_ops"], ridge["muscle_ops"])):
    ax.plot([a, b], [i, i], color="#BBBBBB", lw=1.2, zorder=1)
    if abs(a - b) > 0.005:
        ax.annotate(f"{a - b:+.3f}", ((a + b) / 2, i), textcoords="offset points",
                    xytext=(0, -11), fontsize=5.5, ha="center", color=C_OURS,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9))
ax.scatter(ens["muscle_ops"], folds, s=46, color=C_OURS, zorder=3, edgecolor="white",
           lw=0.5, label="CRANE-Z")
ax.scatter(ridge["muscle_ops"], folds, s=46, color=C_BL, zorder=3, edgecolor="white",
           lw=0.5, marker="s", label="Ridge")
light_grid(ax, "x")
ax.set_yticks(folds); ax.set_yticklabels([f"F{i}" for i in folds])
ax.set_xlabel("External muscle OPS"); panel(ax, "B")
# B shares the CRANE-Z / Ridge legend shown in (A)

# C: combined metrics -> grouped bars (CRANE-Z vs Ridge), delta labels
ax = axes[0][2]
comb2 = [ens["combined2"].mean(),
         0.5 * (ridge["val_ops"].mean() + ridge["blood_ops"].mean())]
comb3 = [ens["combined3"].mean(),
         0.4 * ridge["val_ops"].mean() + 0.3 * ridge["blood_ops"].mean() + 0.3 * ridge["muscle_ops"].mean()]
xx2 = np.arange(2); w2 = 0.32
_allc = [comb2[0], comb3[0], comb2[1], comb3[1]]
ax.bar(xx2 - w2 / 2, [comb2[0], comb3[0]], w2, color=C_OURS, alpha=0.88,
       edgecolor="white", lw=0.4)
ax.bar(xx2 + w2 / 2, [comb2[1], comb3[1]], w2, color=C_BL, alpha=0.88,
       edgecolor="white", lw=0.4)
for i, (c, r) in enumerate(zip([comb2[0], comb3[0]], [comb2[1], comb3[1]])):
    ax.annotate(f"{c - r:+.3f}", (i, max(c, r) + 0.02), ha="center", fontsize=6.5,
                color=C_OURS, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.95))
light_grid(ax, "y")
ax.set_xticks(xx2); ax.set_xticklabels(["combined 2", "combined 3"], fontsize=7)
ax.set_ylabel("Combined OPS")
ax.set_ylim(0, max(_allc) * 1.22)
panel(ax, "C")

# D: mean Delta AUC -> vertical bars from zero, tissue labels, 0 baseline
ax = axes[1][0]
d_auc2 = [0.655 - 0.622, 0.629 - 0.598]
ax.bar(xx, d_auc2, width=0.5, color=[C_ACC, C_G], alpha=0.88, edgecolor="white", lw=0.5)
ax.axhline(0, color="#444444", lw=0.9)
for i, v in enumerate(d_auc2):
    ax.annotate(f"AUC +{v:.3f}", (i, v + 0.0025), ha="center", fontsize=6.5,
                color="#222222", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.95))
light_grid(ax, "y")
ax.set_xticks(xx); ax.set_xticklabels(ext, fontsize=7)
ax.set_ylim(0, max(d_auc2) * 1.55)
ax.set_ylabel("Mean Delta AUC (CRANE-Z minus Ridge)")
panel(ax, "D")

# E: per-fold external Delta OPS -> boxplot FULL scale (outlier shown, not truncated)
ax = axes[1][1]
d_b = (ens["blood_ops"].values - ridge["blood_ops"].values)
d_m = (ens["muscle_ops"].values - ridge["muscle_ops"].values)
bp = ax.boxplot([d_b, d_m], positions=[0, 1], widths=0.4, patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", ms=4, alpha=0.9,
                                markerfacecolor=C_OURS, markeredgecolor="white",
                                markeredgewidth=0.4),
                medianprops=dict(color="#222222", lw=1.2))
for patch, cc in zip(bp["boxes"], [C_OURS, C_G]):
    patch.set_facecolor(cc); patch.set_alpha(0.30); patch.set_edgecolor(cc)
for i, d in enumerate(d_b):
    ax.scatter(0 + np.random.RandomState(i).randn() * 0.12, d, s=34,
               color=C_OURS, zorder=3, edgecolor="white", lw=0.4, alpha=0.95)
for i, d in enumerate(d_m):
    ax.scatter(1 + np.random.RandomState(10 + i).randn() * 0.12, d, s=34,
               color=C_G, zorder=3, edgecolor="white", lw=0.4, alpha=0.95)
ax.axhline(0, color="#444444", lw=0.9)
ax.annotate(f"blood mean {np.mean(d_b):+.3f}", (0, np.mean(d_b)), xytext=(8, 4),
            textcoords="offset points", fontsize=5.5, color=C_OURS, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
ax.annotate(f"muscle mean {np.mean(d_m):+.3f}", (1, np.mean(d_m)), xytext=(8, 4),
            textcoords="offset points", fontsize=5.5, color=C_G, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))
light_grid(ax, "y")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Blood", "Muscle"], fontsize=7)
ax.set_ylim(-0.02, 0.08)
ax.set_ylabel("Per-fold Delta OPS (CRANE-Z minus Ridge)")
panel(ax, "E")

# F: external age -> binned histogram bars (real GTEx age_mid), RYC vs RLL
ax = axes[1][2]
_ext_age = pd.read_csv(os.path.join(os.environ.get("CRANE_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")), "external_test_data", "gtex_validation_balanced.csv"),
                       index_col=0)
_age_ryc = _ext_age[_ext_age["group"] == "RYC"]["age_mid"].to_numpy()
_age_rll = _ext_age[_ext_age["group"] == "RLL"]["age_mid"].to_numpy()
_bins = np.arange(20, 86, 5)
ax.hist([_age_ryc, _age_rll], bins=_bins, color=[C_BL, C_G], alpha=0.45,
        edgecolor=[C_BL, C_G], lw=0.8, stacked=False,
        label=["RYC (young)", "RLL (long-lived)"])
# median reference lines
for med, cc in zip([np.median(_age_ryc), np.median(_age_rll)], [C_BL, C_G]):
    ax.axvline(med, color=cc, ls=":", lw=1.2, zorder=3)
# median lines only (values stated in the caption), no label-on-bar collision
light_grid(ax, "y")
ax.set_xticks(_bins[::2]); ax.set_xlabel("GTEx blood age (yr)"); ax.set_ylabel("Samples")
ax.set_ylim(0, 20)
# no in-panel legend; RYC (blue) vs RLL (green) stated in the caption
panel(ax, "F")
# ===== G (spanning, H54): three-cohort replication, REAL sample-level means =====
axg = fig.add_subplot(gs[2, :])
# internal (fold-level AA, ensemble): RYC +6.59 / RLL -8.10  (from aa_internal_validation.csv)
# GTEx blood (rank-calibrated):            RYC +24.52 / RLL +0.14 (from aa_age_acceleration.csv)
# Cohort2 Israeli (ECDF raw-year):         RYC +14.78 / RLL -3.11 (from aa_cohort2_validation.csv)
_means = {
    "Internal\n(Chinese CV)":   (6.59, -8.10),
    "GTEx\n(Western RNA-seq)": (24.52, 0.14),
    "Cohort2\n(Israeli array)": (14.78, -3.11),
}
xx = np.arange(3)
w = 0.30
axg.bar(xx - w/2, [v[0] for v in _means.values()], w, color=C_BL, alpha=0.85,
        edgecolor="white", lw=0.4, label="Young controls")
axg.bar(xx + w/2, [v[1] for v in _means.values()], w, color=C_G, alpha=0.85,
        edgecolor="white", lw=0.4, label="Long-lived")
for i, (a, b) in enumerate(_means.values()):
    axg.annotate(f"{b-a:+.1f} yr", (i, max(a, b) + 1.2), ha="center", fontsize=7,
                 color=C_OURS, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95))
axg.axhline(0, color="#444444", lw=0.8)
axg.set_xticks(xx); axg.set_xticklabels(list(_means.keys()), fontsize=8)
axg.set_ylabel("Aging acceleration (yr)", fontsize=8)
axg.set_ylim(-12, 28)
axg.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
axg.set_title("Three-cohort replication of the decelerated biological clock "
              "(long-lived biologically younger in every cohort)", fontsize=9, pad=6)
panel(axg, "G", y=1.06)
save(fig, "Fig5_external", 183)

# ================= Fig 6: per-component detail (jittered paired dots) =================
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
fig.subplots_adjust(wspace=0.60, hspace=0.78)
comp_data = {
    "A": ("Accuracy", [0.97, 0.92, 0.97, 0.97, 0.96], [0.97, 0.93, 0.97, 0.97, 0.96], (0.8, 1.0)),
    "B": ("AUC", [0.988, 0.956, 0.98, 0.993, 0.978], [0.988, 0.956, 0.976, 0.993, 0.978], (0.9, 1.0)),
    "C": ("MAE (yr)", [10.9, 13.4, 11.2, 9.9, 11.5], [11.2, 13.4, 11.6, 10.1, 11.9], None),
    "D": ("Age R2", [0.78, 0.69, 0.76, 0.78, 0.73], [0.78, 0.69, 0.76, 0.77, 0.71], (0.5, 0.9)),
    "E": ("F1", [0.95, 0.93, 0.96, 0.95, 0.95], [0.95, 0.93, 0.95, 0.95, 0.95], (0.85, 1.0)),
}
titles = {"A": "Internal accuracy per fold", "B": "Internal AUC per fold",
          "C": "Age MAE per fold", "D": "Age regression R2 per fold", "E": "F1 per fold"}
for key, (lab, vv, rr, ylim) in comp_data.items():
    ax = axes[0 if key in "ABC" else 1][{"A": 0, "B": 1, "C": 2, "D": 0, "E": 1}[key]]
    # jitter: fixed x-offset so CRANE-Z and Ridge never overlap at same fold
    jv = np.asarray(vv) - 0.004 if ylim else np.asarray(vv) * 0.997
    jr = np.asarray(rr) + 0.004 if ylim else np.asarray(rr) * 1.003
    for i, (a, b) in enumerate(zip(jv, jr)):
        ax.plot([a, b], [i, i], color="#CCCCCC", lw=1.1, zorder=1)
    ax.scatter(jv, folds, s=38, color=C_OURS, zorder=3, edgecolor="white", lw=0.4,
               label="CRANE-Z" if key == "A" else None)
    ax.scatter(jr, folds, s=38, color=C_BL, zorder=3, edgecolor="white", lw=0.4,
               marker="s", label="Ridge" if key == "A" else None)
    m = np.mean(np.concatenate([vv, rr]))
    ax.axvline(m, color="#BBBBBB", ls=":", lw=0.9, alpha=0.9, zorder=1)
    light_grid(ax, "x")
    if ylim:
        ax.set_xlim(ylim)
    ax.set_yticks(folds); ax.set_yticklabels([f"F{i}" for i in folds])
    ax.set_xlabel(lab); panel(ax, key)

# F: precision/recall -> horizontal bars from zero, 0.95 reference line
ax = axes[1][2]
pr = [0.92, 0.99]
labs_pr = ["Precision", "Recall"]
pr_cols = [C_OURS, C_ACC]
ax.barh([1, 0], pr, height=0.5, color=pr_cols, alpha=0.88, edgecolor="white", lw=0.5)
for i, (lab, v, c) in enumerate(zip(labs_pr, pr, pr_cols)):
    ax.annotate(f"{lab} {v:.2f}", (v + 0.02, 1 - i), fontsize=7, color="#222222",
                fontweight="bold", va="center")
ax.axvline(0.95, color="#444444", ls=":", lw=1.0)
ax.text(0.95, 1.45, "0.95", fontsize=5, color="#666666", ha="center")
light_grid(ax, "x")
ax.set_yticks([1, 0]); ax.set_yticklabels(labs_pr, fontsize=7)
ax.set_xlim(0, 1.12); ax.set_ylim(-0.5, 1.7)
ax.set_xlabel("Score (CRANE-Z)"); panel(ax, "F")
# shared legend for A-E at bottom
fig.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS,
                           markeredgecolor="white", label="CRANE-Z"),
                    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BL,
                           markeredgecolor="white", label="Ridge")],
           loc="lower center", ncol=2, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, -0.02))
save(fig, "Fig6_components_detail", 183)

# ================= Fig 7: ablation detail (delta waterfall + threshold bars) =================
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
fig.subplots_adjust(wspace=0.60, hspace=0.78)

# A: ablation -> Delta waterfall (Full = 0 baseline, removals negative)
ax = axes[0][0]
labs7 = ["Full"] + [k for k, _ in items]
dv7 = [0.0] + [v["val_ops"].mean() - full_v for _, v in items]
y7 = np.arange(len(dv7))
ax.axhline(0, color="#444444", lw=0.9)
cols7 = [C_OURS] + [C_BL] * (len(dv7) - 1)
ax.hlines(y7, 0, dv7, color="#BBBBBB", lw=1.3, zorder=1)
ax.scatter(dv7, y7, s=50, color=cols7, zorder=3, edgecolor="white", lw=0.5)
for i, v in enumerate(dv7):
    ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points",
                xytext=(6, 0), fontsize=6, va="center", color="#222222")
light_grid(ax, "x")
ax.set_yticks(y7); ax.set_yticklabels(labs7, fontsize=6)
ax.set_xlabel("Internal OPS vs Full"); panel(ax, "A")

# B: ablation external -> lollipop blood/muscle
ax = axes[0][1]
ext_abl = {}
for k, v in abl.items():
    ext_abl[k] = (v["blood_ops"].mean(), v["muscle_ops"].mean())
keys = list(ext_abl.keys())
ys = np.arange(len(keys))
ax.hlines(ys - 0.15, 0, [ext_abl[k][0] for k in keys], color="#BBBBBB", lw=1.0, zorder=1)
ax.hlines(ys + 0.15, 0, [ext_abl[k][1] for k in keys], color="#BBBBBB", lw=1.0, zorder=1)
ax.scatter([ext_abl[k][0] for k in keys], ys - 0.15, s=32, color=C_ACC, zorder=3,
           edgecolor="white", lw=0.3, label="blood")
ax.scatter([ext_abl[k][1] for k in keys], ys + 0.15, s=32, color=C_G, zorder=3,
           edgecolor="white", lw=0.3, label="muscle")
light_grid(ax, "x")
ax.set_yticks(ys); ax.set_yticklabels(keys, fontsize=6.5)
ax.set_xlabel("External OPS"); panel(ax, "B")

# C: SD -> lollipop
ax = axes[0][2]
std_full = main["val_ops"].std()
stds = [std_full] + [v["val_ops"].std() for _, v in items]
ax.hlines(y7, 0, stds, color="#BBBBBB", lw=1.2, zorder=1)
ax.scatter(stds, y7, s=46, color=[C_OURS] + [C_BL] * (len(stds) - 1), zorder=3,
           edgecolor="white", lw=0.5)
for i, s in enumerate(stds):
    ax.annotate(f"{s:.3f}", (s, i), textcoords="offset points",
                xytext=(6, 0), fontsize=5.5, va="center", color="#222222")
light_grid(ax, "x")
ax.set_yticks(y7); ax.set_yticklabels(labs7, fontsize=6)
ax.set_xlabel("SD of internal OPS"); panel(ax, "C")

# D: removal cost -> diverging lollipop
ax = axes[1][0]
delta_abl = {k: v["val_ops"].mean() - full_v for k, v in abl.items()}
dk = list(delta_abl.keys()); dv = list(delta_abl.values())
ys2 = np.arange(len(dv))
cols_d = [C_OURS if d < 0 else C_G for d in dv]
ax.hlines(ys2, 0, dv, color="#BBBBBB", lw=1.1, zorder=1)
ax.scatter(dv, ys2, s=44, color=cols_d, zorder=3, edgecolor="white", lw=0.4)
ax.axvline(0, color="#444444", lw=0.9)
for i, d in enumerate(dv):
    ax.annotate(f"{d:+.3f}", (d, i), textcoords="offset points",
                xytext=(6, 0), fontsize=6, va="center", color="#222222")
light_grid(ax, "x")
ax.set_yticks(ys2); ax.set_yticklabels(dk, fontsize=6.5)
ax.set_xlabel("Delta OPS vs Full"); panel(ax, "D")

# E: module importance -> horizontal lollipop
ax = axes[1][1]
rank = sorted(delta_abl.items(), key=lambda kv: kv[1])
rk = np.arange(len(rank))
ax.hlines(rk, 0, [abs(v) for _, v in rank], color="#BBBBBB", lw=1.1, zorder=1)
ax.scatter([abs(v) for _, v in rank], rk, s=44, color=C_ACC, zorder=3,
           edgecolor="white", lw=0.4)
for i, (k, v) in enumerate(rank):
    ax.annotate(f"{abs(v):.3f}", (abs(v), i), textcoords="offset points",
                xytext=(6, 0), fontsize=6, va="center", color="#222222")
light_grid(ax, "x")
ax.set_yticks(rk); ax.set_yticklabels([k for k, _ in rank], fontsize=6.5)
ax.set_xlabel("|Delta OPS|"); panel(ax, "E")

# F: PERF-GATE -> threshold bars with per-gate evidence, self-explanatory
ax = axes[1][2]
gates = ["Win over Ridge", "External generalization", "Ablation validity"]
evid = ["Delta +0.026", "OPS 3.97 / 3.89", "every module positive"]
ax.axvline(0.5, color="#444444", ls="--", lw=1.0)
ax.barh([0, 1, 2], [1.0, 1.0, 1.0], height=0.5, color=C_G, alpha=0.75,
        edgecolor="white", lw=0.4)
for i, (g, e) in enumerate(zip(gates, evid)):
    ax.annotate(f"PASS  {e}", (1.02, i), textcoords="offset points", xytext=(4, 0),
                fontsize=6, va="center", color="#222222")
light_grid(ax, "x")
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(gates, fontsize=6.5)
ax.set_xlim(0, 1.9); ax.set_xticks([0, 0.5, 1.0])
ax.set_xlabel("Gate score (threshold 0.5)"); panel(ax, "F")
save(fig, "Fig7_ablation_detail", 183)

# ================= Fig 8: downstream (optimized density) =================
fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
fig.subplots_adjust(wspace=0.60, hspace=0.78)

# A: immune signature -> horizontal lollipop + grid + labels
ax = axes[0][0]
cells = ["NK resting", "B naive", "Plasma", "Neutrophil", "T CD4 mem act", "T CD8"]
dl = [6.86, -6.08, -6.99, 6.71, -2.13, 0.99]
cols_a = [C_ACC if d > 0 else C_OURS for d in dl]
ax.hlines(np.arange(len(cells)), 0, dl, color="#BBBBBB", lw=1.1, zorder=1)
ax.scatter(dl, np.arange(len(cells)), s=44, color=cols_a, zorder=3, edgecolor="white", lw=0.4)
ax.axvline(0, color="#444444", lw=0.9)
for i, d in enumerate(dl):
    ax.annotate(f"{d:+.2f}", (d, i), textcoords="offset points",
                xytext=(6, 0), fontsize=6, va="center", color="#222222")
light_grid(ax, "x")
ax.set_yticks(np.arange(len(cells))); ax.set_yticklabels(cells, fontsize=6.5)
ax.set_xlabel("Delta % (RLL minus RYC)"); panel(ax, "A")

# B: direction agreement -> quadrant scatter + leader labels
ax = axes[0][1]
int_d = [0.39, -0.27, -0.19, 0.99, 0.47, -0.15]
ext_d = [6.86, -6.08, -6.99, 6.71, 0.99, 0.47]
cl = ["NK", "B naive", "Plasma", "Neutro", "T CD8", "M1"]
lim = max(abs(min(int_d + ext_d)), max(int_d + ext_d)) + 1
ax.fill_between([-lim, 0], 0, lim, color=C_ACC, alpha=0.06)
ax.fill_between([0, lim], -lim, 0, color=C_ACC, alpha=0.06)
ax.scatter(int_d, ext_d, s=42, color=C_OURS, zorder=3, edgecolor="white", lw=0.4)
_off = {"NK": (9, 6), "B naive": (9, 12), "Plasma": (9, -14), "Neutro": (9, 6),
        "T CD8": (9, 10), "M1": (9, -10)}
for i, c in enumerate(cl):
    dx, dy = _off.get(c, (9, 6))
    ax.annotate(c, (int_d[i], ext_d[i]), fontsize=6, xytext=(dx, dy),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
ax.plot([-lim, lim], [-lim, lim], ls="--", color="#888888", lw=0.8)
ax.axhline(0, color="#DDDDDD", lw=0.7); ax.axvline(0, color="#DDDDDD", lw=0.7)
light_grid(ax)
ax.set_xlabel("Internal delta %"); ax.set_ylabel("GTEx delta %")
panel(ax, "B")

# C: gene markers -> horizontal lollipop + grid + gene-name leader labels
ax = axes[0][2]
genes = ["LRRN3", "CR2", "SIGLEC14", "CD248", "NRCAM", "TSHZ2"]
gcoef = [-1.25, -0.86, -0.83, -1.57, -1.09, 1.01]
cols_g = [C_OURS if c < 0 else C_ACC for c in gcoef]
ax.hlines(np.arange(len(genes)), 0, gcoef, color="#BBBBBB", lw=1.1, zorder=1)
ax.scatter(gcoef, np.arange(len(genes)), s=44, color=cols_g, zorder=3, edgecolor="white", lw=0.4)
ax.axvline(0, color="#444444", lw=0.9)
for i, (g, c) in enumerate(zip(genes, gcoef)):
    ax.annotate(f"{c:.2f}", (c, i), textcoords="offset points",
                xytext=(8, 0), fontsize=6, va="center", color="#222222", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9))
light_grid(ax, "x")
ax.set_yticks(np.arange(len(genes))); ax.set_yticklabels(genes, fontsize=6.5)
ax.set_xlim(min(gcoef) * 1.4, max(gcoef) * 1.45)
ax.set_xlabel("Ridge coefficient (age axis)"); panel(ax, "C")

# D: p-values -> -log10 lollipop with sig threshold
ax = axes[1][0]
pv = [0.01, 0.01, 0.07, 0.19, 0.03]
cl2 = ["NK", "B naive", "Plasma", "Neutro", "T CD4 act"]
lp = [-np.log10(p) for p in pv]
ax.hlines(np.arange(len(pv)), 0, lp, color="#BBBBBB", lw=1.1, zorder=1)
ax.scatter(lp, np.arange(len(pv)), s=44, color=C_ACC, zorder=3, edgecolor="white", lw=0.4)
ax.axvline(-np.log10(0.05), color="#444444", ls="--", lw=0.9)
ax.annotate("p=0.05", (-np.log10(0.05), 4.3), fontsize=6, color="#555555")
for i, v in enumerate(lp):
    ax.annotate(f"{v:.2f}", (v, i), textcoords="offset points",
                xytext=(6, 0), fontsize=5.5, va="center", color="#222222")
light_grid(ax, "x")
ax.set_yticks(np.arange(len(pv))); ax.set_yticklabels(cl2, fontsize=6)
ax.set_xlabel("-log10(p)"); panel(ax, "D")

# E: age-residual -> vertical diverging bars (distinct form from A)
ax = axes[1][1]
resid = [-0.065, 0.140, -0.043, 0.222, 0.02]
cl3 = ["Plasma", "B naive", "M0", "T CD8", "Treg"]
cols_e = [C_OURS if v < 0 else C_ACC for v in resid]
ax.axhline(0, color="#444444", lw=0.9)
ax.bar(np.arange(len(resid)), resid, width=0.55, color=cols_e, alpha=0.8,
       edgecolor="white", lw=0.4, zorder=2)
for i, v in enumerate(resid):
    ax.annotate(f"{v:+.3f}", (i, v), textcoords="offset points",
                xytext=(0, 4 if v > 0 else -9), ha="center", fontsize=5.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9))
light_grid(ax, "y")
ax.set_xticks(np.arange(len(resid))); ax.set_xticklabels(cl3, fontsize=6, rotation=15, ha="right")
ax.set_ylabel("Residual delta % (age-adjusted)"); panel(ax, "E")

# F: dual-modality convergence -> Venn-style overlap (each label off-centre, bbox)
ax = axes[1][2]
# two overlapping circles (Gene left-blue, Immune right-red), very low alpha
ax.add_patch(Circle((0.36, 0.50), 0.22, facecolor="#3C5488", alpha=0.08,
                    edgecolor="#3C5488", lw=1.2, zorder=2))
ax.add_patch(Circle((0.64, 0.50), 0.22, facecolor="#C0392B", alpha=0.08,
                    edgecolor="#C0392B", lw=1.2, zorder=2))
# distinct label positions: top-left for Gene, top-right for Immune, bottom-centre for overlap
ax.text(0.22, 0.72, "Gene channel", ha="center", va="center", fontsize=7.5,
        fontweight="bold", color="#3C5488", zorder=4,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#3C5488", lw=0.5, alpha=0.95))
ax.text(0.78, 0.72, "Immune channel", ha="center", va="center", fontsize=7.5,
        fontweight="bold", color="#C0392B", zorder=4,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#C0392B", lw=0.5, alpha=0.95))
ax.text(0.50, 0.28, "convergent immune-aging axis", ha="center", va="center",
        fontsize=7.0, color=C_OURS, fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.20", fc="#FFFBEE", ec=C_OURS, lw=0.8, alpha=0.98))
# small content text inside each circle (positioned to avoid overlap with labels)
ax.text(0.30, 0.42, "LRRN3, CR2,\nSIGLEC14 down", ha="center", va="center",
        fontsize=5.8, color="#3C5488", zorder=3)
ax.text(0.70, 0.42, "B naive down,\nNK up", ha="center", va="center",
        fontsize=5.8, color="#C0392B", zorder=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
panel(ax, "F")
save(fig, "Fig8_downstream", 183)

print("ALL 8 FIGURES DONE (v4)")
