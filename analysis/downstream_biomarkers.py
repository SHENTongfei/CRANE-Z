# -*- coding: utf-8 -*-
"""Downstream biomarker mining for CRANE-Z manuscript.
(1) 22 immune cell proportions, RLL vs RYC (pooled 5-fold train + GTEx external,
    with age-residualised check). (2) Gene-level markers from linear proxy.
(3) Top-10 candidate biomarker table for Discussion."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, ros.path.dirname(os.path.abspath(__file__)))
import common as C
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

lm22 = pd.read_csv(C.LM22_PATH, sep="\t", index_col=0)

print("=" * 72)
print("PART 1: immune cell RLL vs RYC (pooled 5-fold train) + GTEx external")
print("=" * 72)

# Pool all 5-fold train samples
train_idx, pooled = [], []
for f in range(C.N_FOLDS):
    X_tr, _, p_tr, _ = C.load_fold(f)
    pooled.append(p_tr)
tr = pd.concat(pooled)
tr = tr[~tr.index.duplicated(keep="first")]
imm_tr = C.lm22_deconvolution(tr, lm22)
long_tr = C.make_longevity_label(tr)
age_tr = tr["age"].to_numpy()

rows = []
for c in imm_tr.columns:
    ryc = imm_tr.loc[long_tr == 0, c]; rll = imm_tr.loc[long_tr == 1, c]
    u, p = stats.mannwhitneyu(rll, ryc, alternative="two-sided")
    rows.append((c, ryc.mean() * 100, rll.mean() * 100, (rll.mean() - ryc.mean()) * 100, p))
df = pd.DataFrame(rows, columns=["cell", "RYC_pct", "RLL_pct", "delta_pct", "p"])
df["fdr"] = stats.false_discovery_control(df["p"])
df = df.sort_values("delta_pct", key=abs, ascending=False)
print("Internal pooled (n=%d) 22 immune cells, RLL-RYC sorted by |delta|:" % len(tr))
print(df.head(8).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# age-residualised check (only for top delta cells): immune ~ age + group
print()
print("Age-residualised check (immune ~ [age, 1], residual RLL-RYC):")
Xa = np.column_stack([age_tr, np.ones_like(age_tr)])
imm_np = imm_tr.to_numpy()
imm_r = np.zeros_like(imm_np)
for j in range(imm_np.shape[1]):
    beta, *_ = np.linalg.lstsq(Xa, imm_np[:, j], rcond=None)
    imm_r[:, j] = imm_np[:, j] - Xa @ beta
for c in df["cell"].head(5):
    j = list(imm_tr.columns).index(c)
    d = imm_r[long_tr == 1, j].mean() - imm_r[long_tr == 0, j].mean()
    print(f"  {c:28s} residual delta={d*100:+.3f}%  (raw {df.loc[df.cell==c,'delta_pct'].values[0]:+.2f}%)")

# GTEx external blood
ext = pd.read_csv(C.EXT_BLOOD_PATH, index_col=0)
imm_e = C.lm22_deconvolution(ext, lm22)
long_e = C.make_longevity_label(ext)
print()
print("GTEx external blood (n=%d), RLL-RYC:" % len(ext))
rows_e = []
for c in imm_e.columns:
    ryc = imm_e.loc[long_e == 0, c]; rll = imm_e.loc[long_e == 1, c]
    u, p = stats.mannwhitneyu(rll, ryc, alternative="two-sided")
    rows_e.append((c, ryc.mean() * 100, rll.mean() * 100, (rll.mean() - ryc.mean()) * 100, p))
dfe = pd.DataFrame(rows_e, columns=["cell", "RYC_pct", "RLL_pct", "delta_pct", "p"])
dfe["fdr"] = stats.false_discovery_control(dfe["p"])
dfe = dfe.sort_values("delta_pct", key=abs, ascending=False)
print(dfe.head(8).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Cross-cohort direction agreement (internal vs GTEx) for top cells
print()
print("Cross-cohort direction agreement (internal vs GTEx blood):")
agree = []
for c in imm_tr.columns:
    di = df.loc[df.cell == c, "delta_pct"].values[0]
    de = dfe.loc[dfe.cell == c, "delta_pct"].values[0]
    agree.append((c, di, de, np.sign(di) == np.sign(de)))
ag = pd.DataFrame(agree, columns=["cell", "int_delta", "ext_delta", "agree"])
ag = ag[ag["agree"]].sort_values("int_delta", key=abs, ascending=False)
print(ag.head(6).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("Total cells with same direction:", ag["agree"].sum(), "/ 22")

print()
print("=" * 72)
print("PART 2: gene-level markers from linear proxy (Ridge coefficients)")
print("=" * 72)
from sklearn.linear_model import Ridge
X_tr, _, p_tr, _ = C.load_fold(0)
genes = C.select_hvg(X_tr, k=3000)
zsc = C.make_std_scaler(X_tr[genes])
r = Ridge(alpha=1.0).fit(zsc.transform(C.align_features(X_tr, genes)), p_tr["age"].to_numpy())
coef = pd.Series(r.coef_, index=genes).sort_values(key=abs, ascending=False)
print("Top 15 |Ridge coefficient| genes (age axis):")
for g, c in coef.head(15).items():
    print(f"  {g:15s} coef={c:+.3f}")

print()
print("=" * 72)
print("PART 3: Top-10 candidate biomarker table (immune, cross-cohort)")
print("=" * 72)
top = ag.head(10).copy()
top.columns = ["cell", "internal_delta_pct", "GTEx_delta_pct", "same_direction"]
print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
top.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "top_biomarkers.csv"), index=False)
print("saved output/top_biomarkers.csv")
