# -*- coding: utf-8 -*-
"""CRANE-Z external validation on Cohort2 (GSE123696, Israel PrimeView whole blood).

Loads the 5-fold x 5-seed ensemble, predicts biological age per sample on the
ECDF-calibrated Cohort2 matrix, computes AA = pred - chrono, and tests:
  1. RLL (>=70y) vs RYC (<=50y) AA difference (Welch + Mann-Whitney + Cohen d)
  2. Spearman correlation of AA with chronological age (expected: negative,
     long-lived appear biologically younger)
  3. Per-fold ensemble consistency (direction rate across seeds)
"""
import os, sys
import numpy as np, pandas as pd, torch
from scipy import stats
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
from model_v2 import CraneZV2
import train_v2 as T

DEVICE = "cuda"
COH2 = os.path.join(os.environ.get("CRANE_COHORT2_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "external_test_data", "cohort2")), "cohort2_validation_blood.csv")

ext = pd.read_csv(COH2, index_col=0)
meta_cols = ["age_mid", "group", "sex"]
expr = ext.drop(columns=meta_cols)
age = ext["age_mid"].to_numpy(np.float32)
grp = ext["group"].to_numpy()
sex = C.encode_sex(ext["sex"])

# internal training age distribution (fold0 train) for rank calibration (same as GTEx protocol)
_, _, p0_tr, _ = C.load_fold(0)
internal_ages = p0_tr["age"].to_numpy(dtype=float)

fold_preds = []
for fold in range(5):
    cfg = T.build_features(fold)
    Xq, Xs, imm, rda = cfg["fe"](expr)
    rda_t = torch.from_numpy(rda)
    s_t = torch.from_numpy(sex)
    age_preds = []
    for sd in T.SEEDS:
        pt = os.path.join(C.OUT_DIR, f"fold{fold}_seed{sd}.pt")
        m = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
                     ridge_residual=(0.0 if fold == 1 else 0.15),
                     n_modules=cfg["n_modules"], n_immune=22).to(DEVICE)
        m.load_state_dict(torch.load(pt, map_location=DEVICE))
        m.eval()
        with torch.no_grad():
            out = m(torch.from_numpy(Xq).float().to(DEVICE),
                    torch.from_numpy(Xs).float().to(DEVICE),
                    s_t.to(DEVICE),
                    torch.from_numpy(imm).to(DEVICE),
                    rda_t.to(DEVICE))
            age_preds.append(out["age"].cpu().numpy())  # raw (z-space) per seed
    fold_preds.append(np.mean(age_preds, 0))  # 5-seed mean raw score per fold

pred_raw = np.mean(fold_preds, 0)   # 5-fold ensemble raw age score (same as GTEx protocol)
# rank-based calibration to internal age distribution (identical to _aa_age_acceleration.py)
pct = rankdata(pred_raw) / len(pred_raw)
calib_age = np.quantile(internal_ages, pct)
aa = calib_age - age

rows = []
for i in range(len(ext)):
    for f in range(5):
        rows.append({"sample": ext.index[i], "fold": f, "chrono": age[i],
                     "pred_raw": pred_raw[i], "pred_calib": calib_age[i],
                     "AA": aa[i], "group": grp[i], "sex": ext["sex"].iloc[i]})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "aa_cohort2_validation.csv"), index=False)

print("=== Cohort2 (GSE123696, Israel, n=%d) CRANE-Z AA (rank-calibrated, GTEx protocol) ===" % len(df))
print("samples per fold:", df.groupby("fold").size().tolist())

# 1) group x fold mean AA
g = df.groupby(["group", "fold"])["AA"].mean().unstack()
print("\nmean AA by group x fold:")
print(g.round(2).to_string())

# 2) pooled group difference
rll = df[df.group == "RLL"]["AA"]
ryc = df[df.group == "RYC"]["AA"]
t, p = stats.ttest_ind(rll, ryc, equal_var=False)
u, pu = stats.mannwhitneyu(rll, ryc, alternative="two-sided")
d = (rll.mean() - ryc.mean()) / np.sqrt((rll.std()**2 + ryc.std()**2) / 2)
print(f"\nRLL(n={len(rll)}) AA {rll.mean():+.2f} vs RYC(n={len(ryc)}) AA {ryc.mean():+.2f}")
print(f"delta = {rll.mean()-ryc.mean():+.2f} yr | Welch t={t:.2f} p={p:.2e} | MW p={pu:.2e} | Cohen d={d:.2f}")

# 3) direction consistency across folds
dirs = []
for f in range(5):
    rll_f = df[(df.group == "RLL") & (df.fold == f)]["AA"]
    ryc_f = df[(df.group == "RYC") & (df.fold == f)]["AA"]
    diff = rll_f.mean() - ryc_f.mean()
    dirs.append(diff < 0)
print("negative-AA direction per fold:", dirs, "| consistent:", sum(dirs), "/5")

# 4) AA vs age correlation (all samples, sign: younger-looking with older age)
rho, ps = stats.spearmanr(df["chrono"], df["AA"])
print(f"\nSpearman(age, AA) rho={rho:.3f} p={ps:.2e} (negative = long-lived look biologically younger)")

# 5) per-sample AA on long-lived only (is it < 0?)
rll_aa = df[df.group == "RLL"]["AA"]
t1, p1 = stats.ttest_1samp(rll_aa, 0)
print(f"\nRLL AA vs 0: mean={rll_aa.mean():+.2f} t={t1:.2f} p={p1:.2e}")
# 6) print per-sample means for panel G
psm = df.groupby(["sample", "group"])["AA"].mean().reset_index()
print("\npanel G means (per sample, ensemble):")
print(psm.groupby("group")["AA"].mean().round(2).to_string())
