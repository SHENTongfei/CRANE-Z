# -*- coding: utf-8 -*-
"""CRANE-Z external validation on Cohort2 (GSE123696, Israel PrimeView whole blood).

Paper protocol (v27+): the cohort2 matrix is per-gene ECDF-calibrated onto the
training-fold0 TPM distribution (see data_prep/prep_cohort2.py). The frozen
5-fold x 5-seed ensemble predicts biological age DIRECTLY on this calibrated
matrix (raw-year, denormalized by fold0 age_mean/age_std). AA = pred - chrono.
No second rank calibration (that was a regression that compressed the effect).
Tests: RLL vs RYC AA difference, Spearman(age, AA), per-fold direction.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.join("C:/Users/TS/Desktop/crane", "Code_v2"))
import common as C
from model_v2 import CraneZV2
import train_v2 as T

DEVICE = "cuda"
COH2 = "C:/Users/TS/Desktop/crane/external_test_data/cohort2/cohort2_validation_blood.csv"

ext = pd.read_csv(COH2, index_col=0)
meta_cols = ["age_mid", "group", "sex"]
expr = ext.drop(columns=meta_cols)
age = ext["age_mid"].to_numpy(np.float32)
grp = ext["group"].to_numpy()
sex = C.encode_sex(ext["sex"])

# fold0 age stats for denormalization (paper protocol: fold0 reference)
_, _, p0_tr, _ = C.load_fold(0)
age_mean = float(p0_tr["age"].mean())
age_std = max(1e-6, float(p0_tr["age"].std()))
print(f"[cohort2] fold0 age mean={age_mean:.1f} std={age_std:.1f}")

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

pred_raw = np.mean(fold_preds, 0)          # 5-fold ensemble raw age score (z-space)
bio_age = pred_raw * age_std + age_mean     # raw-year denormalization (fold0 stats)
aa = bio_age - age                          # age acceleration in years

rows = []
for i in range(len(ext)):
    for f in range(5):
        rows.append({"sample": ext.index[i], "fold": f, "chrono": age[i],
                     "pred_raw": pred_raw[i], "pred_calib": bio_age[i],
                     "AA": aa[i], "group": grp[i], "sex": ext["sex"].iloc[i]})

df = pd.DataFrame(rows)
df.to_csv("C:/Users/TS/Desktop/crane/Code_v2/output/aa_cohort2_validation.csv", index=False)

print("\n=== Cohort2 (GSE123696, Israel, n=%d) CRANE-Z AA (raw-year ECDF protocol) ===" % len(ext))
for g in ["RYC", "RLL"]:
    m = df[df.group == g]
    neg = (m.AA < 0).mean()
    print(f"[{g}] n={len(m['sample'].unique())} AA mean={m.AA.mean():+.2f} yr (SD {m.AA.std():.2f}) "
          f"negative-fraction={neg:.2%}")

rll = df[df.group == "RLL"]["AA"].to_numpy()
ryc = df[df.group == "RYC"]["AA"].to_numpy()
# sample-level (unique samples)
s = df.groupby("sample").agg(AA=("AA", "mean"), grp=("group", "first"))
rll_s = s[s.grp == "RLL"]["AA"].to_numpy()
ryc_s = s[s.grp == "RYC"]["AA"].to_numpy()
t, p = stats.ttest_ind(rll_s, ryc_s, equal_var=False)
u, pu = stats.mannwhitneyu(rll_s, ryc_s)
d = (rll_s.mean() - ryc_s.mean()) / np.sqrt((rll_s.std()**2 + ryc_s.std()**2) / 2)
print(f"\n[RLL vs RYC, sample-level n={len(rll_s)}+{len(ryc_s)}]")
print(f"delta = {rll_s.mean()-ryc_s.mean():+.2f} yr | Welch t={t:.2f} p={p:.2e} "
      f"| MW p={pu:.2e} | Cohen d={d:.2f}")
print(f"RLL AA mean={rll_s.mean():+.2f} | RYC AA mean={ryc_s.mean():+.2f}")

# Spearman(age, AA) across all samples
rho, ps = stats.spearmanr(age, s["AA"].to_numpy())
print(f"Spearman(age, AA) rho={rho:.3f} p={ps:.2e} (negative = long-lived look younger)")

# per-fold direction
dirs = []
for f in range(5):
    df_f = df[df.fold == f]
    r_f = df_f[df_f.group == "RLL"]["AA"].mean()
    y_f = df_f[df_f.group == "RYC"]["AA"].mean()
    dirs.append(r_f < y_f)
print(f"per-fold direction (RLL<RYC): {sum(dirs)}/5 {dirs}")
print(f"\nSaved -> output/aa_cohort2_validation.csv")
