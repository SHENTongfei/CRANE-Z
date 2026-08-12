# -*- coding: utf-8 -*-
"""AA (age acceleration) analysis on external GTEx blood, rank-calibrated.

Raw model age output is not in absolute years on external data (ridge prior
z-score drifts under distribution shift), so we calibrate rank-to-quantile:
  calibrated_age = internal-age quantile at the sample's predicted rank.
AA = calibrated predicted biological age - chronological age.
Prediction: RLL (long-lived) shows NEGATIVE AA (biologically younger).
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
from model_v2 import CraneZV2
import train_v2 as T

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[aa] device={device}")

# internal training age distribution (fold0 train) for calibration
_, _, p0_tr, _ = C.load_fold(0)
internal_ages = p0_tr["age"].to_numpy(dtype=float)
print(f"[aa] internal age range {internal_ages.min():.0f}-{internal_ages.max():.0f}")

extd = pd.read_csv(C.EXT_BLOOD_PATH, index_col=0)
n = len(extd)

fold_ages = []
for fold in range(5):
    cfg = T.build_features(fold)
    Xq_e, Xs_e, imm_e, rda_e = cfg["fe"](extd)
    rda_t = torch.from_numpy(rda_e)
    sex_e = C.encode_sex(extd["sex"])
    xe = torch.from_numpy(Xq_e).float().to(device)
    xes = torch.from_numpy(Xs_e).float().to(device)
    se = torch.from_numpy(sex_e).to(device)
    ie = torch.from_numpy(imm_e).to(device)
    ages_e = []
    for s in T.SEEDS:
        pt = os.path.join(C.OUT_DIR, f"fold{fold}_seed{s}.pt")
        m = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
                     ridge_residual=(0.0 if fold == 1 else 0.15),
                     n_modules=cfg["n_modules"], n_immune=22).to(device)
        m.load_state_dict(torch.load(pt, map_location=device))
        m.eval()
        with torch.no_grad():
            out = m(xe, xes, se, ie, rda_t.to(device))
            ages_e.append(out["age"].cpu().numpy())
    fold_ages.append(np.mean(ages_e, 0))  # 5-seed mean raw predicted age per fold
    print(f"[aa] fold{fold} done")

pred_raw = np.mean(fold_ages, 0)            # 5-fold ensemble raw age score
chrono = extd["age_mid"].to_numpy(dtype=float)
group = extd["group"].to_numpy(dtype=str)

# rank-based calibration to internal age distribution
pct = rankdata(pred_raw) / len(pred_raw)    # 0..1 percentile of each sample
calib_age = np.quantile(internal_ages, pct)  # map to internal age range
aa = calib_age - chrono

out = pd.DataFrame({
    "sample": extd.index, "group": group,
    "chronological_age": chrono,
    "predicted_bio_age_raw": pred_raw,
    "predicted_bio_age_calibrated": calib_age,
    "age_acceleration": aa,
})
out.to_csv(os.path.join(C.OUT_DIR, "aa_age_acceleration.csv"), index=False)

print("\n===== AA (age acceleration) results: external GTEx blood (rank-calibrated) =====")
for g in ["RYC", "RLL"]:
    m = out[out["group"] == g]
    t, p = stats.ttest_1samp(m["age_acceleration"], 0)
    neg_frac = (m["age_acceleration"] < 0).mean()
    print(f"[{g}] n={len(m)}  AA mean={m['age_acceleration'].mean():+.2f} yr "
          f"(SD {m['age_acceleration'].std():.2f})  t={t:.2f} p={p:.3f}  "
          f"negative-fraction={neg_frac:.2%}")
t2, p2 = stats.ttest_ind(out[out["group"] == "RLL"]["age_acceleration"],
                         out[out["group"] == "RYC"]["age_acceleration"])
rll = out[out["group"] == "RLL"]
print(f"\n[RLL vs RYC] AA difference t={t2:.2f} p={p2:.3f}")
print(f"[RLL] calibrated bio age mean={rll['predicted_bio_age_calibrated'].mean():.1f} yr "
      f"vs chronological mean={rll['chronological_age'].mean():.1f} yr "
      f"(delta {rll['predicted_bio_age_calibrated'].mean()-rll['chronological_age'].mean():+.1f} yr)")
print("\nSaved -> output/aa_age_acceleration.csv")
