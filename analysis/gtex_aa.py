# -*- coding: utf-8 -*-
"""Compute real GTEx blood per-sample AA for the three-cohort replication panel G."""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
from model_v2 import CraneZV2
import train_v2 as T

DEVICE = "cuda"
ext = pd.read_csv(C.EXT_BLOOD_PATH, index_col=0)
age = ext["age_mid"].to_numpy(np.float32)
grp = ext["group"].to_numpy()
sex = C.encode_sex(ext["sex"])

rows = []
for fold in range(5):
    cfg = T.build_features(fold)
    Xq, Xs, imm, rda = cfg["fe"](ext)
    rda_t = torch.from_numpy(rda)
    s_t = torch.from_numpy(sex)
    preds = []
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
            preds.append((out["age"].cpu().numpy() * cfg["age_std"]) + cfg["age_mean"])
    age_ens = np.mean(preds, 0)
    for i in range(len(ext)):
        rows.append({"sample": ext.index[i], "fold": fold, "chrono": age[i],
                     "pred": age_ens[i], "AA": age_ens[i] - age[i], "group": grp[i]})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "aa_gtex_blood.csv"), index=False)
g = df.groupby("group")["AA"]
for name, sub in g:
    print(f"  {name}: AA {sub.mean():+.2f} (SD {sub.std():.2f}) n={len(sub)}")
rll = df[df.group == "RLL"]["AA"]
ryc = df[df.group == "RYC"]["AA"]
from scipy import stats
t, p = stats.ttest_ind(rll, ryc, equal_var=False)
print(f"GTEx blood RLL vs RYC: delta {rll.mean()-ryc.mean():+.2f} yr, Welch p={p:.2e}")
