# -*- coding: utf-8 -*-
"""Internal 5-fold validation AA: model predicted age vs chronological, by group."""
import os, sys
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
from model_v2 import CraneZV2
import train_v2 as T

device = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
for fold in range(5):
    cfg = T.build_features(fold)
    _, X_va, _, p_va = C.load_fold(fold)
    Xq, Xs, imm, rda = cfg["fe"](X_va)
    rda_t = torch.from_numpy(rda)
    age_va = p_va["age"].to_numpy(np.float32)
    sex_va = C.encode_sex(p_va["sex"])
    ages = []
    for s in T.SEEDS:
        pt = os.path.join(C.OUT_DIR, f"fold{fold}_seed{s}.pt")
        m = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
                     ridge_residual=(0.0 if fold == 1 else 0.15),
                     n_modules=cfg["n_modules"], n_immune=22).to(device)
        m.load_state_dict(torch.load(pt, map_location=device))
        m.eval()
        with torch.no_grad():
            out = m(torch.from_numpy(Xq).float().to(device),
                    torch.from_numpy(Xs).float().to(device),
                    torch.from_numpy(sex_va).to(device),
                    torch.from_numpy(imm).to(device), rda_t.to(device))
            ages.append(out["age"].cpu().numpy())
    age_p = np.mean(ages, 0) * cfg["age_std"] + cfg["age_mean"]
    g = p_va["group"].to_numpy(dtype=str)
    for i in range(len(g)):
        rows.append({"fold": fold, "group": g[i], "chrono": age_va[i], "pred": age_p[i]})

df = pd.DataFrame(rows)
df["AA"] = df["pred"] - df["chrono"]
df.to_csv(os.path.join(C.OUT_DIR, "aa_internal_validation.csv"), index=False)
print("===== internal 5-fold validation AA =====")
for g in ["RYC", "RLL"]:
    m = df[df["group"] == g]
    t, p = stats.ttest_1samp(m["AA"], 0)
    neg = (m["AA"] < 0).mean()
    print(f"[{g}] n={len(m)}  AA={m['AA'].mean():+.2f} yr (SD {m['AA'].std():.2f})  t={t:.2f} p={p:.4f}  neg-frac={neg:.1%}")
t2, p2 = stats.ttest_ind(df[df["group"] == "RLL"]["AA"], df[df["group"] == "RYC"]["AA"])
print(f"[RLL vs RYC] t={t2:.2f} p={p2:.4f}")
