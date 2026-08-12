# -*- coding: utf-8 -*-
"""Sex-stratified aging acceleration (AA) on internal CV folds."""
import os, sys
import numpy as np, pandas as pd, torch
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
from model_v2 import CraneZV2
import train_v2 as T

device = "cuda"
rows = []
for fold in range(5):
    cfg = T.build_features(fold)
    _, X_va, _, p_va = C.load_fold(fold)
    Xq, Xs, imm, rda = cfg["fe"](X_va)
    age_va = p_va["age"].to_numpy(np.float32)
    sex_va = C.encode_sex(p_va["sex"])
    g = p_va["group"].to_numpy(dtype=str)
    ages = []
    for s in T.SEEDS:
        pt = os.path.join(C.OUT_DIR, f"fold{fold}_seed{s}.pt")
        m = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
                     ridge_residual=(0.0 if fold == 1 else 0.15),
                     n_modules=cfg["n_modules"], n_immune=22).to(device)
        m.load_state_dict(torch.load(pt, map_location=device)); m.eval()
        with torch.no_grad():
            out = m(torch.from_numpy(Xq).float().to(device), torch.from_numpy(Xs).float().to(device),
                    torch.from_numpy(sex_va).to(device), torch.from_numpy(imm).to(device),
                    torch.from_numpy(rda).to(device))
            ages.append(out["age"].cpu().numpy())
    age_p = np.mean(ages, 0) * cfg["age_std"] + cfg["age_mean"]
    for i in range(len(g)):
        rows.append({"sample": p_va.index[i], "fold": fold, "group": g[i],
                     "sex": p_va["sex"].iloc[i], "chrono": age_va[i], "pred": age_p[i]})

df = pd.DataFrame(rows)
df["AA"] = df["pred"] - df["chrono"]
df.to_csv("output/aa_sex_internal.csv", index=False)

# ensure sex codes align with phenotype.csv (F/M)
ph = pd.read_csv(os.path.join(os.environ.get("CRANE_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")), "phenotype.csv"), index_col=0)
df = df.join(ph[["sex"]], on="sample", rsuffix="_ph")
# unify: keep phenotype sex if present else own
df["sex_u"] = df["sex_ph"].fillna(df["sex"])

print("n =", len(df), "| per group×sex:")
res = df.groupby(["group", "sex_u"])["AA"].agg(["count", "mean", "std"]).round(2)
print(res)

def t(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p, pu

print("\n--- sex difference within group ---")
for g in ["RYC", "RLL"]:
    m = df[(df["group"] == g) & (df["sex_u"] == "M")]["AA"]
    f = df[(df["group"] == g) & (df["sex_u"] == "F")]["AA"]
    p, pu = t(m, f)
    print(f"[{g}] M: n={len(m)} AA={m.mean():+.2f}±{m.std():.2f} | F: n={len(f)} AA={f.mean():+.2f}±{f.std():.2f} | Welch p={p:.4f} MW p={pu:.4f}")

print("\n--- overall sex difference ---")
m_all = df[df["sex_u"] == "M"]["AA"]; f_all = df[df["sex_u"] == "F"]["AA"]
p, pu = t(m_all, f_all)
print(f"[All] M: n={len(m_all)} AA={m_all.mean():+.2f}±{m_all.std():.2f} | F: n={len(f_all)} AA={f_all.mean():+.2f}±{f_all.std():.2f} | Welch p={p:.4f} MW p={pu:.4f}")