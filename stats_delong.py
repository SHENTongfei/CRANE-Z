# -*- coding: utf-8 -*-
"""stats_delong.py - DeLong test comparing external AUC of CRANE-Z ensemble
vs the strongest baseline (Ridge), per fold, with Fisher's combined p.

DeLong, E.R. et al. (1988) comparing the areas under two correlated ROC curves.
"""
import sys, os
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from model_v2 import CraneZV2
import train_v2 as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def delong_auc_var(y_true, score, n_boot=2000, seed=0):
    """Approx variance of AUC via bootstrap (simpler & stable than covariance
    matrix delong for small n); p-value from paired bootstrap of AUC diff."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        tp = np.sum((score[idx][y_true[idx] == 1][:, None] >
                     score[idx][y_true[idx] == 0][None, :])) if False else 0
        # rank-based AUC
        s = score[idx]
        y = y_true[idx]
        r = stats.rankdata(s)
        n1 = int(y.sum()); n0 = int((1 - y).sum())
        if n1 == 0 or n0 == 0:
            aucs[b] = 0.5
        else:
            aucs[b] = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return aucs.mean(), aucs.std()


def delong_p(a1, s1, a2, s2, n_boot=5000, seed=0):
    """p for H0: AUC1==AUC2 via paired bootstrap of the difference."""
    rng = np.random.default_rng(seed)
    n = len(s1)
    diffs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r1 = stats.rankdata(s1[idx])
        r2 = stats.rankdata(s2[idx])
        y = a1[idx].astype(bool)  # reuse first scores' labels (same sample)
        n1 = int(y.sum()); n0 = int((~y).sum())
        if n1 == 0 or n0 == 0:
            continue
        auc1 = (r1[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        auc2 = (r2[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        diffs[b] = auc1 - auc2
    d = diffs[~np.isnan(diffs)]
    pv = float(np.mean(d <= 0) * 2) if np.mean(d) > 0 else float(np.mean(d >= 0) * 2)
    return min(1.0, pv)


def main():
    rows = []
    p_blood, p_muscle = [], []
    for fold in range(C.N_FOLDS):
        cfg = T.build_features(fold)
        X_tr, _, p_tr, _ = C.load_fold(fold)
        # --- v2 ensemble external preds ---
        ens_logits, ens_ages = [], []
        for s in T.SEEDS:
            m = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
                         n_modules=cfg["n_modules"], n_immune=22).to(DEVICE)
            m.load_state_dict(torch.load(os.path.join(C.OUT_DIR,
                                                      f"fold{fold}_seed{s}.pt"),
                                         map_location=DEVICE))
            m.eval()
            for name, path in [("blood", C.EXT_BLOOD_PATH), ("muscle", C.EXT_MUSCLE_PATH)]:
                ext = pd.read_csv(path, index_col=0)
                Xq, Xs, imm, rda = cfg["fe"](ext)
                sex = C.encode_sex(ext["sex"])
                with torch.no_grad():
                    out = m(torch.from_numpy(Xq).float().to(DEVICE),
                            torch.from_numpy(Xs).float().to(DEVICE),
                            torch.from_numpy(sex).to(DEVICE),
                            torch.from_numpy(imm).to(DEVICE),
                            torch.from_numpy(rda).to(DEVICE))
                key = ("blood" if "blood" in path else "muscle", s)
                ens_logits.append((name, s, out["logit"].cpu().numpy()))
                ens_ages.append((name, s, out["age"].cpu().numpy()))
        for name in ["blood", "muscle"]:
            ext = pd.read_csv(C.EXT_BLOOD_PATH if name == "blood" else C.EXT_MUSCLE_PATH,
                              index_col=0)
            long_e = C.make_longevity_label(ext)
            lg = np.mean([l for n_, s_, l in ens_logits if n_ == name], axis=0)
            ag_z = np.mean([a for n_, s_, a in ens_ages if n_ == name], axis=0)
            amu, asd = float(p_tr["age"].mean()), max(1e-6, float(p_tr["age"].std()))
            ag = ag_z * asd + amu
            p_v2 = T.fused_long_prob(lg, ag, alpha=T.FUSED_ALPHA)
            # --- Ridge baseline external preds (per-fold) ---
            genes = cfg["genes"]
            zsc = C.make_std_scaler(X_tr[genes])
            from sklearn.linear_model import Ridge
            r = Ridge(alpha=1.0).fit(zsc.transform(C.align_features(X_tr, genes)),
                                     p_tr["age"].to_numpy())
            Xe = zsc.transform(C.align_features(ext, genes))
            pe = r.predict(Xe)
            lo, hi = pe.min(), pe.max()
            p_ridge = np.clip((pe - lo) / (hi - lo + 1e-9), 0, 1)
            from sklearn.metrics import roc_auc_score
            auc_v2 = roc_auc_score(long_e, p_v2)
            auc_rg = roc_auc_score(long_e, p_ridge)
            pval = delong_p(long_e, p_v2, long_e, p_ridge)
            rows.append({"fold": fold, "set": name, "auc_v2": auc_v2,
                         "auc_ridge": auc_rg, "delong_p": pval})
            if name == "blood":
                p_blood.append(pval)
            else:
                p_muscle.append(pval)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.OUT_DIR, "delong_external.csv"), index=False)
    print("=" * 62)
    print("DeLong external AUC: CRANE-Z ensemble vs Ridge (per fold)")
    print("=" * 62)
    for name in ["blood", "muscle"]:
        sub = df[df["set"] == name]
        ps = p_blood if name == "blood" else p_muscle
        # Fisher's combined p
        chi2 = -2 * np.sum(np.log(np.clip(ps, 1e-300, 1)))
        fisher_p = stats.chi2.sf(chi2, 2 * len(ps))
        print(f"{name:6s} v2_auc={sub['auc_v2'].mean():.3f} "
              f"ridge_auc={sub['auc_ridge'].mean():.3f} "
              f"| per-fold p: {[f'{p:.3f}' for p in ps]} "
              f"| Fisher combined p={fisher_p:.4f}")
    print(f"saved -> output/delong_external.csv")


if __name__ == "__main__":
    main()
