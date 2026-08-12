# -*- coding: utf-8 -*-
"""stats_bootstrap.py - bootstrap 95% CI for internal & external OPS/AUC.

Uses the trained per-fold 5-seed ensemble weights (fold{f}_seed{s}.pt):
- internal: bootstrap over val samples (per fold, 1000 iters) -> OPS CI
- external blood/muscle: bootstrap over GTEx samples -> OPS + AUC CI
Reports per-fold means and 5-fold mean (with between-fold spread).

Usage: py311 python stats_bootstrap.py [--n 1000]
"""
import sys, os, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from model_v2 import CraneZV2
import train_v2 as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ensemble_predictions(fold, Xq, Xs, imm, rda, sex):
    """Average 5-seed predictions -> (age_real, logit)."""
    ages, logits = [], []
    for s in T.SEEDS:
        pt = os.path.join(C.OUT_DIR, f"fold{fold}_seed{s}.pt")
        m = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
                     n_modules=cfg["n_modules"], n_immune=22).to(DEVICE)
        m.load_state_dict(torch.load(pt, map_location=DEVICE))
        m.eval()
        with torch.no_grad():
            out = m(torch.from_numpy(Xq).float().to(DEVICE),
                    torch.from_numpy(Xs).float().to(DEVICE),
                    torch.from_numpy(sex).to(DEVICE),
                    torch.from_numpy(imm).to(DEVICE),
                    torch.from_numpy(rda).to(DEVICE))
            ages.append(out["age"].cpu().numpy())
            logits.append(out["logit"].cpu().numpy())
    return np.mean(ages, 0), np.mean(logits, 0)


def boot_ci(scores_y, scores_p, metric, n_iter, rng, alpha=0.5,
            age_true=None, age_pred=None, long_true=None):
    """bootstrap 95% CI of an OPS-type metric over samples."""
    vals = []
    n = len(scores_y)
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        if metric == "internal_ops":
            lp = T.fused_long_prob(scores_p[idx], age_pred[idx], alpha=alpha)
            ops, _ = C.compute_ops_internal(age_true[idx], age_pred[idx],
                                            long_true[idx], lp)
        elif metric == "external_ops":
            lp = T.fused_long_prob(scores_p[idx], age_pred[idx], alpha=alpha)
            ops, _ = C.compute_ops_external(scores_y[idx], lp)
        else:  # auc
            from sklearn.metrics import roc_auc_score
            ops = roc_auc_score(scores_y[idx], scores_p[idx])
        vals.append(ops)
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=T.FUSED_ALPHA)
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    global cfg

    rows = []
    for fold in range(C.N_FOLDS):
        cfg = T.build_features(fold)
        X_tr, X_va, p_tr, p_va = C.load_fold(fold)
        amu, asd = float(p_tr["age"].mean()), max(1e-6, float(p_tr["age"].std()))
        Xq_va, Xs_va, imm_va, rda_va = cfg["fe"](X_va)
        age_va = p_va["age"].to_numpy(np.float32)
        long_va = C.make_longevity_label(p_va)
        sex_va = C.encode_sex(p_va["sex"])
        age_z, logit = load_ensemble_predictions(fold, Xq_va, Xs_va, imm_va,
                                                 rda_va, sex_va)
        age_p = age_z * asd + amu
        lo, hi, mean = boot_ci(long_va, logit, "internal_ops", args.n, rng,
                               args.alpha, age_va, age_p, long_va)
        rows.append({"fold": fold, "metric": "internal_ops", "ci_lo": lo,
                     "mean": mean, "ci_hi": hi})

        for name, path in [("blood", C.EXT_BLOOD_PATH), ("muscle", C.EXT_MUSCLE_PATH)]:
            ext = pd.read_csv(path, index_col=0)
            Xq_e, Xs_e, imm_e, rda_e = cfg["fe"](ext)
            long_e = C.make_longevity_label(ext)
            sex_e = C.encode_sex(ext["sex"])
            age_z_e, logit_e = load_ensemble_predictions(fold, Xq_e, Xs_e, imm_e,
                                                         rda_e, sex_e)
            age_p_e = age_z_e * asd + amu
            lo, hi, mean = boot_ci(long_e, logit_e, "external_ops", args.n, rng,
                                   args.alpha, None, age_p_e, None)
            rows.append({"fold": fold, "metric": f"external_{name}",
                         "ci_lo": lo, "mean": mean, "ci_hi": hi})

    df = pd.DataFrame(rows)
    out = os.path.join(C.OUT_DIR, "bootstrap_ci.csv")
    df.to_csv(out, index=False)
    print("=" * 62)
    print(f"bootstrap 95% CI (n_iter={args.n}, alpha={args.alpha:.2f})")
    print("=" * 62)
    for metric in ["internal_ops", "external_blood", "external_muscle"]:
        sub = df[df["metric"] == metric]
        m = sub["mean"].mean()
        lo = sub["ci_lo"].mean()
        hi = sub["ci_hi"].mean()
        print(f"{metric:16s} 5-fold mean {m:.4f}  CI [{lo:.4f}, {hi:.4f}]")
        for _, r in sub.iterrows():
            print(f"   fold{int(r['fold'])}: {r['mean']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
