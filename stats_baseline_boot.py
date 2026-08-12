# -*- coding: utf-8 -*-
"""baseline bootstrap CI for internal/external (same protocol as v2)."""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, ros.path.dirname(os.path.abspath(__file__)))
import common as C

from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge, Lasso

rng = np.random.default_rng(7)


def boot_ops(long_y, prob, n_iter=1000, age_y=None, age_p=None):
    vals = []
    n = len(long_y)
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        if age_y is not None:  # internal
            ops, _ = C.compute_ops_internal(age_y[idx], age_p[idx], long_y[idx], prob[idx])
        else:
            ops, _ = C.compute_ops_external(long_y[idx], prob[idx])
        vals.append(ops)
    v = np.array(vals)
    return np.percentile(v, 2.5), v.mean(), np.percentile(v, 97.5)


models = [
    ("CLC-S", lambda: LinearSVR(C=1.0, max_iter=10000, random_state=0)),
    ("Ridge", lambda: Ridge(alpha=1.0)),
    ("Lasso", lambda: Lasso(alpha=0.001, max_iter=10000)),
]
rows = []
for name, mf in models:
    int_ops_lo, int_ops_m, int_ops_hi = [], [], []
    ext_lo, ext_m, ext_hi = {}, {}, {}
    for f in range(C.N_FOLDS):
        X_tr, X_va, p_tr, p_va = C.load_fold(f)
        genes = C.select_hvg(X_tr, k=3000)
        zsc = C.make_std_scaler(X_tr[genes])
        Xtr = zsc.transform(C.align_features(X_tr, genes))
        Xva = zsc.transform(C.align_features(X_va, genes))
        mdl = mf()
        mdl.fit(Xtr, p_tr["age"].to_numpy())
        pv = mdl.predict(Xva)
        lo, hi = pv.min(), pv.max()
        lp = np.clip((pv - lo) / (hi - lo + 1e-9), 0, 1)
        long_va = C.make_longevity_label(p_va)
        l, m, h = boot_ops(long_va, lp, age_y=p_va["age"].to_numpy(), age_p=pv)
        int_ops_lo.append(l); int_ops_m.append(m); int_ops_hi.append(h)
        for en, path in [("blood", C.EXT_BLOOD_PATH), ("muscle", C.EXT_MUSCLE_PATH)]:
            ext = pd.read_csv(path, index_col=0)
            Xe = zsc.transform(C.align_features(ext, genes))
            pe = mdl.predict(Xe)
            le, he = pe.min(), pe.max()
            lp_e = np.clip((pe - le) / (he - le + 1e-9), 0, 1)
            l2, m2, h2 = boot_ops(C.make_longevity_label(ext), lp_e)
            ext_lo.setdefault(en, []).append(l2)
            ext_m.setdefault(en, []).append(m2)
            ext_hi.setdefault(en, []).append(h2)
    print(f"{name:6s} internal {np.mean(int_ops_m):.3f} [{np.mean(int_ops_lo):.3f}, {np.mean(int_ops_hi):.3f}] | "
          f"blood {np.mean(ext_m['blood']):.3f} [{np.mean(ext_lo['blood']):.3f}, {np.mean(ext_hi['blood']):.3f}] | "
          f"muscle {np.mean(ext_m['muscle']):.3f} [{np.mean(ext_lo['muscle']):.3f}, {np.mean(ext_hi['muscle']):.3f}]")
