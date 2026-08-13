# -*- coding: utf-8 -*-
"""P5b: age-bucket error structure of the internal 5-fold OOF predictions.

Shows where CRANE-Z predicts reliably: MAE/bias by age bucket, and the
age distribution of negative AA in long-lived (deceleration location).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.OUT_DIR, "age_bucket_errors.csv")


def main():
    df = pd.read_csv(os.path.join(C.OUT_DIR, "aa_internal_validation.csv"))
    df = df[df.group.isin(["RLL", "RYC"])].copy()
    df["err"] = df["pred"] - df["chrono"]       # signed error
    df["abs_err"] = df["err"].abs()

    print("=== P5b: age-bucket error structure (internal 5-fold OOF) ===")
    buckets = [(0, 50, "<50"), (50, 70, "50-70"), (70, 200, ">=70")]
    rows = []
    for lo, hi, name in buckets:
        b = df[(df.chrono >= lo) & (df.chrono < hi)]
        if len(b) == 0:
            continue
        rll_b = b[b.group == "RLL"]
        ryc_b = b[b.group == "RYC"]
        rows.append({
            "bucket": name, "n": len(b), "n_rll": len(rll_b), "n_ryc": len(ryc_b),
            "mae": b.abs_err.mean(), "mae_sd": b.abs_err.std(),
            "bias": b.err.mean(),
            "rll_mae": rll_b.abs_err.mean() if len(rll_b) else np.nan,
            "rll_neg_frac": (rll_b.chrono - rll_b.pred > 0).mean() if len(rll_b) else np.nan,
        })
        print(f"[{name:6s}] n={len(b):3d} MAE={b.abs_err.mean():5.2f} "
              f"bias={b.err.mean():+5.2f} | RLL n={len(rll_b):3d} "
              f"RLL_MAE={rll_b.abs_err.mean():5.2f} RLL_negAA={(rll_b.chrono - rll_b.pred > 0).mean():.2%}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")

    # deceleration location: RLL AA distribution
    rll = df[df.group == "RLL"]
    neg = rll[rll.AA < 0]
    print(f"\nRLL total {len(rll)}, negative AA {len(neg)} ({len(neg)/len(rll):.1%})")
    print(f"negative-AA RLL age: mean {neg.chrono.mean():.1f} (range {neg.chrono.min():.0f}-{neg.chrono.max():.0f})")
    print(f"positive-AA RLL age: mean {rll[rll.AA>=0].chrono.mean():.1f}")
    # correlation of AA magnitude with age within RLL
    rho, p = stats.spearmanr(rll.chrono, rll.AA)
    print(f"within-RLL Spearman(age, AA)={rho:+.3f} p={p:.2e}")

    # global age correlation
    rho2, p2 = stats.spearmanr(df.chrono, df.pred)
    print(f"global Spearman(chrono, pred)={rho2:.3f} p={p2:.2e}")


if __name__ == "__main__":
    main()
