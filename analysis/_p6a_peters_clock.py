# -*- coding: utf-8 -*-
"""P6a: CRANE-Z vs Peters 2015 transcriptomic clock (TRAP general predictor).

Authoritative-baseline comparison for CNS-level claim: on the same external
cohorts (GTEx blood, GTEx muscle, Israeli cohort2 array), compute
transcriptomic age with the published Peters 2015 general predictor formula
(FORMULA-GENERAL-PREDICTOR-GENE_ID.txt), then AA stratification vs CRANE-Z.

Protocol (H31 input check first):
  - Parse the formula string into {gene: coefficient} (drop intercept terms)
  - Align genes to each external cohort expression matrix
  - transcriptomic_age = sum(coef * expr[gene]) (+ formula constant if present)
  - AA = predicted age - chronological (Peters formula returns raw years)
  - Compare group separation (RLL vs RYC) Cohen's d, delta, Spearman(age,AA)
    against CRANE-Z values (gtex blood -24.4 yr, muscle -24.4, cohort2 -17.9)
Output: output/p6a_peters_vs_cranez.csv
"""
import os, sys, re
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FORMULA_PATH = "C:/Users/TS/Desktop/crane/external_test_data/clocks/trap/backend/FORMULA-SCALED-GENERAL-PREDICTOR-GENE_ID.txt"
OUT = os.path.join(C.OUT_DIR, "p6a_peters_vs_cranez.csv")


def parse_formula(path):
    """Parse '55.808884324-0.209312870169917*selection[,"CD248"]...' -> (const, {gene: coef})"""
    with open(path, encoding="utf-8") as f:
        s = f.read().strip()
    # constant = leading number (may be signed)
    m0 = re.match(r"^([+-]?\d+\.\d+)", s)
    const = float(m0.group(1)) if m0 else 0.0
    # terms: [+-]coef*selection[,"GENE"]
    terms = re.findall(r"([+-]?[\d.]+(?:[eE][+-]?\d+)?)\*selection\[\,\"(\w+)\"\]", s)
    coefs = {}
    for coef, gene in terms:
        coefs[gene] = float(coef)
    print(f"parsed {len(coefs)} genes, constant={const:.4f}")
    return const, coefs


def cohend(a, b):
    s = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (a.mean() - b.mean()) / s if s else 0.0


def main():
    print("=== P6a input check (H31) ===")
    const, coefs = parse_formula(FORMULA_PATH)
    print(f"formula genes: {len(coefs)}")

    # internal fold0 train expression to check gene overlap
    X0, _, p0, _ = C.load_fold(0)
    avail = set(X0.columns)
    overlap = set(coefs) & avail
    print(f"Peters genes in internal fold0 space: {len(overlap)}/{len(coefs)}")

    rows = []
    cohorts = [
        ("internal", X0, p0["age"], p0["group"]),
    ]
    ext_specs = [
        ("gtex_blood", C.EXT_BLOOD_PATH),
        ("gtex_muscle", C.EXT_MUSCLE_PATH),
        ("cohort2", "C:/Users/TS/Desktop/crane/external_test_data/cohort2/cohort2_validation_blood.csv"),
    ]
    for name, path in ext_specs:
        ext = pd.read_csv(path, index_col=0)
        meta = [c for c in ["age_mid", "group", "sex"] if c in ext.columns]
        expr = ext.drop(columns=meta) if meta else ext
        cohorts.append((name, expr, ext["age_mid"], ext["group"]))

    from scipy.stats import rankdata
    # internal fold0 train ages for rank calibration (same as CRANE-Z external protocol)
    _, _, p0_tr, _ = C.load_fold(0)
    internal_ages = p0_tr["age"].to_numpy(dtype=float)

    for name, expr, age, grp in cohorts:
        g_avail = set(expr.columns) & set(coefs)
        missing = set(coefs) - set(expr.columns)
        # TRAP protocol step 1: missing gene -> 0 contribution, then + const
        raw_score = np.full(len(expr), const)
        for g in g_avail:
            raw_score += coefs[g] * expr[g].to_numpy()
        # FAIR protocol (same as CRANE-Z external): rank-calibrate raw score
        # to the internal fold0 age distribution, AA = calib - chronological
        pct = rankdata(raw_score) / len(raw_score)
        calib_age = np.quantile(internal_ages, pct)
        aa = calib_age - age.to_numpy()
        mask = np.isin(grp, ["RLL", "RYC"])
        rll = aa[(grp == "RLL") & mask]
        ryc = aa[(grp == "RYC") & mask]
        d = cohend(rll, ryc)
        rho, ps = stats.spearmanr(age[mask], aa[mask])
        rows.append({"clock": "Peters2015", "cohort": name,
                     "n_rll": len(rll), "n_ryc": len(ryc),
                     "delta_aa": rll.mean() - ryc.mean(),
                     "cohen_d": d,
                     "spearman_age_aa": rho,
                     "genes_used": len(g_avail), "genes_missing": len(missing),
                     "aa_rll": rll.mean(), "aa_ryc": ryc.mean()})
        print(f"[{name:10s}] Peters(rank-cal): n={len(rll)}+{len(ryc)} d={d:+.2f} "
              f"delta={rll.mean()-ryc.mean():+.1f} spearman={rho:+.3f} "
              f"genes {len(g_avail)}/{len(coefs)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")

    # CRANE-Z reference values (from P4b unified table)
    print("\n=== CRANE-Z reference (pooled-variance d) ===")
    ref = {
        "internal": {"d": -1.18, "delta": -14.69},
        "gtex_blood": {"d": -0.89, "delta": -24.4},
        "gtex_muscle": {"d": -0.89, "delta": -24.4},
        "cohort2": {"d": -1.39, "delta": -17.89},
    }
    for _, r in df.iterrows():
        cr = ref.get(r["cohort"], {})
        print(f"  {r['cohort']:10s} Peters d={r['cohen_d']:+.2f} vs CRANE-Z d={cr.get('d', 0):+.2f} | "
              f"Peters delta={r['delta_aa']:+.1f} vs CRANE-Z {cr.get('delta', 0):+.1f}")


if __name__ == "__main__":
    main()
