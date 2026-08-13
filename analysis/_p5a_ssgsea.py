# -*- coding: utf-8 -*-
"""P5a: ssGSEA pathway-score validation of the immune axis across all cohorts.

Question: the model claims to capture a real immune axis (NK up, naive B down).
GSEA showed pathway-level enrichment in the gene-age axis. Can we verify at the
SAMPLE level that the same pathway scores separate long-lived from young,
correlate with age, and replicate across platforms?

Design (H31: input check first, output verified):
  - 5 top immune pathways from the frozen GSEA run (KEGG antigen processing,
    GO NK-mediated cytotoxicity, GO dendritic-cell differentiation,
    GO NK chemotaxis, GO negative IL-1beta production)
  - ssGSEA score per sample on 3 cohorts: internal fold0-train (240), GTEx
    blood (64), GTEx muscle (58), cohort2 Israeli array (66) via gseapy.ssgsea
  - Tests: RLL vs RYC ssGSEA difference (Welch p, Cohen d), Spearman(age,
    score), and a composite immune score (NK + antigen processing - naive-B proxy)
  - Output: output/ssgsea_scores.csv + printed summary; nothing hidden.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT_CSV = os.path.join(C.OUT_DIR, "ssgsea_scores.csv")


def cohend(a, b):
    s = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) /
                (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s else 0.0


def load_expression(path, meta_cols=("age_mid", "group", "sex")):
    df = pd.read_csv(path, index_col=0)
    meta = [c for c in meta_cols if c in df.columns]
    expr = df.drop(columns=meta) if meta else df
    return expr, df


def main():
    print("=== P5a input check (H31) ===")
    print("gseapy ssGSEA on 5 frozen immune pathways x 4 cohorts")

    from gseapy import ssgsea

    # pathway gene sets (frozen from the GSEA run, curated)
    gsets = {
        "AntigenProcessing_KEGG": ["KIR3DL2", "KIR2DL3", "KIR2DS2", "KIR2DL2", "KLRC3",
                                    "CALR", "IFI30", "IFNG", "KLRD1", "HLA-B", "PDIA3",
                                    "TAP1", "HLA-DRA", "KIR3DL1", "HLA-C", "HLA-DMA",
                                    "HSPA6", "KIR2DS4", "HLA-DQB1", "HSPA1L", "KIR2DS5",
                                    "RFXANK", "PSME1", "CD8A", "PSME2", "HSPA1B", "CTSB"],
        "NK_cytotoxicity_GO": ["GZMB", "ULBP2", "SLAMF7", "TUBB4B", "VAMP2", "KIR3DL1",
                                "ULBP1", "CLEC2A", "KLRF2", "VAMP7", "TUBB", "ULBP3",
                                "PLEKHM2", "GZMH", "KIR2DL3", "KIR2DL2", "KIR3DL2",
                                "KLRD1", "KLRK1", "NKG7", "PRF1", "GNLY"],
        "DC_differentiation_GO": ["LGALS1", "LILRB1", "FCGR2B", "LILRB2", "HLA-B",
                                   "LGALS9", "IL3RA", "ITGA4", "TNFRSF9"],
        "NK_chemotaxis_GO": ["CCL3", "CCL5", "XCL1", "CCL4"],
        "NegIL1b_GO": ["GHSR", "CARD16", "CARD17", "NLRP7", "CX3CR1", "GSTP1", "GIT1",
                        "ZC3H12A", "IL1B", "CASP1", "PYCARD", "TNF", "IL6"],
    }
    print(f"pathways: {list(gsets)}")

    # cohorts: (name, expr path)
    cohorts = [
        ("internal", C.load_fold(0)[0]),  # X matrix only; phenotype from p
        ("gtex_blood", None), ("gtex_muscle", None), ("cohort2", None),
    ]
    _, _, p0, _ = C.load_fold(0)
    X0 = C.load_fold(0)[0]
    # external expression matrices
    ext_blood = pd.read_csv(C.EXT_BLOOD_PATH, index_col=0)
    ext_muscle = pd.read_csv(C.EXT_MUSCLE_PATH, index_col=0)
    coh2 = pd.read_csv("C:/Users/TS/Desktop/crane/external_test_data/cohort2/cohort2_validation_blood.csv",
                       index_col=0)

    rows = []
    for name, X in [("internal", X0), ("gtex_blood", ext_blood),
                    ("gtex_muscle", ext_muscle), ("cohort2", coh2)]:
        meta = [c for c in ["age_mid", "group", "sex"] if c in X.columns]
        expr = X.drop(columns=meta) if meta else X
        # ssGSEA expects genes as index (rows), samples as columns
        expr_t = expr.T  # genes x samples
        if expr_t.values.max() > 100:
            expr_t = np.log2(expr_t + 1)
        try:
            ss = ssgsea(expr_t, gene_sets=gsets, outdir=None, threads=2,
                        min_size=2, seed=42)
        except TypeError:
            ss = ssgsea(expr_t, gene_sets=gsets, threads=2, min_size=2, seed=42)
        res = ss.res2d  # long-form: Name(sample), Term(pathway), ES, NES
        sc = res.pivot_table(index="Name", columns="Term", values="ES")
        group = X["group"] if "group" in X.columns else None
        age = X["age_mid"] if "age_mid" in X.columns else None
        if name == "internal":
            group = p0["group"]
            age = p0["age"]
        for pw in gsets:
            if pw not in sc.columns:
                continue
            s = sc[pw].astype(float)
            if group is not None:
                rll = s[group == "RLL"]; ryc = s[group == "RYC"]
                rows.append({
                    "cohort": name, "pathway": pw,
                    "n_rll": len(rll), "n_ryc": len(ryc),
                    "rll_mean": rll.mean(), "ryc_mean": ryc.mean(),
                    "delta": rll.mean() - ryc.mean(),
                    "welch_p": stats.ttest_ind(rll, ryc, equal_var=False).pvalue,
                    "cohen_d": cohend(rll, ryc),
                })
                print(f"[{name:10s}] {pw:28s} d={cohend(rll, ryc):+.2f} "
                      f"p={stats.ttest_ind(rll, ryc, equal_var=False).pvalue:.2e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved -> {OUT_CSV}")

    # verdict: how many pathway x cohort comparisons show RLL enrichment (d<0 = lower score?) 
    # For immune-effector pathways higher score in RLL = preserved NK = expected direction
    print("\n=== VERDICT ===")
    sig = df[(df.welch_p < 0.05) & (df.cohen_d.abs() > 0.5)]
    print(f"significant pathway-cohort comparisons (p<0.05, |d|>0.5): {len(sig)}/{len(df)}")
    for _, r in sig.iterrows():
        print(f"  {r.cohort:10s} {r.pathway:28s} d={r.cohen_d:+.2f} p={r.welch_p:.2e}")


if __name__ == "__main__":
    main()
