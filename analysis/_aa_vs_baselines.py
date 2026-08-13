# -*- coding: utf-8 -*-
"""P4b: AA (biological-age acceleration) stratification quality, CRANE-Z vs ALL baselines.

Reviewer question: "Is AA stratification (d=1.18, p=1e-22) unique to CRANE-Z,
or can linear baselines produce the same from the same features?"

Protocol: for every model, on every cohort (internal OOF, GTEx blood, GTEx muscle,
GSE123696):
  - CRANE-Z: internal OOF AA (existing aa_internal_validation.csv), GTEx blood AA
    (existing aa_age_acceleration.csv rank-calibrated), muscle + cohort2 computed here
    with the SAME rank-calibration to fold0-train internal age distribution.
  - Baselines: Ridge/Lasso/CLC-S/KNN/DT fitted per protocol (fold-train, same HVG
    feature space + std scaler), raw predicted age -> rank-calibrated to the SAME
    internal age distribution, AA = calib_age - chronological.
Metric rows: n, mean AA by group, delta, Welch p, MW p, Cohen's d, RLL AA<0 fraction,
Spearman(age, pred). All numbers written to CSV; nothing hidden (H32).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from train_v2 import build_features  # noqa: E402
from model_v2 import CraneZV2  # noqa: E402
import train_v2 as T  # noqa: E402
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

OUT = os.path.join(C.OUT_DIR, "aa_stratification_vs_baselines.csv")
COHORT2_PATH = "C:/Users/TS/Desktop/crane/external_test_data/cohort2/cohort2_validation_blood.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[P4b] device={device}")


def cohend(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if s == 0:
        return 0.0
    return (a.mean() - b.mean()) / s


def summarize(model, cohort, aa_rll, aa_ryc, chrono, pred, aa_all):
    return {
        "model": model, "cohort": cohort,
        "n_rll": len(aa_rll), "n_ryc": len(aa_ryc),
        "aa_rll_mean": float(aa_rll.mean()), "aa_ryc_mean": float(aa_ryc.mean()),
        "delta_aa": float(aa_rll.mean() - aa_ryc.mean()),
        "welch_p": float(stats.ttest_ind(aa_rll, aa_ryc, equal_var=False).pvalue),
        "mw_p": float(stats.mannwhitneyu(aa_rll, aa_ryc, alternative="two-sided").pvalue),
        "cohen_d": float(cohend(aa_rll, aa_ryc)),
        "rll_aa_neg_frac": float((aa_rll < 0).mean()),
        "spearman_age_pred": float(stats.spearmanr(chrono, pred).correlation),
        "spearman_age_aa": float(stats.spearmanr(chrono, aa_all).correlation),
    }


def rank_calibrate(pred, internal_ages):
    pct = rankdata(pred) / len(pred)
    return np.quantile(internal_ages, pct)


# ---------------------------------------------------------------------------
# 0. internal age distribution (fold0 train) for ALL rank calibration
# ---------------------------------------------------------------------------
_, _, p0_tr, _ = C.load_fold(0)
internal_ages = p0_tr["age"].to_numpy(dtype=float)
print(f"[P4b] internal ages n={len(internal_ages)} range "
      f"{internal_ages.min():.0f}-{internal_ages.max():.0f}")

# ---------------------------------------------------------------------------
# 1. CRANE-Z AA per cohort
# ---------------------------------------------------------------------------
out_rows = []

# 1a. internal OOF (existing CSV, already calibrated)
int_df = pd.read_csv(os.path.join(C.OUT_DIR, "aa_internal_validation.csv"))
int_df = int_df[int_df["group"].isin(["RLL", "RYC"])]
int_rll = int_df[int_df["group"] == "RLL"]["AA"].to_numpy()
int_ryc = int_df[int_df["group"] == "RYC"]["AA"].to_numpy()
int_chrono = int_df["chrono"].to_numpy()
int_pred = int_df["pred"].to_numpy()
out_rows.append(summarize("CRANE-Z", "internal", int_rll, int_ryc, int_chrono,
                          int_pred, int_df["AA"].to_numpy()))
print(f"[CRANE-Z] internal OOF d={cohend(int_rll, int_ryc):+.2f}")

# 1b. GTEx blood (existing rank-calibrated master)
master = pd.read_csv(os.path.join(C.OUT_DIR, "aa_age_acceleration.csv"))
mb = master[master["sample"].str.contains("SM-")]
mb = mb[mb["group"].isin(["RLL", "RYC"])]
b_rll = mb[mb["group"] == "RLL"]["age_acceleration"].to_numpy()
b_ryc = mb[mb["group"] == "RYC"]["age_acceleration"].to_numpy()
b_chrono = mb["chronological_age"].to_numpy()
b_pred = mb["predicted_bio_age_calibrated"].to_numpy()
out_rows.append(summarize("CRANE-Z", "gtex_blood", b_rll, b_ryc, b_chrono, b_pred,
                          mb["age_acceleration"].to_numpy()))
print(f"[CRANE-Z] gtex_blood d={cohend(b_rll, b_ryc):+.2f}")

# 1c/1d. GTEx muscle + cohort2: compute CRANE-Z AA here (rank-calibrated)
def crane_aa_external(path, tag):
    extd = pd.read_csv(path, index_col=0)
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
        fold_ages.append(np.mean(ages_e, 0))
        print(f"[CRANE-Z {tag}] fold{fold} done")
    pred_raw = np.mean(fold_ages, 0)
    chrono = extd["age_mid"].to_numpy(dtype=float)
    group = extd["group"].to_numpy(dtype=str)
    calib = rank_calibrate(pred_raw, internal_ages)
    aa = calib - chrono
    mask = np.isin(group, ["RLL", "RYC"])
    return summarize("CRANE-Z", tag, aa[(group == "RLL") & mask],
                     aa[(group == "RYC") & mask], chrono[mask], calib[mask], aa[mask])

# 1c. GTEx muscle + cohort2: use existing CRANE-Z AA CSVs
muscle_df = None
if os.path.exists(os.path.join(C.OUT_DIR, "aa_gtex_muscle.csv")):
    muscle_df = pd.read_csv(os.path.join(C.OUT_DIR, "aa_gtex_muscle.csv"))
if muscle_df is None:
    print("[CRANE-Z] gtex_muscle AA not found - computing now")
    muscle_df = crane_aa_external(C.EXT_MUSCLE_PATH, "gtex_muscle")
    # crane_aa_external returns dict; wrap as dataframe row
    out_rows.append(muscle_df)
    print(f"[CRANE-Z] gtex_muscle d={muscle_df['cohen_d']:+.2f}")
else:
    muscle_df = muscle_df[muscle_df["group"].isin(["RLL", "RYC"])]
    m_rll = muscle_df[muscle_df["group"] == "RLL"]["AA"].to_numpy()
    m_ryc = muscle_df[muscle_df["group"] == "RYC"]["AA"].to_numpy()
    m_chrono = muscle_df["chrono"].to_numpy()
    m_pred = muscle_df["pred"].to_numpy()
    out_rows.append(summarize("CRANE-Z", "gtex_muscle", m_rll, m_ryc, m_chrono,
                              m_pred, muscle_df["AA"].to_numpy()))
    print(f"[CRANE-Z] gtex_muscle d={cohend(m_rll, m_ryc):+.2f}")

c2_path = os.path.join(C.OUT_DIR, "aa_cohort2_validation.csv")
if os.path.exists(c2_path):
    c2_df = pd.read_csv(c2_path)
    c2_df = c2_df[c2_df["group"].isin(["RLL", "RYC"])]
    c2_rll = c2_df[c2_df["group"] == "RLL"]["AA"].to_numpy()
    c2_ryc = c2_df[c2_df["group"] == "RYC"]["AA"].to_numpy()
    c2_chrono = c2_df["chrono"].to_numpy()
    c2_pred = c2_df["pred_calib"].to_numpy() if "pred_calib" in c2_df.columns else c2_df["pred"].to_numpy()
    out_rows.append(summarize("CRANE-Z", "cohort2", c2_rll, c2_ryc, c2_chrono,
                              c2_pred, c2_df["AA"].to_numpy()))
    print(f"[CRANE-Z] cohort2 d={cohend(c2_rll, c2_ryc):+.2f}")
else:
    c2 = crane_aa_external(COHORT2_PATH, "cohort2")
    out_rows.append(c2)
    print(f"[CRANE-Z] cohort2 d={c2['cohen_d']:+.2f}")

# ---------------------------------------------------------------------------
# 2. Baselines: internal OOF + all externals
# ---------------------------------------------------------------------------
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=10000),
    "CLC-S": LinearSVR(C=1.0, max_iter=10000, random_state=0),
    "KNN": KNeighborsRegressor(n_neighbors=15),
    "DT": DecisionTreeRegressor(max_depth=8, random_state=0),
}

# 2a. internal OOF (5-fold, concat) - group from p_va["group"] column
for name, mdl in models.items():
    aa_all, chrono_all, pred_all, grp_all = [], [], [], []
    for fold in range(5):
        X_tr, X_va, p_tr, p_va = C.load_fold(fold)
        genes = build_features(fold)["genes"]
        zsc = C.make_std_scaler(X_tr[genes])
        Xtr = zsc.transform(C.align_features(X_tr, genes))
        Xva = zsc.transform(C.align_features(X_va, genes))
        mdl.fit(Xtr, p_tr["age"].to_numpy())
        pv = mdl.predict(Xva)
        aa_all.extend(pv - p_va["age"].to_numpy())
        chrono_all.extend(p_va["age"].to_numpy())
        pred_all.extend(pv)
        grp_all.extend(p_va["group"].to_numpy())
    aa_all = np.array(aa_all); chrono_all = np.array(chrono_all)
    pred_all = np.array(pred_all); grp_all = np.array(grp_all)
    out_rows.append(summarize(name, "internal",
                              aa_all[grp_all == "RLL"], aa_all[grp_all == "RYC"],
                              chrono_all, pred_all, aa_all))
    print(f"[{name}] internal d={cohend(aa_all[grp_all=='RLL'], aa_all[grp_all=='RYC']):+.2f}")

# 2b. externals: FAIR comparison - 5-fold fitted models, 5-fold mean prediction
ext_specs = [("gtex_blood", C.EXT_BLOOD_PATH), ("gtex_muscle", C.EXT_MUSCLE_PATH),
             ("cohort2", COHORT2_PATH)]
folds_data = []
for fold in range(5):
    X_tr, _, p_tr, _ = C.load_fold(fold)
    genes = build_features(fold)["genes"]
    zsc = C.make_std_scaler(X_tr[genes])
    folds_data.append((zsc, C.align_features(X_tr, genes), p_tr["age"].to_numpy(), genes))

# external: 5-fold fitted models, 5-fold mean prediction (same ensemble spirit as CRANE-Z)
for cohort, path in ext_specs:
    extd = pd.read_csv(path, index_col=0)
    meta = [c for c in ["age_mid", "group", "sex"] if c in extd.columns]
    expr = extd.drop(columns=meta) if meta else extd
    chrono = extd["age_mid"].to_numpy(dtype=float)
    group = extd["group"].to_numpy(dtype=str)
    for name, mdl in models.items():
        preds = []
        for zsc, Xtrh, age_tr, genes_f in folds_data:
            mdl.fit(Xtrh, age_tr)
            Xe_hvg = zsc.transform(C.align_features(expr, genes_f))
            preds.append(mdl.predict(Xe_hvg))
        pv = np.mean(preds, 0)          # 5-fold mean predicted age (raw)
        calib = rank_calibrate(pv, internal_ages)
        aa = calib - chrono
        mask = np.isin(group, ["RLL", "RYC"])
        out_rows.append(summarize(name, cohort,
                                  aa[(group == "RLL") & mask],
                                  aa[(group == "RYC") & mask],
                                  chrono[mask], calib[mask], aa[mask]))
        print(f"[{name}] {cohort} d={cohend(aa[(group=='RLL')&mask], aa[(group=='RYC')&mask]):+.2f}")

# ---------------------------------------------------------------------------
# 3. Save + verdict
# ---------------------------------------------------------------------------
df_out = pd.DataFrame(out_rows)
df_out.to_csv(OUT, index=False)
print(f"\nsaved -> {OUT}")

print("\n=== VERDICT: baseline AA stratification vs CRANE-Z ===")
cr = df_out[df_out["model"] == "CRANE-Z"].set_index("cohort")
for _, b in df_out[df_out["model"] != "CRANE-Z"].iterrows():
    if b["cohort"] in cr.index:
        c = cr.loc[b["cohort"]]
        print(f"{b['model']:6s} {b['cohort']:12s} d={b['cohen_d']:+.2f} "
              f"delta={b['delta_aa']:+.1f} vs CRANE-Z d={c['cohen_d']:+.2f} "
              f"delta={c['delta_aa']:+.1f}")
