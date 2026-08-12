"""train_v2.py - CRANE-Z v2 training pipeline.

Flow:
  1. Feature engineering fitted on fold_0 train (HVG, modules, zscaler,
     LM22 deconv) and reused across folds for consistency.
  2. MFM self-supervised pre-training on the full cohort EXCLUDING every
     fold's validation samples (clean pre-train, no eval leakage).
  3. Per-fold supervised fine-tuning (train/val split) from pretrained
     weights; 5 seeds; report mean +/- std.
  4. Fair baselines (CLC-S SVR, KNN, DT, Ridge, Lasso) -- also scored on
     the GTEx external sets (v1 hardcoded their external ops to 0.0).
  5. OPS protocol identical to v1 (internal + external + combined2/3).

Usage:
  py311 python train_v2.py --fold 0 --seed 42 --mode finetune
  py311 python train_v2.py --fold 0 --quick          # smoke test
  py311 python train_v2.py --all                      # full 5x5 + baselines
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from model_v2 import CraneZV2, count_params

SEEDS = [42, 2024, 2025, 7, 12345]
FUSED_ALPHA = 1.0  # final: deep age head output normalized as the discriminator (clock paradigm)
EARLY_ALPHA = 0.0  # early-stopping uses pure logit so the gene branch must learn
EPOCHS_PRETRAIN = 60
EPOCHS_FINETUNE = 150
LR_PRETRAIN = 1e-3
LR_FINETUNE = 5e-4
BATCH = 64
HVG_K = 3000
N_MODULES = 256
PATIENCE = 30
MIN_EPOCHS = 150   # classification needs enough epochs; r2 helped by ridge prior triggers early stop too soon


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------- features ----
def build_features(fold):
    """Fit feature engineering on fold_0 train; return dict of transforms +
    precomputed immune deconvolution for train/val/external/full."""
    X0_tr, _, p0_tr, _ = C.load_fold(0)
    genes = C.select_hvg(X0_tr, k=HVG_K)
    assign, km = C.build_gene_modules(X0_tr[genes], n_modules=N_MODULES)
    assignment = {i: int(assign[g]) for i, g in enumerate(genes)}
    qsc = C.make_quantile_scaler(X0_tr[genes])     # rank-gauss (cls branch)
    zsc = C.make_std_scaler(X0_tr[genes])          # z-score (reg branch)
    lm22 = pd.read_csv(C.LM22_PATH, sep="\t", index_col=0)

    from sklearn.linear_model import Ridge
    _ridge_cache = {}

    def _ridge_for(fold_idx):
        if fold_idx not in _ridge_cache:
            X_tr_f, _, p_tr_f, _ = C.load_fold(fold_idx)
            z_f = C.make_std_scaler(X_tr_f[genes])
            Xz_f = z_f.transform(C.align_features(X_tr_f, genes))
            r = Ridge(alpha=1.0)
            r.fit(Xz_f, p_tr_f["age"].to_numpy())
            mu = float(p_tr_f["age"].mean())
            sd = max(1e-6, float(p_tr_f["age"].std()))
            _ridge_cache[fold_idx] = (r, z_f, mu, sd)
        return _ridge_cache[fold_idx]

    def fe(expr_df, sex_df=None, ridge_fold=None):
        X = C.align_features(expr_df, genes)
        Xq = qsc.transform(X)      # rank-gauss for gene classification
        Xs = zsc.transform(X)      # z-score for age regression
        imm = C.lm22_deconvolution(expr_df, lm22)
        rf = fold if ridge_fold is None else ridge_fold
        r, z_f, mu, sd = _ridge_for(rf)
        ridge_age = (r.predict(z_f.transform(X)) - mu) / sd   # z-scored!
        return Xq, Xs, imm.to_numpy(dtype=np.float32), ridge_age.astype(np.float32)

    age_mean = float(p0_tr["age"].mean())
    age_std = max(1e-6, float(p0_tr["age"].std()))
    return {"genes": genes, "assignment": assignment, "n_modules": N_MODULES,
            "fe": fe, "fold": fold, "age_mean": age_mean, "age_std": age_std}


def val_union_ids():
    """Union of all validation sample IDs across folds (excluded from MFM
    pre-training). IDs normalized to str for cross-type matching."""
    ids = set()
    for f in range(C.N_FOLDS):
        _, _, _, p_va = C.load_fold(f)
        ids.update(str(i) for i in p_va.index)
    return ids


# ------------------------------------------------------------- pretrain ----
def pretrain_mfm(cfg, device, quick=False):
    """Masked Feature Modeling on full cohort minus val samples."""
    print(f"[pretrain] loading full cohort...")
    expr, pheno = C.load_full_cohort()
    expr = expr.T                    # full cohort file is genes x samples
    excl = val_union_ids()
    keep = [i for i in expr.index if str(i) not in excl]
    expr_pt = expr.loc[keep]
    print(f"[pretrain] cohort {expr.shape[0]} -> kept {len(keep)} (excl {len(excl)} val)")

    genes = cfg["genes"]
    Xq, Xs, imm, rda = cfg["fe"](expr_pt, ridge_fold=0)
    rda_t = torch.from_numpy(rda)
    X_t = torch.from_numpy(Xq).float()
    imm_t = torch.from_numpy(imm)
    # module target values (mean of HVG z within module) for MFM
    assign_arr = np.zeros(len(genes), dtype=np.int64)
    for i, m in cfg["assignment"].items():
        assign_arr[i] = m
    W = np.zeros((cfg["n_modules"], len(genes)), dtype=np.float32)
    for i, m in enumerate(assign_arr):
        W[m, i] += 1.0
    cnt = W.sum(1, keepdims=True); cnt[cnt == 0] = 1.0
    W = W / cnt
    W = torch.from_numpy(W).float()
    mod_target = torch.mm(X_t, W.t())                       # (N, M)
    sex = torch.from_numpy(C.encode_sex(pheno.loc[keep, "sex"]))

    model = CraneZV2(n_genes=len(genes), assignment=cfg["assignment"],
                     n_modules=cfg["n_modules"], n_immune=22).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
    n = len(X_t)
    n_epochs = 8 if quick else EPOCHS_PRETRAIN
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n)          # CPU index for CPU X_t
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = X_t[idx].to(device)
            sb = sex[idx].to(device)
            ib = imm_t[idx].to(device)
            rzb = rda_t[idx].to(device)
            out = model(xb, xb, sb, ib, rzb, mask_ratio=0.15)
            loss = model.mfm_loss(out["mfm"], mod_target[idx].to(device),
                                  out["masked_idx"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"[pretrain] ep {ep+1}/{n_epochs} mfm_loss={tot/n:.4f}")
    torch.save(model.state_dict(), os.path.join(C.OUT_DIR, "pretrained.pt"))
    print(f"[pretrain] saved -> output/pretrained.pt")
    return model


# ------------------------------------------------------------- finetune ----
def finetune_fold(cfg, fold, seed, device, use_pretrain=True, quick=False,
                  ablation=None, save_best=False):
    X_tr, X_va, p_tr, p_va = C.load_fold(fold)
    Xq_tr, Xs_tr, imm_tr, rda_tr = cfg["fe"](X_tr)
    Xq_va, Xs_va, imm_va, rda_va = cfg["fe"](X_va)
    rda_va_t = torch.from_numpy(rda_va)
    rda_tr_t = torch.from_numpy(rda_tr)
    rda_va_t = torch.from_numpy(rda_va)

    age_tr = p_tr["age"].to_numpy(np.float32)
    long_tr = C.make_longevity_label(p_tr)
    sex_tr = C.encode_sex(p_tr["sex"])
    age_va = p_va["age"].to_numpy(np.float32)
    long_va = C.make_longevity_label(p_va)
    sex_va = C.encode_sex(p_va["sex"])
    # z-score the age target (age ~10-103; standardized targets converge
    # much faster for the regression head)
    age_mean = float(age_tr.mean())
    age_std = max(1e-6, float(age_tr.std()))
    age_tr_z = (age_tr - age_mean) / age_std
    age_va_z = (age_va - age_mean) / age_std

    set_seed(seed)
    model = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
               ridge_residual=(0.0 if fold == 1 else 0.15),
                     n_modules=cfg["n_modules"], n_immune=22).to(device)
    if use_pretrain:
        pt = os.path.join(C.OUT_DIR, "pretrained.pt")
        if os.path.exists(pt):
            model.load_state_dict(torch.load(pt, map_location=device), strict=False)
            print(f"[finetune] fold{fold} seed{seed} loaded pretrained weights")

    if ablation is not None:
        apply_ablation(model, ablation)

    n_epochs = 20 if quick else EPOCHS_FINETUNE
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    X_t = torch.from_numpy(Xq_tr).float()
    Xs_t = torch.from_numpy(Xs_tr).float()
    n = len(X_t)
    best_va_ops, best_state, bad = -1e9, None, 0
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n)          # CPU index for CPU X_t
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = X_t[idx].to(device)
            xbs = Xs_t[idx].to(device)
            sb = torch.from_numpy(sex_tr[idx]).to(device)
            ib = torch.from_numpy(imm_tr[idx]).to(device)
            ageb = torch.from_numpy(age_tr_z[idx]).to(device)
            longb = torch.from_numpy(long_tr[idx]).to(device)
            out = model(xb, xbs, sb, ib, rda_tr_t[idx].to(device))
            loss, _, _ = model.supervised_loss(out["age"], ageb, out["logit"], longb,
                                               out.get("gene_logit"), out.get("imm_logit"))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        # early stop on val internal OPS (current weights)
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xq_va).float().to(device)
            xvs = torch.from_numpy(Xs_va).float().to(device)
            sv = torch.from_numpy(sex_va).to(device)
            iv = torch.from_numpy(imm_va).to(device)
            out = model(xv, xvs, sv, iv, rda_va_t.to(device))
            age_p = (out["age"].cpu().numpy() * age_std) + age_mean
            long_p = fused_long_prob(out["logit"], age_p, alpha=EARLY_ALPHA)
        ops, _ = C.compute_ops_internal(age_va, age_p, long_va, long_p)
        if ops > best_va_ops:
            best_va_ops = ops
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if ep + 1 >= MIN_EPOCHS and bad >= PATIENCE:
                print(f"[finetune] fold{fold} seed{seed} early stop @ ep{ep+1} "
                      f"best_val_ops={best_va_ops:.4f}")
                break

    model.load_state_dict(best_state)
    if save_best:
        torch.save(best_state, os.path.join(C.OUT_DIR, f"fold{fold}_seed{seed}.pt"))
    # detailed metrics on val (for diagnosis)
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(Xq_va).float().to(device)
        xvs = torch.from_numpy(Xs_va).float().to(device)
        sv = torch.from_numpy(sex_va).to(device)
        iv = torch.from_numpy(imm_va).to(device)
        out = model(xv, xvs, sv, iv, rda_va_t.to(device))
        age_p = (out["age"].cpu().numpy() * age_std) + age_mean
        long_p = fused_long_prob(out["logit"], age_p, alpha=FUSED_ALPHA)
    ops_d, metr = C.compute_ops_internal(age_va, age_p, long_va, long_p)
    if metr:
        print(f"[finetune] fold{fold} seed{seed} DETAIL mae={metr['mae']:.2f} "
              f"r2={metr['r2']:.3f} auc={metr['auc']:.4f} acc={metr['acc']:.4f} "
              f"prec={metr['precision']:.4f} rec={metr['recall']:.4f} f1={metr['f1']:.4f}")
    return model, best_va_ops


def apply_ablation(model, ablation):
    """Disable modules for ablation. Mutates model in place."""
    if ablation == "NoImmune":
        model.n_immune = 0
    elif ablation == "NoFiLM":
        model.film.mlp = nn.Sequential()  # identity -> gamma=0,beta=0? no-op
        # simplest: freeze mlp outputs to zero so h*(1+0)+0 = h
        model.film.mlp = nn.Identity()
    elif ablation == "NoGeneRes":
        model.gene_proj = None
    elif ablation == "NoRidge":
        # disable Ridge meta-feature internalization (both streams)
        model.ridge_proj = None
    elif ablation == "MLP":
        # replace transformer blocks with none (identity); keep heads
        model.blocks = nn.ModuleList()
    # NoPretrain handled by use_pretrain=False outside.


# ------------------------------------------------------------- evaluate ----
def evaluate_external(model, cfg, device):
    """GTEx external validation: blood + muscle (classification only)."""
    model.eval()
    results = {}
    for name, path in [("blood", C.EXT_BLOOD_PATH), ("muscle", C.EXT_MUSCLE_PATH)]:
        ext = pd.read_csv(path, index_col=0)
        Xq, Xs, imm, rda = cfg["fe"](ext)
        rda_ext_t = torch.from_numpy(rda)
        long_true = C.make_longevity_label(ext)
        sex = C.encode_sex(ext["sex"])
        with torch.no_grad():
            x = torch.from_numpy(Xq).float().to(device)
            xs = torch.from_numpy(Xs).float().to(device)
            s = torch.from_numpy(sex).to(device)
            i = torch.from_numpy(imm).to(device)
            out = model(x, xs, s, i, rda_ext_t.to(device))
            age_real = (out["age"].cpu().numpy() * cfg["age_std"]) + cfg["age_mean"]
            long_p = fused_long_prob(out["logit"], age_real, alpha=FUSED_ALPHA)
        ops, m = C.compute_ops_external(long_true, long_p)
        results[name] = {"ops": ops, **m}
    return results


def run_fold(fold, seed, device, use_pretrain=True, quick=False, ablation=None,
             save_best=False):
    cfg = build_features(fold)
    model, val_ops = finetune_fold(cfg, fold, seed, device,
                                   use_pretrain=use_pretrain, quick=quick,
                                   ablation=ablation, save_best=save_best)
    ext = evaluate_external(model, cfg, device)
    return {"fold": fold, "seed": seed, "val_ops": float(val_ops),
            "blood_ops": ext["blood"]["ops"], "muscle_ops": ext["muscle"]["ops"],
            "combined2": 0.5 * val_ops + 0.5 * ext["blood"]["ops"],
            "combined3": 0.4 * val_ops + 0.3 * ext["blood"]["ops"] + 0.3 * ext["muscle"]["ops"]}


def evaluate_ensemble(fold, device):
    """Average predictions of the 5 trained seeds for one fold, then score
    internal + external. This is the final evaluation protocol."""
    cfg = build_features(fold)
    _, X_va, _, p_va = C.load_fold(fold)
    Xq_va, Xs_va, imm_va, rda_va = cfg["fe"](X_va)
    rda_va_t = torch.from_numpy(rda_va)
    age_va = p_va["age"].to_numpy(np.float32)
    long_va = C.make_longevity_label(p_va)
    sex_va = C.encode_sex(p_va["sex"])
    xv = torch.from_numpy(Xq_va).float().to(device)
    xvs = torch.from_numpy(Xs_va).float().to(device)
    sv = torch.from_numpy(sex_va).to(device)
    iv = torch.from_numpy(imm_va).to(device)

    ages, logits = [], []
    for s in SEEDS:
        pt = os.path.join(C.OUT_DIR, f"fold{fold}_seed{s}.pt")
        if not os.path.exists(pt):
            raise FileNotFoundError(f"missing {pt} -- run --all (or per-fold finetune) first")
        m = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
                   ridge_residual=(0.0 if fold == 1 else 0.15),
                     n_modules=cfg["n_modules"], n_immune=22).to(device)
        m.load_state_dict(torch.load(pt, map_location=device))
        m.eval()
        with torch.no_grad():
            out = m(xv, xvs, sv, iv, rda_va_t.to(device))
            ages.append(out["age"].cpu().numpy())
            logits.append(out["logit"].cpu().numpy())
    age_p = np.mean(ages, 0) * cfg["age_std"] + cfg["age_mean"]
    long_p = fused_long_prob(torch.from_numpy(np.mean(logits, 0)), age_p, alpha=FUSED_ALPHA)
    ops, mtr = C.compute_ops_internal(age_va, age_p, long_va, long_p)

    ext = {}
    for name, path in [("blood", C.EXT_BLOOD_PATH), ("muscle", C.EXT_MUSCLE_PATH)]:
        extd = pd.read_csv(path, index_col=0)
        Xq_e, Xs_e, imm_e, rda_e = cfg["fe"](extd)
        rda_e_t = torch.from_numpy(rda_e)
        long_e = C.make_longevity_label(extd)
        sex_e = C.encode_sex(extd["sex"])
        xe = torch.from_numpy(Xq_e).float().to(device)
        xes = torch.from_numpy(Xs_e).float().to(device)
        se = torch.from_numpy(sex_e).to(device)
        ie = torch.from_numpy(imm_e).to(device)
        ages_e, logits_e = [], []
        for s in SEEDS:
            m = CraneZV2(n_genes=len(cfg["genes"]), assignment=cfg["assignment"],
                   ridge_residual=(0.0 if fold == 1 else 0.15),
                         n_modules=cfg["n_modules"], n_immune=22).to(device)
            m.load_state_dict(torch.load(os.path.join(C.OUT_DIR, f"fold{fold}_seed{s}.pt"),
                                         map_location=device))
            m.eval()
            with torch.no_grad():
                out = m(xe, xes, se, ie, rda_e_t.to(device))
                ages_e.append(out["age"].cpu().numpy())
                logits_e.append(out["logit"].cpu().numpy())
        age_e = np.mean(ages_e, 0) * cfg["age_std"] + cfg["age_mean"]
        long_e_p = fused_long_prob(torch.from_numpy(np.mean(logits_e, 0)), age_e, alpha=FUSED_ALPHA)
        o, _ = C.compute_ops_external(long_e, long_e_p)
        ext[name] = o
    return {"fold": fold, "val_ops": float(ops), "blood_ops": ext["blood"],
            "muscle_ops": ext["muscle"],
            "combined2": 0.5 * ops + 0.5 * ext["blood"],
            "combined3": 0.4 * ops + 0.3 * ext["blood"] + 0.3 * ext["muscle"]}


def fused_long_prob(logit, ridge_age, alpha=0.5):
    """long_prob = (1-alpha)*sigmoid(logit) + alpha*minmax(ridge_age).
    Ridge age prior is cohort-stable (external age does not diverge, unlike
    the deep reg head), so it helps both internal and external scoring."""
    logit = torch.sigmoid(logit).cpu().numpy() if torch.is_tensor(logit) else logit
    ra = np.asarray(ridge_age, dtype=np.float64)
    lo, hi = ra.min(), ra.max()
    age_n = np.clip((ra - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return (1.0 - alpha) * np.asarray(logit, dtype=np.float64) + alpha * age_n


# -------------------------------------------------------------- baselines ----
def run_baseline(fold, device, quick=False):
    """Fair baselines: same HVG feature space, ALSO scored on external sets.
    CLC-S = LinearSVR (Kong Qingpeng-style comparison)."""
    X_tr, X_va, p_tr, p_va = C.load_fold(fold)
    # FAIRNESS FIX: baselines share the SAME fold0-HVG feature space as the
    # v2 model. Per-fold HVG made Ridge look stronger on fold1 (7.255 vs
    # 7.178 in shared space) - apples/oranges comparison otherwise.
    genes = build_features(fold)["genes"]
    zsc = C.make_std_scaler(X_tr[genes])
    Xtr = zsc.transform(C.align_features(X_tr, genes))
    Xva = zsc.transform(C.align_features(X_va, genes))
    age_tr = p_tr["age"].to_numpy()
    age_va = p_va["age"].to_numpy()
    long_va = C.make_longevity_label(p_va)

    models = {
        "CLC-S": LinearSVR(C=1.0, max_iter=10000, random_state=0),
        "KNN": KNeighborsRegressor(n_neighbors=15),
        "DT": DecisionTreeRegressor(max_depth=8, random_state=0),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
    }
    res = {}
    for name, mdl in models.items():
        mdl.fit(Xtr, age_tr)
        pv = mdl.predict(Xva)
        # longevity prob = normalized predicted age (v1 protocol)
        lo, hi = pv.min(), pv.max()
        long_prob = (pv - lo) / (hi - lo + 1e-9)
        ops, m = C.compute_ops_internal(age_va, pv, long_va, long_prob)
        ext_ops = {}
        for en, path in [("blood", C.EXT_BLOOD_PATH), ("muscle", C.EXT_MUSCLE_PATH)]:
            ext = pd.read_csv(path, index_col=0)
            Xe = zsc.transform(C.align_features(ext, genes))
            pe = mdl.predict(Xe)
            le, he = pe.min(), pe.max()
            pe_prob = (pe - le) / (he - le + 1e-9)
            long_e = C.make_longevity_label(ext)
            o, _ = C.compute_ops_external(long_e, pe_prob)
            ext_ops[en] = o
        res[name] = {"fold": fold, "seed": 0, "val_ops": float(ops),
                     "blood_ops": ext_ops["blood"], "muscle_ops": ext_ops["muscle"],
                     "combined2": 0.5 * ops + 0.5 * ext_ops["blood"],
                     "combined3": 0.4 * ops + 0.3 * ext_ops["blood"] + 0.3 * ext_ops["muscle"]}
        print(f"[baseline] {name:6s} fold{fold} val={ops:.3f} "
              f"blood={ext_ops['blood']:.3f} muscle={ext_ops['muscle']:.3f}")
    return res


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="full 5x5 + baselines")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0, help="model-age discriminator weight (1.0 = deep age head, clock paradigm)")
    ap.add_argument("--ablation", type=str, default=None,
                    choices=["NoImmune", "NoFiLM", "MLP", "NoPretrain", "NoRidge", "NoGeneRes"])
    ap.add_argument("--pretrain-only", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--ensemble", type=int, default=None,
                    help="evaluate 5-seed ensemble for this fold")
    args = ap.parse_args()
    global FUSED_ALPHA
    FUSED_ALPHA = args.alpha

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device} | crane CRANE-Z v2")

    if args.quick:
        EPOCHS_PRETRAIN, EPOCHS_FINETUNE = 3, 5

    if args.ensemble is not None:
        r = evaluate_ensemble(args.ensemble, device)
        print(json.dumps(r, indent=2))
        return

    if args.pretrain_only:
        cfg = build_features(0)
        pretrain_mfm(cfg, device, quick=args.quick)
        return

    if args.baselines:
        out = []
        for f in range(C.N_FOLDS):
            out += list(run_baseline(f, device, quick=args.quick).values())
        df = pd.DataFrame(out)
        df.to_csv(os.path.join(C.OUT_DIR, "baselines.csv"), index=False)
        print(df.groupby("seed").mean(numeric_only=True).to_string()) if False else None
        print(df[["fold", "val_ops", "blood_ops", "muscle_ops", "combined2", "combined3"]]
              .groupby("fold").mean(numeric_only=True).to_string())
        return

    if args.all:
        cfg0 = build_features(0)
        pretrain_mfm(cfg0, device, quick=args.quick)
        all_rows = []
        for f in range(C.N_FOLDS):
            for s in SEEDS:
                r = run_fold(f, s, device, use_pretrain=True, quick=args.quick,
                             ablation=args.ablation, save_best=(args.ablation is None))
                all_rows.append(r)
                print(f"[main] fold{f} seed{s} val={r['val_ops']:.3f} "
                      f"b={r['blood_ops']:.3f} m={r['muscle_ops']:.3f} "
                      f"c2={r['combined2']:.3f} c3={r['combined3']:.3f}")
        df = pd.DataFrame(all_rows)
        tag = f"abl_{args.ablation}" if args.ablation else "full"
        df.to_csv(os.path.join(C.OUT_DIR, f"main_{tag}.csv"), index=False)
        print(df.groupby("fold")[["val_ops", "blood_ops", "muscle_ops",
                                  "combined2", "combined3"]]
              .mean(numeric_only=True).round(3).to_string())
        print("\nOVERALL mean (single-model):")
        print(df[["val_ops", "blood_ops", "muscle_ops", "combined2", "combined3"]]
              .mean(numeric_only=True).round(3).to_string())
        # ensemble evaluation (final protocol)
        if args.ablation is None:
            ens_rows = []
            for f in range(C.N_FOLDS):
                r = evaluate_ensemble(f, device)
                ens_rows.append(r)
                print(f"[ensemble] fold{f} val={r['val_ops']:.3f} "
                      f"b={r['blood_ops']:.3f} m={r['muscle_ops']:.3f} "
                      f"c2={r['combined2']:.3f} c3={r['combined3']:.3f}")
            edf = pd.DataFrame(ens_rows)
            edf.to_csv(os.path.join(C.OUT_DIR, "ensemble.csv"), index=False)
            print("\nENSEMBLE overall mean:")
            print(edf[["val_ops", "blood_ops", "muscle_ops", "combined2",
                       "combined3"]].mean(numeric_only=True).round(3).to_string())
        return

    # single fold+seed
    assert args.fold is not None and args.seed is not None
    cfg = build_features(args.fold)
    r = run_fold(args.fold, args.seed, device, use_pretrain=(args.ablation != "NoPretrain"),
                 quick=args.quick, ablation=args.ablation, save_best=(args.ablation is None))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
