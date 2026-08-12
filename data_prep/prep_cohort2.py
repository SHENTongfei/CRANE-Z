# -*- coding: utf-8 -*-
"""Build CRANE-Z Cohort2 (GSE123696, Israel PrimeView whole blood, ages 23-96).

Pipeline:
1. Parse GSE123696 series matrix -> probe x sample expression (log2 intensity)
2. Map probes -> gene symbols via GPL15207 annotation
3. Collapse multi-probe genes by max
4. Cross-platform calibration: per-gene ECDF mapping from chip log2 scale
   onto the training-fold0 TPM distribution (reference quantiles)
5. Write gene x sample matrix + age_mid/group/sex metadata columns,
   same layout as gtex_validation_balanced.csv
"""
import os, sys, gzip, re
import numpy as np
import pandas as pd

BASE = os.environ.get("CRANE_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
COH2 = os.path.join(BASE, "external_test_data/cohort2")
OUT = os.path.join(COH2, "cohort2_validation_blood.csv")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C

# ---------------- 1. parse series matrix ----------------
with gzip.open(os.path.join(COH2, "GSE123696_sm.gz"), "rt", encoding="latin-1") as f:
    lines = f.readlines()

meta = {}
for l in lines:
    if l.startswith("!Sample_geo_accession"):
        sids = [s.strip('"') for s in l.strip().split("\t")[1:]]
    if l.startswith("!Sample_characteristics_ch1"):
        m = re.findall(r"age: (\d+)", l)
        if m:
            ages = [int(x) for x in m]
        gm = re.findall(r"gender: (\w+)", l)
        if gm:
            gends = gm

# data table
dstart = None
for i, l in enumerate(lines):
    if l.startswith("!series_matrix_table_begin"):
        dstart = i + 2
        break
hdr = lines[dstart - 1].strip().split("\t")
samp_cols = [h.strip('"') for h in hdr[1:]]
print("samples:", len(samp_cols), "| ages:", len(ages), "| genders:", len(gends))
assert len(samp_cols) == len(ages) == len(gends)

probes, mat = [], []
for l in lines[dstart:]:
    if l.startswith("!"):
        break
    p = l.strip().split("\t")
    probes.append(p[0].strip('"'))
    mat.append([float(x) for x in p[1:]])
expr = pd.DataFrame(np.array(mat), index=probes, columns=samp_cols, dtype=float)
print("probe matrix:", expr.shape)

# ---------------- 2. probe -> gene map (ArrayExpress ADF, full 49372 probes) ----------------
with open(os.path.join(COH2, "primeview_adf.txt"), "r", encoding="utf-8", errors="replace") as f:
    alines = f.readlines()
hdr_i = None
for i, l in enumerate(alines):
    if l.startswith("Reporter Name\t"):
        hdr_i = i
        break
acols = alines[hdr_i].rstrip("\n").split("\t")
sym_i = acols.index("Comment[Gene Symbol]")
probe2sym = {}
for l in alines[hdr_i + 1:]:
    if l.startswith("Composite Element") or l.startswith("Array Design"):
        break
    if not l.strip():
        continue
    parts = l.rstrip("\n").split("\t")
    if len(parts) <= sym_i:
        continue
    probe = parts[0].strip()
    sym = parts[sym_i].strip()
    if probe and sym and sym != "---":
        probe2sym[probe] = sym.split(" /// ")[0]
print("probe->gene mapped:", len(probe2sym), "/", expr.shape[0])

# ---------------- 3. collapse to genes (max probe) ----------------
expr["__gene"] = [probe2sym.get(p, "") for p in expr.index]
expr = expr[expr["__gene"] != ""]
expr = expr.groupby("__gene").max()
print("gene-level matrix:", expr.shape)

# ---------------- 4. cross-platform calibration ----------------
# Reference distribution = training fold0 genes (TPM)
X0_tr, _, _, _ = C.load_fold(0)
genes = list(expr.index)
common_genes = [g for g in genes if g in X0_tr.columns]
print("genes in train reference:", len(common_genes), "/", len(genes))
ref = X0_tr[common_genes]

def ecdf_map_target(col, ref_vals):
    """Map chip log2 values onto the TPM reference distribution (rank-based)."""
    col = np.asarray(col, dtype=float)
    ref_sorted = np.sort(ref_vals.values)
    n = len(ref_sorted)
    ranks = np.argsort(np.argsort(col))  # 0-based rank of each value
    frac = (ranks + 1) / (len(col) + 1)  # quantile of each chip value within chip dist
    idx = np.clip((frac * (n + 1) - 1).astype(int), 0, n - 1)
    return ref_sorted[idx]

cal = pd.DataFrame(index=expr.index, columns=expr.columns)
for g in common_genes:
    cal.loc[g] = ecdf_map_target(expr.loc[g].values, ref[g])

# Clean: NaN/Inf from constant probes -> replace with reference median (0 effect after rank-gauss)
ref_med = ref.median()
cal = cal.replace([np.inf, -np.inf], np.nan)
for g in cal.index:
    mask = cal.loc[g].isna()
    if mask.any():
        cal.loc[g, mask] = ref_med[g] if g in ref_med.index else 0.0
cal = cal.fillna(0.0)
assert not np.isnan(cal.values).any() and not np.isinf(cal.values).any(), "calibration residual NaN/Inf"
print("calibrated:", cal.shape, "| NaN/Inf cleaned")

# ---------------- 5. output with metadata ----------------
cal = cal.T  # samples x genes
cal["age_mid"] = ages
cal["sex"] = [1 if g == "Male" else 2 for g in gends]
cal["group"] = ["RLL" if a >= 70 else ("RYC" if a <= 50 else "MID") for a in ages]
cal.index = samp_cols
cal.to_csv(OUT)
print("saved:", OUT, cal.shape)
print("group counts:", cal["group"].value_counts().to_dict())
print("sex counts:", cal["sex"].value_counts().to_dict())
