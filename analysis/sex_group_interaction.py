# -*- coding: utf-8 -*-
"""Sex x age-group interaction test (formal): AA ~ group * sex with interaction term.

Upgrades the exploratory stratified Welch test (p=0.029) into a formal
interaction test: does the AA difference between groups depend on sex?
Uses per-sample ensemble AA on internal validation folds (aa_sex_internal.csv
has per-sample AA with sex/group). Linear model: AA ~ 1 + group + sex + group:sex.
"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE = r"C:/Users/TS/Desktop/crane"
OUT = os.path.join(BASE, "Code_v2", "output")

# ---- per-sample AA from _aa_sex.py output ----
df = pd.read_csv(os.path.join(OUT, "aa_sex_internal.csv"))
print("rows:", len(df), "| cols:", df.columns.tolist())

# sex already in the per-sample file (from fold pheno); normalize code
df["sex_u"] = df["sex"].astype(str).map(lambda v: "F" if v in ("F", "2", "female", "Female") else "M")

# keep one row per sample (per-sample ensemble AA) -> collapse folds
s = df.groupby(["sample", "group", "sex_u"])["AA"].mean().reset_index()
print("per-sample n:", len(s), "| group counts:", s["group"].value_counts().to_dict())

# ---- formal interaction model ----
s["group_rll"] = (s["group"] == "RLL").astype(int)
s["sex_f"] = (s["sex_u"] == "F").astype(int)

X = sm.add_constant(s[["group_rll", "sex_f"]])
X["group_x_sex"] = X["group_rll"] * X["sex_f"]
m = sm.OLS(s["AA"], X).fit()
print("\n=== OLS AA ~ group + sex + group:sex (per-sample) ===")
print(m.summary2().tables[1].round(4).to_string())

inter_p = float(m.pvalues["group_x_sex"])
print(f"\n>>> Interaction p (group x sex) = {inter_p:.4f}")

# ---- interpret ----
print(f"\nInterpretation: interaction {'SIGNIFICANT' if inter_p < 0.05 else 'NOT significant'} at alpha=0.05")

# main effects table for the paper
rows = []
for g in ["RYC", "RLL"]:
    for sx in ["M", "F"]:
        sub = s[(s["group"] == g) & (s["sex_u"] == sx)]["AA"]
        rows.append({"group": g, "sex": sx, "n": len(sub),
                     "AA_mean": sub.mean(), "AA_sd": sub.std()})
tbl = pd.DataFrame(rows)
tbl.to_csv(os.path.join(OUT, "sex_group_interaction_table.csv"), index=False)
print("\n=== table ===")
print(tbl.round(2).to_string(index=False))

# marginal sex effect within each group (for narrative, alongside interaction)
for g in ["RYC", "RLL"]:
    a = s[(s["group"] == g) & (s["sex_u"] == "F")]["AA"]
    b = s[(s["group"] == g) & (s["sex_u"] == "M")]["AA"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"[{g}] F AA {a.mean():+.2f} vs M {b.mean():+.2f} | Welch p={p:.4f}")

# save model summary for records
with open(os.path.join(OUT, "sex_group_interaction_model.txt"), "w") as f:
    f.write(str(m.summary2()))
print("\nsaved: sex_group_interaction_table.csv + sex_group_interaction_model.txt")
