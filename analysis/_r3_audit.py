# -*- coding: utf-8 -*-
"""R3 forensic numeric audit for CRANE-Z cohort2 claims (run from Code_v2 dir)."""
import pandas as pd, numpy as np
from scipy import stats
df = pd.read_csv("output/aa_stratification_vs_baselines.csv")
c = df[(df.model == "CRANE-Z") & (df.cohort == "cohort2")].iloc[0]
print(f"n: RLL={c.n_rll} RYC={c.n_ryc} (sample-level, 5-fold averaged)")
aa = pd.read_csv("output/aa_cohort2_validation.csv")
s = aa.groupby("sample").agg(AA=("AA", "mean"), grp=("group", "first"))
rll = s[s.grp == "RLL"]["AA"]; ryc = s[s.grp == "RYC"]["AA"]
d = (rll.mean() - ryc.mean()) / np.sqrt((rll.std() ** 2 + ryc.std() ** 2) / 2)
t, p = stats.ttest_ind(rll, ryc, equal_var=False)
print(f"d={d:.3f} (paper 1.39) | delta={rll.mean()-ryc.mean():.2f} (paper -17.89)")
print(f"Welch p={p:.2e} (paper 5.9e-06) | RLL mean={rll.mean():.2f} (paper -3.11)")
print(f"RYC mean={ryc.mean():.2f} (paper +14.78) | neg frac={(rll<0).mean():.3f} (paper 0.707)")
