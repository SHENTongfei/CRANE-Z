# -*- coding: utf-8 -*-
"""GSEA mechanism deepening (optimized: 100 perms, 4 threads)."""
import os, time
import numpy as np
import pandas as pd
import gseapy as gp

BASE = r"C:/Users/TS/Desktop/crane"
OUT = os.path.join(BASE, "Code_v2", "output")

libs = {
    "KEGG_2021_Human": "KEGG",
    "GO_Biological_Process_2021": "GOBP",
    "Reactome_2022": "Reactome",
    "WikiPathway_2021_Human": "WikiPath",
}
all_rows = []
for lib, tag in libs.items():
    t0 = time.time()
    try:
        gs = gp.get_library(name=lib, organism="Human")
        pre = gp.prerank(rnk=os.path.join(OUT, "gsea_ranked.rnk"),
                         gene_sets=gs, min_size=5, max_size=500,
                         permutation_num=100, outdir=None, seed=42,
                         no_plot=True, threads=4, format="tsv")
        res = pre.res2d
        if res is None or res.empty:
            print(f"[{tag}] EMPTY")
            continue
        res = res.copy()
        res["source"] = tag
        res["library"] = lib
        all_rows.append(res)
        print(f"[{tag}] {time.time()-t0:.0f}s rows={len(res)} FDR<0.25: {(res['FDR q-val']<0.25).sum()} FDR<0.05: {(res['FDR q-val']<0.05).sum()}")
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:100]}")

if all_rows:
    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(os.path.join(OUT, "gsea_results_all.csv"), index=False)
    imm_kw = ["T cell", "B cell", "NK", "natural killer", "cytokine", "interleukin",
              "immune", "chemokine", "inflammation", "lymphocyte", "leukocyte",
              "adaptive", "innate", "antigen", "CD4", "CD8", "memory", "naive",
              "killer", "neutrophil", "monocyte", "mast cell", "myeloid", "dendritic"]
    mask = full["Term"].str.lower().str.contains("|".join(imm_kw), na=False)
    imm = full[mask].sort_values("FDR q-val")
    imm.to_csv(os.path.join(OUT, "gsea_immune_pathways.csv"), index=False)
    print(f"\n=== immune-pathway hits FDR<0.25: {(imm['FDR q-val']<0.25).sum()} | FDR<0.05: {(imm['FDR q-val']<0.05).sum()} ===")
    show = imm[imm["FDR q-val"] < 0.25][["source", "Term", "NES", "FDR q-val"]].sort_values("NES", ascending=False)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}")[:4000])
    print("\nALL DONE -> gsea_results_all.csv + gsea_immune_pathways.csv")
else:
    print("NO RESULTS")
