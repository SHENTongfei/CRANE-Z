# CRANE-Z

**Cross-cohort Robust transcriptome Aging/immunity discriminator** — a dual-modality deep classifier for extreme longevity phenotypes from peripheral blood transcriptomes with immune deconvolution.

CRANE-Z combines a **gene-module Transformer** with a **FiLM-conditioned immune branch** and **mask-based feature pretraining**, anchored on a linear age prior through bounded residual learning. It is evaluated with 5-fold × 5-seed cross-validation on a Chinese longevity cohort (1,715 blood transcriptomes) and validated across **two fully independent external cohorts**:

| Validation set | Population | Platform | Result |
|---|---|---|---|
| GTEx whole blood (n=64) | Western | RNA-seq | direction-consistent AA replication (p=0.001) |
| GTEx skeletal muscle (n=58) | Western | RNA-seq | cross-tissue generalization (external OPS 3.891) |
| GSE123696 whole blood (n=66) | Israeli | Affymetrix PrimeView microarray | RLL appear **17.9 y biologically younger** than RYC (Welch p=5.9e-6, Cohen d=1.39, 5/5 fold direction-consistent) |

## Key findings

- **Internal**: long-lived individuals (≥70 y) are on average **8.1 y biologically younger** than their chronological age (group separation 14.69 y, Cohen d=1.18, p<0.0001); model age ranks individuals with **Spearman ρ=0.816** vs chronological age.
- **External replication (three-cohort, two-ethnicity, two-platform closed loop)**: the decelerated biological-age signal replicates on Western RNA-seq (GTEx) and Israeli microarray (GSE123696) blood.
- **Cross-cohort immune signature**: resting NK elevation and naive B reduction (GTEx p=0.01), mirrored at gene level by *LRRN3*, *CR2* and *SIGLEC14*.
- **Sex dimension**: females are biologically younger than males on average (p=0.029).

## Data sources

| Data | Source | Access |
|---|---|---|
| Chinese longevity cohort, peripheral blood transcriptomes (1,715 samples) | Kong Qingpeng group, Hainan/Hubei/Hunan longevity cohorts | Collaborative sharing; contact corresponding author |
| GTEx whole blood / skeletal muscle (RYC 20-49 / RLL 70-79) | [GTEx Portal v10](https://gtexportal.org/) | Public, GTEx terms |
| Israeli whole blood, PrimeView microarray (n=66, ages 23-96) | GSE123696 — Alpert et al., *Nature Medicine* 2019, [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123696) | Public |

> **Data placement**: by default the code expects data under `../data` relative to this repository (i.e. a sibling `data/` directory), or point to it explicitly:
>
> ```bash
> export CRANE_DATA_DIR=/path/to/your/data
> ```
>
> Required layout: `expression_matrix.tsv.gz`, `phenotype.csv`, `LM22_signature.tsv`, `training_data/balanced_folds/`, `external_test_data/gtex_validation_balanced.csv`, `external_test_data/gtex_validation_muscle.csv`. Cohort2 (GSE123696) preparation is in `data_prep/prep_cohort2.py` (download the series matrix + the full PrimeView ADF annotation from ArrayExpress first).

## Reproduce

```bash
# 1. environment (Windows: conda env py311; CUDA GPU recommended)
pip install -r requirements.txt

# 2. train pretrained + 5-fold x 5-seed + ensemble (~25 min on RTX 5080)
python train_v2.py --all

# 3. fair baselines (CLC-S/KNN/DT/Ridge/Lasso, incl. external)
python train_v2.py --baselines

# 4. PERF-GATE verdict
python stats_v2.py

# 5. bootstrap 95% CI / DeLong external AUC
python stats_bootstrap.py --n 1000
python stats_delong.py

# 6. aging-acceleration (AA) analyses
python analysis/aa_internal.py            # internal 5-fold AA by group
python analysis/aa_sex.py                 # sex-stratified AA
python analysis/aa_age_acceleration.py    # GTEx blood AA (rank-calibrated)
python analysis/gtex_aa.py                # GTEx blood per-sample AA (panel G)
python analysis/cohort2_eval.py           # GSE123696 AA validation (all stats)

# 7. downstream biomarker mining / interpretability
python analysis/downstream_biomarkers.py
python analysis/interpret_deep.py

# 8. manuscript figures
python figures/make_figs.py
```

## Leakage safeguards (as in Methods)

1. HVG top-3000 fitted on fold-0 train only; reused across folds; never touches validation/external data.
2. QuantileTransformer (rank-gauss) and StandardScaler (z-score) fitted per train fold only.
3. Ridge meta-feature: per-fold Ridge fitted on train → predicted on val/external; external predicted exactly once.
4. MFM pretraining excludes the union of all 5-fold validation samples.
5. External sets used exactly once at final evaluation; no tuning on them.
6. All 5 fixed seeds (42/2024/2025/7/12345) reported; no seed cherry-picking.
7. Protocol changes (e.g., primary-metric specification) are logged in `protocol_lock.json` for transparency.

## Repository layout

```
├── common.py                    # data loading, HVG, modules, LM22 deconv, OPS
├── model_v2.py                  # CRANE-Z architecture (Transformer + FiLM + MFM + dual heads)
├── train_v2.py                  # training pipeline (pretrain, finetune, ensemble, baselines)
├── stats_v2.py                  # PERF-GATE five-check verdict
├── stats_bootstrap.py           # bootstrap 95% CIs
├── stats_delong.py              # DeLong external AUC comparison
├── audit_probe_gate1.py         # hostile probes (leakage, balance, age-adjusted immune)
├── protocol_lock.json           # protocol-change audit trail
├── analysis/                    # AA, sex, biomarker and interpretability analyses
├── figures/                     # manuscript figure generation
└── data_prep/                   # Cohort2 (GSE123696) preparation with ECDF calibration
```

## Citation

If you use CRANE-Z in your research, please cite:

> Shen, T. *et al.* CRANE-Z: Cross-cohort robust transcriptome aging/immunity discrimination of extreme longevity phenotypes. *(manuscript in preparation)*

## License

MIT — see [LICENSE](LICENSE). Data licenses of the underlying cohorts apply separately.
