# CRANE-Z — Top-journal Format Adaptation Guide

This file accompanies the manuscript and the v29 PDF in the top-journal
submission package. It tells the corresponding author exactly how to
re-format the manuscript from the current OUP/Bioinformatics template
to either Nature Aging or Genome Biology, the two recommended targets
that the CRANE-Z scope fits best.

## 1. Recommended target journals (in order of fit)

1. **Genome Biology** (Springer Nature, BMC family, IF ~13-14, open access)
   - Scope: methods + biology findings. CRANE-Z is both.
   - Hard requirements we already meet: code availability at a public
     repository, full data-source statement, reproducibility.
   - Editorial timeline: ~8-12 weeks to first decision.
2. **Nature Aging** (Nature Portfolio, IF ~16-17)
   - Strongest thematic match (longevity), strict Nature formatting.
   - Hard requirements: code/data availability, referee-suggested
     3 reviewers, conflict-of-interest statements, graphical abstract.
3. **Nature Communications** (IF ~14-16) — backup; format identical
   to Nature Aging.

## 2. What the OUP/Bioinformatics template already gives us (no edit)

- Numeric citations in square brackets `[1]`, ordered by first appearance.
- Vancouver-style numeric bibliography via `\bibliographystyle{unsrtnat}`.
- Abstract with structured motivation/results/availability/contact blocks.
- `\corresp` and `\authormark` for corresponding author and short author
  label.
- `\received/\revised/\accepted` placeholder dates.

## 3. What to change for Genome Biology submission

Genome Biology uses the `bmc-art.cls` document class from Springer Nature.
The changes are mechanical and can be done in a single pass.

1. Swap document class:
   - Replace
     `\documentclass[unnumsec,webpdf,contemporary,large]{oup-authoring-template}`
     with
     `\documentclass[bmc, doublespacing]{bmc-art}`
   - Remove `\bibpunct{[}{]}{,}{n}{}{,}` (BMC uses numeric by default).
   - Remove `\graphicspath{{Fig/}}` (BMC expects `figures/` or
     embedded).
2. Move metadata into BMC macros:
   - `\title{CRANE-Z: ...}` (already present, keep as-is).
   - `\author[...]`, `\address[...]`, `\corresp[...]` already present;
     wrap each with `\bmcAuthor{...}` style if requested by the cls.
   - BMC expects `\bmcAuthNote{...}` for co-first authors.
3. Structured abstract:
   - BMC supports structured abstract; keep the current
     Motivation/Results/Availability/Contact labels verbatim.
4. Data and code availability:
   - Already done (`https://github.com/SHENTongfei/CRANE-Z`,
     MIT, full reproducibility).
5. References: convert from OUP bib to BMC bib style. Run BibTeX with
   `unsrtnat`; BMC cls accepts it.
6. Figure files: rename `Fig/` to `figures/` or add `\graphicspath{{Fig/}}`
   back. The current `.png` and `.pdf` pairs in `local_build/Fig/` and
   `Code_v2/figures_ms/` can be linked directly.

Estimated effort: half a working day. No content edits required.

## 4. What to change for Nature Aging submission

Nature uses the `sn-article.cls` / `sn-aag-template` document class. The
changes are more involved (Nature has stricter typesetting, graphical
abstract, structured referee suggestions).

1. Swap document class to `sn-article` with the ageing journal options.
2. Replace abstract with a single-paragraph 200-word abstract
   (Nature Aging style), without labels. Keep the motivation-results
   content; drop the "Availability/Contact" lines (move to
   `Data availability` and `Code availability` sections).
3. Add a graphical-abstract figure (Nature requirement). Re-purpose
   Fig9 panel G as the graphical abstract; annotate it with the key
   numbers (delta AA $-17.9$ years, p=5.9e-6, Cohen d=1.39).
4. Add 3-6 keywords (BMC-style, no periods).
5. Add a 30-word summary (Nature requirement, separate from abstract).
6. Add a "Letter-format" or "Article-format" choice at submission portal.
7. Author contribution statement already present; declare CRediT roles
   (Conceptualization, Methodology, Software, Validation, Formal
   analysis, Investigation, Writing - Original Draft, Writing - Review
   and Editing).
8. Competing interests already declared (none).
9. References: switch to Nature Vancouver style. `\bibliographystyle{myplain}`
   or the Nature `unsrtnat` variant (BibTeX style bundled in the
   `sn-article` package).

Estimated effort: one full working day for class swap, graphical
abstract, and referee suggestions. No content edits required.

## 5. Referee suggestions (recommended for both targets)

1. Professor of Computational Biology with deep-learning + immunology
   expertise (suggestion 1: someone with single-cell + aging-clock work).
2. Professor of Population Genetics of longevity (suggestion 2: someone
   with East-Asian longevity-cohort work).
3. Professor of Statistical Genetics or Biostatistics (suggestion 3:
   someone with multi-cohort replication methodology).

Replace the names with the corresponding author's known network.
Conflicts (same institution, recent co-author, family) must be
declared.

## 6. Suggested cover letter

The accompanying file `cover_letter_nature_aging.tex` is a ready-to-use
cover letter for the Nature Aging submission. Adapt the salutation
if submitting to Genome Biology or Nature Communications.

## 7. Pre-submission checklist (both targets)

- [ ] Code availability URL live and reproducible (already
      `https://github.com/SHENTongfei/CRANE-Z`).
- [ ] Data availability statement lists every public cohort used
      (GTEx Portal, GSE123696).
- [ ] Conflict of interest signed by every co-author.
- [ ] Author contribution statement using CRediT taxonomy.
- [ ] 3+ suggested referees, with institutional emails and 2-line
      rationale.
- [ ] Graphical abstract (Nature only).
- [ ] Highlights (Nature only, 3 bullet points, 85 characters each).
- [ ] ORCID for every author.
- [ ] Funding statement with grant numbers (already in manuscript).

## 8. Why this manuscript is competitive

- 1,715-sample internal cohort, 5x5-fold evaluation.
- Two external cohorts (GTEx and GSE123696), three-cohort replication
  across two ethnicities and two platforms.
- 19 FDR<0.05 pathways, 18 age-up, immune-effector dominated.
- Internal effect size: Cohen d=1.18, p<1e-22.
- External effect size: Cohen d=1.39, p=5.9e-6.
- Code fully public, MIT licensed, with reproducible pipeline.
- Honest reporting throughout (no cherry-picking, no fabricated p-values,
  protocol changes logged in `protocol_lock.json`).