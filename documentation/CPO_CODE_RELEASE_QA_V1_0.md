# CPO code release QA V1.0 — relocated reproduction, 2026-09-29

**Historical V1.0 audit, retained for provenance. Its open figure-layout items
are superseded by the separately verified V1.1 locked reporting endpoint.
This document is not the current release status. Current licence, numeric
crosswalk and seal gates are in CPO_CODE_RELEASE_QA_V1_2.md / RELEASE_STATUS.json.**

Status: **REPRODUCIBILITY_PASS / LOCAL_CANDIDATE_NOT_SEALED / NOT_PUBLISHED**. Not an overall RELEASE PASS.

Scope: fresh restricted-source-data rerun, frozen-result equivalence, fold/history-clock audit, main-table cross-check and figure technical/visual review. Statistical definitions were not changed.

| Status | Check |
|---|---|
| PASS | 35 allowlisted analysis/reporting files; function ASTs unchanged from sources |
| PASS | 39 Python files compile; no personal absolute source paths or detected token/private-key literals |
| PASS | A/B/C blocks match the locked manifest; nested blocks and key hyperparameters verified |
| PASS | Synthetic-only tests: (L−4,L] predictor window and 0.60 W / 0.5–3 h confirmation boundaries |
| PASS | Private frozen R1 outputs verified in place: 7,753 landmarks / 840 patients / 27 hospitals / 469 event landmarks; no rows exported |
| PASS | 43 predictor definitions/units retained; cohort missingness columns excluded |
| PASS | Archive allowlist: code/documentation/variable definitions only; no datasets, model artifacts or patient-level CSV/log files |
| PASS | Confirmed title/author order written to README and CITATION.cff; YAML syntax validated; no equal-contribution or unapproved identifier/licence |
| PASS | 27/27 relocated raw-data analysis stages completed in a fresh private directory |
| PASS | 85/85 numerical output CSVs match frozen MIMIC and final eICU R1 references (rtol 1e-10 / atol 1e-12; integer/string values exact) |
| PASS | MIMIC patient-grouped clinical/observation folds and eICU hospital-grouped observation folds have no group overlap; clinical preprocessing/calibration source boundaries checked |
| PASS | 920,360 MIMIC and 253,784 eICU history-summary cells independently reconstructed within the specified measurement/availability clocks; charttime is not claimed to prove real-time availability |
| PASS | 72/72 Table 1 numeric cells match the current manuscript; manuscript was not edited |
| PASS | Graphical Abstract + main Figures 1-4 + supplementary Figures S1-S3 regenerated; 8 vector PDFs with embedded fonts and no raster objects; main PNG/TIFF exports at 600 dpi |
| OPEN | Table 2 has eight external-validation reporting differences; differences recorded, manuscript unchanged at user request |
| OPEN | Graphical Abstract Model A text approaches box border; supplement panel lettering needs EHJ alignment; frozen plotting scripts not redesigned |
| OPEN | eICU descriptive first-patient row uses sid/landmark order, not verified cross-stay chronology; source convention retained and disclosed |
| OPEN | Software licence/copyright approval; optional Word-output environment and final publication compliance not sealed |
| NOT_PUBLISHED | GitHub repository / Zenodo DOI; manuscript availability placeholder unchanged |

Privacy finding: no raw or patient-level files or learned models were copied. Automated token/path checks are a preflight, not a guarantee that a final publication review can be skipped.

Both original-freeze and verified-rerun environments are recorded. The draft requirements use the verified rerun environment; no bitwise-equivalence claim is made for a new run.

Source R1 is SEALED; this newly organized release candidate is not. Licence remains pending at the author's request. Table 2 reporting differences and figure layout/compliance items remain open. No public repository or DOI was created.

Independent slope arithmetic used absolute tolerance 1e-10 only for slopes: one constant SvO2 series produces a ~9.9e-12 polyfit cancellation artifact versus zero from closed-form OLS. Frozen-output comparison tolerances were not relaxed.

The academic-plotting workflow kept the existing figures and generated numerical plots from relocated outputs; PDF review identified the remaining layout/lettering items. The code/data availability workflow kept restricted outputs and trained models outside the repository.
