# Recent CPO history and short-term low-CPO prediction

**Software v1.0.0 — LOCAL SEALED RELEASE — MIT — NOT YET UPLOADED.**

Release-gate revision: V1.2 (2026-09-29). V1.0/V1.1 were local draft revisions,
not public software tags. See `documentation/RELEASE_STATUS.json` and
`documentation/CPO_CODE_RELEASE_QA_V1_2.md` for the final evidence and scope.

Code supporting the manuscript prepared for European Heart Journal – Acute
Cardiovascular Care:

*Recent cardiac power output history improves six-hour prediction of low-CPO
episodes during pulmonary artery catheter monitoring*

Authors: Zhihao Lin; Tangjiang Wan; Wenyue Lv; Shitong Shen; Luyang Wu;
Yafei Li. Corresponding author: Yafei Li. No equal-contribution designation.

## What is locked

- Graphical Abstract: reviewed blue/navy/gold V5 with real frozen ROC/Brier results.
- Main Figures 1–4, Supplementary Figures S1–S3 and Tables 1–2: reference-style V1.
- External endpoint: final discharge-boundary R1, **7,753 landmarks**, 840 patients,
  27 hospitals. External Brier labels: **0.04882 / 0.04747**, not the superseded values.
- The 25 core statistical modules and all 10_figures_tables drawing modules are
  byte-for-byte unchanged from the successfully relocated V1.1 code candidate.
  This final update changes only licence, release documentation and QA metadata.

`documentation/FIGURE_LOCK.json` records the 10 approved output identities
(PNG/PDF/SVG), nominal raster sizes and provenance. Approved images are in the
separate local figure package, **not in this code-only repository**. Private
predictions and selected landmark traces must never be uploaded to GitHub.

## Data and software

MIMIC-IV v3.1 and eICU Collaborative Research Database v2.0 require credentialed
PhysioNet access and compliance with their data-use agreements. No source,
patient-level or landmark-level data, trained models, credentials or private
execution logs are distributed here. Authorized users regenerate models locally.

Verified analysis environment: Python 3.12.14, scientific package versions in
`requirements.txt`. Reporting also requires `requirements-reporting.txt`.
Arial was used for the local figure identity test; Helvetica/DejaVu Sans are
fallbacks but may change text geometry and PNG hashes. PDF timestamps can vary.

## Reproduce the locked figures in VS Code

After a completed private analysis and reporting-source preparation, open this
repository in VS Code and select the configured Python environment. Run:

```bash
python regenerate_final_figures.py --plan
python regenerate_final_figures.py \
  --work-dir /path/to/private/completed-analysis \
  --output-dir /path/to/new-private-figure-output \
  --check-lock
```

Replace both example paths with your private directories. The output directory
must be new and outside this repository. Outputs are `png/` (600 dpi), `vector/`
(PDF/SVG), `previews/`, and `LOCK_REPRODUCTION_QA.json`. No training,
recalibration, bootstrap or manuscript edits occur in this reporting command.

The locked renderer uses the **bundled** unchanged draw functions via a private
code tree assembled by `run_workflow.py --prepare`; it does not require the old
V1/V2/V5 design folders. Do not call the legacy figure generator as the final
submission endpoint. It is retained only to prepare private plot-source data.

## Full private reproduction

1. Set `CPO_MIMIC_DIR`, `CPO_EICU_ZIP` and a new `CPO_WORK_DIR` outside this repo.
2. Inspect `python run_workflow.py --plan` (read-only).
3. Run `python run_workflow.py --prepare`, then `python run_workflow.py --execute`.
4. Run `python run_workflow.py --report`. This prepares the legacy private
   figure-source CSVs, recomputes main-table display cells from frozen outputs,
   then generates the locked figures into a NEW private output folder.
5. Use `validate_reproduction.py` with authorized private reference outputs to
   check numerical equivalence. References are not bundled.

`--resume` uses matching successful private execution checkpoints.
`--report --manuscript PATH` reads that Word file for comparison only; no editing.
The V1.2 gate extracted the V1.1 code-only ZIP into a new directory and executed
all 27 stages from raw, explicitly supplied data. No old results, trained models
or notebook cache were copied. All 85 frozen-reference CSV comparisons passed;
floats use rtol=1e-10/atol=1e-12, while integer/string checks are separate.
Patient/hospital fold separation and independently reconstructed CPO histories
also passed. The locked reporting endpoint reproduced all 10 approved PNGs
exactly in the recorded environment. Licence/metadata edits after that rerun
did not change any analytical or drawing module.

The private audit initially denied macOS timezone resources and the installed
rg executable. Those runtime-whitelist setup events and the failed eICU attempt
were preserved; execution resumed after allowing those runtime resources.
Python file-open and absolute subprocess-argument auditing is not universal
OS/C syscall tracing. No old-project dependency was observed within that scope.

The analysis-only environment does not generate editable Word documents.
Numeric summary CSVs and locked PNG/PDF/SVG outputs are the verified report
endpoints. The optional legacy Supplement Word generator needs a separately
provisioned python-docx environment; it is not needed to reproduce statistics.

## Frozen scientific definitions

- Model A: latest CPO plus shared clinical/observation context, including
  measurement counts/recencies. Those are not the physiological history block.
- Model B: Model A plus recent 4-h CPO mean, minimum, maximum, SD, slope,
  first-to-last change and span over `(L−4 h, L]`.
- Model C: Model B plus latest and 4-h mPAP/SvO2 summaries; secondary comparison.
- Horizon `(L, L+6 h]`; confirmed/repeated low CPO: two CPO values <0.60 W,
  separated by 0.5–3 h. Dense-negative rule: ≥4 future measurements, last ≥L+5 h,
  maximum gap ≤2 h including current-to-future gap.
- Charttime primary; storetime strict sensitivity. Primary charttime is not
  proven real-time availability. Final eICU uses discharge-boundary R1.
- External models are frozen: no eICU retraining or recalibration.

## Licence and release scope

Author-approved MIT License; copyright (c) 2026 the CPO study author team:
Zhihao Lin, Tangjiang Wan, Wenyue Lv, Shitong Shen, Luyang Wu, and Yafei Li.
See `LICENSE`. Dependencies retain their own licences. This code licence does
not license or distribute MIMIC/eICU source data, patient-level derivatives,
trained models, or third-party database rights.

Local technical/publication-payload gates passed. The corrected main manuscript
matches 72/72 Table 1 and 36/36 Table 2 numeric cells; six stale prose metric
labels were corrected in a separate Word copy, not by the statistical pipeline.
Supplementary Tables S1–S8 were checked against fresh outputs (580 table cells,
including mapping/definition contracts). Supplement title/availability wording
and embedded-preview editorial items remain manuscript-only review notes, not
changes to the sealed code. The publication checklist separates these scopes.

eICU descriptive patient rows use deterministic sid/landmark ordering; sid is
not evidence of chronological order across different ICU stays. This documented
convention was preserved, not silently changed.

Only code, metadata, variable definitions and aggregate QA are included. No
patient-level, landmark-level or selected-case source-data files, credentials,
logs, notebook cache, database dump or trained-model files are included.
Automated safety scans are scoped preflight, not a guarantee of detecting
every possible secret or a substitute for author rights confirmation.

`SHA256SUMS.txt` and `CPO_CODE_RELEASE_SHA256_V1_2.txt` are identical and cover
every payload file except those two checksum lists. The manifest omits itself
and the two checksum lists to avoid circular hashing; its own hash is included
in both checksum lists. Prior draft QA is marked historical and superseded.

## Code availability

Sealed locally and ready for a code-only upload, but **not yet publicly available**.
No GitHub repository/URL, Git tag or Zenodo DOI has been created or verified by
this task. Use v1.0.0 for the first public software release. Add the real public
URL/DOI to the manuscript only after upload and independent access verification.
