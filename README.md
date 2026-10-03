# Recent CPO history and short-term low-CPO prediction

**Software version 1.0.0 — PUBLIC CODE RELEASE AVAILABLE — MIT.**

Repository: https://github.com/haoshuqixia/cpo-history-dynamic-prediction

Public release **v1.0.0** is available:
https://github.com/haoshuqixia/cpo-history-dynamic-prediction/releases/tag/v1.0.0

The existing v1.0.0 tag and release are retained as the historical code snapshot.
The main branch includes manuscript-title and publication-metadata updates for
the Scientific Reports submission.
These publication-metadata updates do not alter the frozen analysis code, model
specifications, statistical results, or the existing v1.0.0 release.

Release-gate revision: V1.2 (2026-09-29). V1.0/V1.1 were local draft revisions,
not public software tags. The V1.2 release-gate and pre-release metadata records
in `documentation/` are historical audit evidence. Statements in those records
that the tag/release was pending describe the pre-release state, not the current
public release status.

Code supporting the manuscript prepared for submission to the
**Scientific Reports**:

*Recent cardiac power output history improves 6-hour prediction of low-CPO
episodes during pulmonary artery catheter monitoring*

Authors: Zhihao Lin; Tangjiang Wan; Wenyue Lv; Shitong Shen; Luyang Wu;
Yafei Li. Corresponding author: Yafei Li. No equal-contribution designation.

## Historical locked analysis and reporting outputs

The output identities below describe the historical local reporting lock.
The current Scientific Reports submission documents are maintained separately; editorial
changes to their titles, legends, and checklist pagination do not change the
frozen statistical analysis.

- Graphical Abstract: reviewed blue/navy/gold V5 with real frozen ROC/Brier results.
- Main Figures 1–4, Supplementary Figures S1–S3 and Tables 1–2: reference-style V1.
- External endpoint: final discharge-boundary R1, **7,753 landmarks**, 840 patients,
  27 hospitals. External Brier labels: **0.04882 / 0.04747**, not the superseded values.
- The 25 core statistical modules and all 10_figures_tables drawing modules are
  byte-for-byte unchanged from the successfully relocated V1.1 code candidate.
  The V1.2 release-gate update changed only licence, release documentation and
  QA metadata; the present README revision updates publication metadata.

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
- Horizon `(L, L+6 h]`; low-CPO episode confirmed by repeated measurements:
  two CPO values <0.60 W,
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

The historical V1.2 local technical/publication-payload gates passed. The
manuscript checked in that audit
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

On the main branch, `CPO_CODE_RELEASE_MANIFEST_V1_2.csv`, `SHA256SUMS.txt`, and
`CPO_CODE_RELEASE_SHA256_V1_2.txt` retain the records generated for the earlier
publication-metadata candidate. Their legacy filenames are retained for validator
compatibility. Subsequent README-only publication-metadata updates are not
represented in those historical checksum records; they do not rerun or change
the historical V1.2 audit or the v1.0.0 tag. Use the unchanged v1.0.0 tag for the
frozen public code snapshot and its checksum records. This README update changes
publication metadata only; it does not constitute a new model run, manuscript
acceptance, or certification of the whole manuscript submission.

## Code availability

Code for cohort construction, CPO derivation, rolling landmark generation,
predictor and outcome construction, observation weighting, model development,
temporal and external validation, sensitivity analyses, and figure generation
is publicly available at https://github.com/haoshuqixia/cpo-history-dynamic-prediction.
The repository contains analysis code, variable definitions and reproducibility
instructions. Raw or patient-level MIMIC-IV and eICU data are not redistributed
in accordance with the relevant PhysioNet data-use agreements.

Public release **v1.0.0** is available at the release link above. The main branch
may contain subsequent publication-metadata edits; use the existing v1.0.0 tag
for the frozen public code snapshot. This README does not claim a Zenodo DOI or
that the manuscript has been accepted or published.
