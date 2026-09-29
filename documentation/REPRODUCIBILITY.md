# Reproducibility notes — software v1.0.0, release gate V1.2

## Provenance

MIMIC code is selected from the 2026-09-15/16 complete numerical rerun.
Shared extraction comes from its Stage −1A dependency. External-validation
modules come from the frozen eICU chain and the final sealed
`CPO_EICU_EXTERNAL_DISCHARGE_BOUNDARY_R1_V1_0/run_eicu_r1.py`.
Legacy numeric drawing functions use the 2026-09-28 vector/600-dpi code.
The final reporting endpoint now applies the approved reference-style V1
to main/supplementary figures and table images, and reviewed V5 to the GA.
See FIGURE_LOCK_README.md and FIGURE_LOCK.json. No private result files are bundled.

## Why original module names are retained

Preserving module names prevents silently changing imports, file contracts,
model feature definitions and the order of transformations. Numbered folders
are a browsable index. `run_workflow.py --prepare` reassembles those modules
in a private work tree with their original relative dependencies.

`--execute` runs the analysis; `--resume` uses private stage checkpoints.
`--report --manuscript PATH` prepares legacy plot sources, regenerates table cells,
then generates locked figures in a new private output directory. It
reads the specified Word manuscript for a numerical comparison. It never
edits the manuscript. Only the final discharge-boundary R1 outputs are
used for external-validation reporting. Outputs remain in private work.

The CSV main-table adapter projects the existing main-table summary functions.
Its only changes are final R1 input locations, the 7,753 denominator assertion,
and the two mPAP/SvO2 row labels used by the current manuscript. It preserves
the existing descriptive selection order: actual landmark datetime for
MIMIC patients; unit-stay identifier followed by landmark hour for eICU.
The latter is deterministic ordering, not proof of chronology across separate
eICU ICU stays. No new subgroup or inferential analysis is introduced.

The full MIMIC extraction module was initially part of a broader observability
audit. It is retained because the CPO pipeline consumes its cleaned/pairing
outputs. Its broader extraction/aggregation routines have not been newly
reduced or rewritten in this release.

## Relocation changes

Only full local MIMIC/eICU source-path literals were mechanically converted
to `CPO_MIMIC_DIR`/`CPO_EICU_ZIP` environment variables. A stdlib `os` import
was added where needed. No statistical hyperparameters, predictor lists,
outcome rules, bootstrap sizes or plotting values were changed.
The local source manifest records before/after SHA-256 hashes.

## External R1 endpoint

The baseline eICU E0 outputs are intermediate dependencies of the final R1
runner. Their old 7,754 count is not a final reported result. R1 rebuilds labels
using CPO timestamps at/before `unitDischargeOffset`, observation weights and
frozen-model performance. The final observed denominator must be 7,753.
Do not call only `run_eicu_stage_e2.py` and claim final reproduction.

The local final R1 labels and predictions are available only in the private
research workspace. They are deliberately excluded from the code archive.
The trained MIMIC models are also excluded: authorized users must regenerate
them locally before external validation. There is no eICU refit/recalibration.

## Checks completed and not completed — 2026-09-29

Completed: code allowlisting, syntax checks, personal source-path removal,
synthetic definition tests, nested model/block comparisons, a fresh 27-stage
restricted-source-data run, 85-file numerical equivalence, patient/hospital
fold separation, independent MIMIC/eICU history-summary arithmetic, all 72
Table 1 numeric cells, 12 figure-source CSV comparisons, vector/embedded-font
and 600-dpi checks, citation metadata, archive content audit and file hashes.

V1.1 completed: approved 10-output figure/table lock, updated portable reporting
entry point, exact 10/10 approved PNG reproduction and unchanged checksums
for 25 statistical modules. Statistical source data were not reanalysed.

V1.2 completed: the V1.1 code-only ZIP was newly extracted; its complete 27-stage
raw-data run and 85-file numerical equivalence passed again without old output
or trained-model copies. Grouped-fold and history-clock reconstructions passed.
All 10 locked PNGs matched exactly. Current corrected manuscript Table 1/2
matches 72/72 and 36/36 cells. Supplement S1–S8 matches 580 cells, including
definition contracts; S8 count rows were independently checked against rerun
counts. Six stale prose labels were corrected in a separate Word copy, without
changing statistics, images, styles or the author's original Word file.

The author confirmed public rights and the MIT software licence with copyright
held by the six-person author team. After documentation/licence updates, fresh
path/credential/data-file checks and complete checksums were regenerated.
The code-only payload is SEALED and the author uploaded it to
https://github.com/haoshuqixia/cpo-history-dynamic-prediction.
Anonymous access and all 74 original file identities were verified on 2026-09-29.
The formal v1.0.0 tag/release is pending and no Zenodo DOI is assigned.
See RELEASE_STATUS.json and PUBLIC_METADATA_QA_V1_0.md for current status;
CPO_CODE_RELEASE_QA_V1_2.md preserves the historical local-seal evidence.

The statistical environment does not provide optional legacy editable-Word
generation; summary CSVs plus PNG/PDF/SVG are the verified reproduction endpoint.
Document title/availability wording and embedded Supplement previews still have
editorial review notes. Those are manuscript-only issues, not unresolved code
numerical gates, and the package does not claim entire submission compliance.

## Sensitive outputs

Running analysis code creates patient/landmark-level intermediate files in the
private work tree. That is necessary for authorized local reproduction, not
permission to upload them. Do not copy private_work, work, results, audit,
models, trajectories, bootstrap/prediction files or logs to GitHub.

## Publication wording

Anonymous access to the actual repository URL has been verified; manuscript
Code availability may use that URL. Do not claim a formal v1.0.0 release or DOI
until it exists and is checked. The software licence does not grant permission
to redistribute restricted datasets. Local preflight checks are not a live
remote verification: check the metadata-upload commit before tagging.
