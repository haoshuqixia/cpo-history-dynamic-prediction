# Reporting crosswalk — sealed local release, gate V1.2

The statistical runner executes original relative-layout modules in private
work. The numbered repository folders are an index, not new analytical stages.

| Reported output | Private input / producing module | Verification |
|---|---|---|
| Main Table 1 | Stage 1A LANDMARK_FEATURES_V1; adult_stays; final eICU R1 predictions/features; reproduce_main_table_cells.py | 72/72 manuscript numeric cells match |
| Main Table 2 | Stage 1B temporal_metrics/paired_temporal_deltas; final eICU R1 absolute_metrics/primary_comparison_ci | 36/36 cells match the corrected manuscript; prior eight differences resolved |
| Main Figure 1 | Locked cohort-flow constants in gen_figure2_cohort_flow.py; landmark/outcome denominators | Final eICU 7,753 printed; source data compared with the previous final-R1 export |
| Main Figure 2 | Stage 1B temporal predictions, metrics, bootstrap intervals and decision curves | Regenerated from relocated results |
| Main Figure 3 | Stage 1A features; Stage 1B predictions; raw derived CPO; Stage 1D alert details | Same illustrative-subset rules; source CSVs numerically equivalent |
| Main Figure 4 | Final R1 eICU predictions, absolute metrics, intervals and hospital summaries | Regenerated from relocated final R1, not baseline E2 |
| Graphical Abstract | locked/graphical_abstract.py + ga_base.py; frozen temporal ROC and final R1 Brier/paired CI | Approved V5; exact PNG reproduction; 0.04882/0.04747 preserved |
| Supplementary Tables S1–S8 | build_submission_supplement_r1.py make_table_s1–s8; variable dictionary, weights, sensitivities, hospital/alert/outcome results | 580/580 table cells match the located R1 Supplement; includes source/definition contracts; S8 counts independently checked |
| Supplementary Figures S1–S3 | build_submission_supplement_r1.py + locked/locked_style.py; weights, paired sensitivity results and C−B comparisons | Approved reference style V1; uppercase panel letters; exact PNG reproduction |

The final main figures and table image previews use locked/locked_style.py.
All 10 PNG outputs exactly match the approved file hashes. Legacy
generate_ehjacc_figures.py is only a private source-data preparation dependency,
not the final submission rendering endpoint.

The main-table summary functions were projected from the existing
build_main_tables_docx.py generator. Changes were restricted to input paths,
the final eICU R1 denominator assertion (7,753), two current-availability row
labels and CSV/read-only comparison output. All descriptive arithmetic and
the existing patient-row selection conventions were retained.

The eICU first patient row is sorted by unit-stay identifier (sid) and then
landmark hour. This deterministic convention must not be described as
verified chronological ordering across multiple ICU stays without further
evidence. No alternative patient-selection analysis was introduced here.

Reference result files, private figure source-data files (including selected
patients), patient-level intermediates and trained models are not bundled.
The sealed local package contains code, reviewed metadata and aggregate QA only.

The final main-manuscript basis is Manuscript_numbers_corrected.docx, SHA-256
d62f3b82c3555dd0628f0458d1d4b36cd94ccfdffea9d81cdbd97b889b2292cb.
Its six prose-only metric corrections do not alter tables, image content,
styles, formulas or other document parts. Main Figures 1–4 have exact RGB
pixel identity to the fresh locked outputs; the smaller embedded GA was
visually checked for correct R1 labels. The CSV crosswalk explicitly separates
numerical PASS from pending repository identifiers and editorial notes.
