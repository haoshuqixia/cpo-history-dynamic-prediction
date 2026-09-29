# Author-approved figure/table lock — 2026-09-29

The author's instruction was to lock the current figures/tables and update
the local GitHub code candidate. Scope: **10 outputs** — Graphical Abstract V5,
main Figures 1–4, supplementary Figures S1–S3 and Tables 1–2 from reference
style V1. The superseded V4 illustrated GA and the AI editorial preview are
not submission outputs and are not in this lock or code package.

`FIGURE_LOCK.json` contains the approved PNG, PDF and SVG SHA-256 identities.
The separate local image archive contains the approved files under matching
names. This code archive contains only source, metadata and QA; it contains
no patient-level prediction, history, observation-weight or selected-case file.

## Reporting path

`run_workflow.py --report` prepares frozen reporting inputs using the retained
legacy generators, then calls `regenerate_final_figures.py`. Alternatively,
call the latter directly once all private reporting inputs exist. It launches:

1. `10_figures_tables/locked/graphical_abstract.py`: reviewed V5 wording/layout.
2. `10_figures_tables/locked/ga_base.py`: unchanged frozen ROC/Brier draw logic.
3. `10_figures_tables/locked/locked_style.py`: main, supplementary and table style.

Legacy numeric draw functions are taken from the bundled code as staged into
the private work directory. Their file hashes must match the repository. No
imports depend on the author's original V1/V2/V5 design folders.

The sole final output naming convention is `Graphical_Abstract`, `Figure_1`
through `Figure_4`, `Figure_S1` through `Figure_S3`, `Table_1`, `Table_2`.
PNG exports are 600 dpi; PDF/SVG are vector. Statistical panels retain their
approved plotting data. Main and supplementary panel letters are uppercase.
Tables are image previews, not substitutes for editable manuscript tables.

## Exact rendering versus scientific reproducibility

The local reporting endpoint reproduced **10/10 PNGs byte-for-byte** against
the approved files using the recorded software/fonts. A different font or
library can change pixels without changing estimates; do not automatically
relock such output. `--check-lock` requires exact approved PNG hashes.
PDF/SVG generation dates can differ, so regenerated vector-file byte equality
is not claimed. Original approved vector files retain their recorded hashes.

This update does not rerun models or bootstrap. The numerical and visual
source values, including final R1 Brier 0.04882/0.04747 and 7,753 landmarks,
were retained. The 25 statistical modules in `ANALYSIS_CODE_LOCK.json` match
V1.0 exactly. The V1.2 release gate additionally reran the extracted V1.1 code
from raw inputs in a new private tree: 27 stages and 85 reference files passed.
All 10 approved PNG identities were reproduced from those fresh results.

Figure locking alone does not close licence or publication gates. The separate
V1.2 code-release audit records author-approved MIT and corrected-main numeric
checks. No GitHub/Zenodo upload or overall manuscript-compliance approval is
claimed by the figure lock itself. Word embedded GA remains a smaller preview;
use the separate locked native PNG/vector files for image submission.
