# CPO code release V1.2 — software v1.0.0

Historical local-seal snapshot, before the author's GitHub upload. The verdict
and evidence below are preserved, not a statement of current public availability.
Current publication metadata: RELEASE_STATUS.json and PUBLIC_METADATA_QA_V1_0.md.

## Verdict

```ini
RELOCATED_END_TO_END_RERUN = PASS
NUMERICAL_FROZEN_REFERENCE_QA = PASS
MANUSCRIPT_NUMERICAL_CROSSWALK = PASS_CORRECTED_MAIN
LICENCE = MIT_AUTHOR_APPROVED
PRIVACY_CREDENTIAL_SCAN = PASS_AUTOMATED_CODE_ONLY_PREFLIGHT
RELEASE_MANIFEST = REGENERATED_COMPLETE_PAYLOAD
PUBLIC_RELEASE_READY = PASS_CODE_ONLY_PAYLOAD
CODE_RELEASE = SEALED_LOCAL_RELEASE
PUBLICLY_PUBLISHED = FALSE
```

## Evidence and unchanged code

- The V1.1 code ZIP was extracted into a fresh directory and executed from supplied raw MIMIC-IV/eICU sources: all 27 stages completed. No former outputs, model files or notebook cache copied.
- Frozen-reference numerical comparison: 85/85 files pass (rtol=1e-10, atol=1e-12 for floats; integer/string/missingness checks separate). Grouped folds and independent CPO history reconstructions pass.
- Final observed landmarks: development 21,619; temporal 3,416; external 7,753. External patients 840, stays 850, hospitals 27, event landmarks 469, event patients 157.
- Corrected main manuscript matches 72/72 Table 1 and 36/36 Table 2 numeric cells. Six prose labels were fixed separately without changing tables, pictures or formatting. The correct Word SHA256 is recorded in RELEASE_STATUS.json.
- Supplement S1–S8: 580/580 cells match; definitions/mapping are contracts, not newly estimated numbers. S8 count rows separately checked against rerun count outputs.
- 10/10 locked PNGs exactly reproduced. Embedded main Figures 1–4 have exact RGB identity. Embedded GA labels are correct; use its larger standalone native file for submission.
- All 43 original Python files except validate_locked_release.py are unchanged from V1.1; all 25 core statistical hashes match. No analysis, plotting, hyperparameter or bootstrap edit was made.
- Only licence, release documentation, aggregate QA metadata and the preflight validator changed after raw-data rerun. The validator now accepts and verifies LICENSE, final metadata and complete checksum inventory.
- Author-approved MIT with copyright held by the six authors. The code licence does not include restricted MIMIC/eICU data, patient-level derivatives or dependency licences. No independent legal opinion is claimed.

## Audit limits and historical events

- Eight preserved runtime-whitelist denied attempts involved macOS timezone resources or the installed rg executable, including the initial eICU extraction failure. The exact runtime exceptions were allowed and unchanged stages resumed; failure logs remain private.
- Python file-open/subprocess-argument auditing is not universal OS/C syscall tracing. No old-project dependency was observed within that scope.
- Automated extension/CSV allowlists, credential literals, personal paths, syntax and hashes are scoped preflight, not an exhaustive proof that every possible secret is absent.
- Optional legacy editable-Word generation is outside the verified scientific environment. Verified reproduction outputs are numeric CSV plus locked PNG/PDF/SVG; no new Word runtime was installed.

## Scope and next actions

- This seal is for the local code-only upload payload. No GitHub upload, Git tag, real public URL or Zenodo DOI was created. The first intended software release is v1.0.0.
- Public repository access must be checked after upload, then actual identifiers can be added to CITATION and manuscript availability statements.
- Supplement title/availability wording and preview-lettering are retained editorial notes, not unpassed code numerical gates. Whole-manuscript submission compliance is not certified.
- No source data, individual predictions, selected-case traces, model binaries, credentials, private logs, cache, figures or Word files are bundled.
- Historical V1.0/V1.1 reports are marked superseded. The former draft stays intact outside this new package.
- Manifest excludes itself and the two checksum lists to avoid circular hashes. Both identical checksum lists cover all non-checksum files, including the manifest.

MIT text source: https://opensource.org/license/mit
