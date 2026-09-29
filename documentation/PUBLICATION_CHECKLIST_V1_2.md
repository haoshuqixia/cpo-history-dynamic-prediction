# Release checklist — v1.0.0, audit V1.2

## Code-only local release gates

- [x] Author-confirmed rights and MIT software licence; author-team copyright.
- [x] Six authors, title and corresponding author preserved; no equal contribution invented.
- [x] Newly extracted code-only ZIP; 27/27 raw-data stages completed.
- [x] 85/85 frozen numerical-reference files match.
- [x] Patient/hospital grouped folds and independent history-clock reconstructions pass.
- [x] External final discharge-boundary R1: 7,753 landmarks, 840 patients, 27 hospitals, 469 event landmarks, 157 event patients.
- [x] Main Table 1: 72/72; Main Table 2: 36/36 corrected manuscript numeric cells match.
- [x] Six stale prose metric labels corrected in a separate Word copy; no statistics or table arithmetic changed.
- [x] Supplement S1–S8: 580 table cells match; source/definition contracts distinguished from numerical estimates.
- [x] All 10 locked figure/table PNGs reproduced exactly; independent main-figure RGB check passed.
- [x] All analytical/drawing Python files except the release-preflight validator unchanged from the rerun candidate.
- [x] Private raw data, patient/landmark-level outputs, selected histories, models, credentials, logs and caches excluded.
- [x] eICU deterministic sid ordering explicitly documented, not claimed as verified chronology.
- [x] Numeric CSV and locked PNG/PDF/SVG endpoints verified; optional legacy Word generation out of analysis-runtime scope.

Final automated payload preflight and regenerated manifest/checksums are recorded
in RELEASE_STATUS.json and CPO_CODE_RELEASE_QA_V1_2.md. The final seal covers
the code-only payload. Current publication evidence is separately recorded in
PUBLIC_METADATA_QA_V1_0.md; the original V1.2 QA remains a historical snapshot.

## Actions after local sealing

- [x] Author uploaded the public GitHub repository; anonymous access and 74/74 original payload file identities verified on 2026-09-29.
- [x] Prepare actual repository URL in README/CITATION and synchronized publication-state metadata; no DOI invented.
- [ ] Upload this post-upload metadata update and verify the new remote file hashes.
- [ ] Publish the first software tag/release v1.0.0 and verify public access.
- [ ] Add the verified repository URL to manuscript Code availability; this task does not edit Word.
- [ ] Optional Zenodo archive and DOI, only after actual assignment.
- [ ] Review Supplement title/source-data wording and embedded-preview editorial notes.

No new model, subgroup, figure design or sensitivity analysis is required by
these remaining administrative/editorial actions. Updating publication metadata
after upload requires a new checksum record; do not silently overwrite a sealed
payload while retaining its old hashes.
